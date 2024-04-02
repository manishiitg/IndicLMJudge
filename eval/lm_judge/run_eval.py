import argparse
import os
import random
import torch
import json
from datasets import load_dataset
from eval.utils import (
    dynamic_import_function,
)
from huggingface_hub import HfApi
from transformers import AutoTokenizer
import vllm
from datasets import Dataset
from datetime import date
import torch

# in this we simply save prompts outputs to a huggingface repo
# i using gemini pro (Free) as LM judge to rate the ouputs
# https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl


@torch.no_grad()
def eval_hf_model(args, model, tokenizer, prompts):
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=512,
        stop=["<|im_end|>"],
    )
    # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
    generations = model.generate(prompts, sampling_params)

    prompt_to_output = {
        g.prompt: g.outputs[0].text.strip() for g in generations
    }
    outputs = [prompt_to_output[prompt]
               if prompt in prompt_to_output else "" for prompt in prompts]

    return outputs


def main(args):
    random.seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    chat_formatting_function = dynamic_import_function(
        args.chat_formatting_function) if args.use_chat_format else None

    dataset = load_dataset("manishiitg/human_eval")
    test_data = dataset["train"]

    existing_data = {}
    api = HfApi()
    if api.repo_exists(repo_id=args.push_output, repo_type="dataset"):
        ds = load_dataset(args.push_output, split="train")
        for row in ds:
            if row["model_name"] == args.model_name_or_path:
                existing_data[row["prompt"]] = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)
    prompts = []
    simple_prompts = []

    mt_idx = {}

    default_system_en = "You are a helpful assistant."
    default_system_hi = "You are a helpful assistant."
    translate = {
        "दिए गए विषय पर एक ब्लॉग लिखें": "Write a blog on a given topic",
        "आप एक सहायक हैं। कृपया एक लंबा और विस्तृत उत्तर दें।": "You are a helper. Please give a long and detailed answer.",
    }

    processed_row = []
    idx = 0

    for i, example in enumerate(test_data):

        lang = example["lang"]
        system = default_system_en
        if lang == "hi":
            system = default_system_hi

        if example["type"] == "gpt4-multi-turn-hi" or "mt_bench-" in example["type"]:
            mt_idx[idx] = {
                "question": 0,
                "answer": idx,
            }
            prompt = example["mt_question"][mt_idx[idx]["question"]]
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
            example["messages"] = messages

        else:
            messages = json.loads(example["messages"])
            system = ""
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                    break

            if len(system) > 0 and system in translate:
                messages[0]["content"] = translate[system]

        if args.model_name_or_path == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            if lang != "en":
                messages[-1]["content"] = messages[-1]["content"] + \
                    "\n Respond only in hindi"

        if args.use_chat_format:
            prompt = chat_formatting_function(messages, tokenizer, args)
        else:
            prompt = "\n\n".join([x["content"] for x in prompt])

        simple_prompts.append("\n\n".join(
            [f"{x['role']} : {x['content']}" for x in messages]))
        prompts.append(prompt)
        processed_row.append(example)
        idx += 1

    if len(prompts) > 0:
        if args.awq:
            print("Loading model and tokenizer vllm awq...")
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                tokenizer_mode="auto",
                tensor_parallel_size=torch.cuda.device_count(),
                # max_num_batched_tokens=4096,
                quantization="AWQ",
                max_model_len=4096,
                dtype="float16",
            )
        else:
            print("Loading model and tokenizer vllm...")
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                tokenizer_mode="auto",
                tensor_parallel_size=torch.cuda.device_count(),
                # max_num_batched_tokens=4096,
                max_model_len=4096,
            )
        outputs = eval_hf_model(args, model, tokenizer, prompts)

        final_data = []
        with open(os.path.join(args.save_dir, f"lm_judge_predictions.jsonl"), "w") as fout:
            for example, output, simple_prompt, prompt in zip(processed_row, outputs, simple_prompts, prompts):
                row = {}
                row["prompt"] = prompt
                row["response"] = output
                row["type"] = example["type"]
                row["lang"] = example["lang"]
                row["model_name"] = args.model_name_or_path
                row["simple_prompt"] = simple_prompt
                row["judgement_pending"] = True
                row["judgement"] = ""
                row["rating"] = float(-1)
                final_data.append(row)

        new_outputs = outputs
        while True:
            print("eval mt turns")
            new_prompts = []
            simple_prompts = []
            org_row = []
            new_mt_idx = {}
            idx = 0
            for row_idx, mt_ix in mt_idx.items():
                row = processed_row[row_idx]
                answer = new_outputs[mt_ix["answer"]]

                lang = row["lang"]
                system = default_system_en
                if lang == "hi":
                    system = default_system_hi

                # print("row_idx, mt_ix", row_idx, mt_ix)
                next_ques_idx = mt_ix["question"] + 1
                if next_ques_idx < len(row["mt_question"]):
                    mt_idx[row_idx] = next_ques_idx
                    new_mt_idx[row_idx] = {
                        "question": next_ques_idx,
                        "answer": idx,
                    }
                    next_ques = row["mt_question"][next_ques_idx]

                    messages = row["messages"]
                    messages.append({"role": "assistant", "content": answer})
                    messages.append(
                        {"role": "user", "content": next_ques})
                    processed_row[row_idx]["messages"] = messages

                    if args.model_name_or_path == "mistralai/Mixtral-8x7B-Instruct-v0.1":
                        if lang != "en":
                            messages[-1]["content"] = messages[-1]["content"] + \
                                "\n Respond only in hindi"
                    if args.use_chat_format:
                        prompt = chat_formatting_function(
                            messages, tokenizer, args)
                    else:
                        prompt = "\n\n".join([x["content"] for x in prompt])

                    org_row.append(row)
                    new_prompts.append(prompt)
                    simple_prompts.append("\n\n".join(
                        [x["content"] for x in messages]))
                    idx += 1

            new_outputs = eval_hf_model(args, model, tokenizer, new_prompts)
            mt_idx = new_mt_idx

            for pix, _ in enumerate(new_prompts):
                row = {}
                row["prompt"] = new_prompts[pix]
                row["response"] = new_outputs[pix]
                row["type"] = org_row[pix]["type"]
                row["lang"] = org_row[pix]["lang"]
                row["model_name"] = args.model_name_or_path
                row["simple_prompt"] = simple_prompts[pix]
                row["judgement_pending"] = True
                row["judgement"] = ""
                row["rating"] = float(-1)
                final_data.append(row)

            if len(new_prompts) == 0:
                break

        with open(os.path.join(args.save_dir, f"lm_judge_predictions.jsonl"), "w") as fout:
            json.dump(final_data, fout, indent=4)

        api = HfApi()
        if api.repo_exists(repo_id=args.push_output, repo_type="dataset"):
            ds = load_dataset(args.push_output, split="train")
            for row in ds:
                if row["model_name"] != args.model_name_or_path:
                    final_data.append(row)

        dataset = process_and_update_dataset(final_data)
        dataset.push_to_hub(args.push_output, private=False)


def process_and_update_dataset(new_data):
    new_data_formatted = {key: [item[key]
                                for item in new_data] for key in new_data[0].keys()}
    new_dataset_chunk = Dataset.from_dict(new_data_formatted)
    dataset2 = new_dataset_chunk
    return dataset2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=1,
                        help="number of examples to use for few-shot evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str,
                        default="/sky-notebook/eval-results/indicwikibio/llama-7B/")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--max_context_length", type=int, default=3750, help="maximum number of tokens in the context passage."
    )
    parser.add_argument("--eval_batch_size", type=int,
                        default=1, help="batch size for evaluation.")

    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts.",
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_by_template",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`.",
    )

    parser.add_argument(
        "--awq",
        action="store_true",
        help="Load model as awq"
    )
    parser.add_argument(
        "--push_output",
        type=str,
        default="manishiitg/llm_judge",
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    args = parser.parse_args()
    args.eval_batch_size = args.eval_batch_size * torch.cuda.device_count()
    main(args)
