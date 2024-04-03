from tqdm import tqdm
import argparse
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import vllm
from datasets import Dataset
from datetime import date
import torch
import re

# in this we simply save prompts outputs to a huggingface repo
# https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl

prompt = """
As an impartial evaluator, please assess the quality of the AI assistant's response to the user's question below. 
Your evaluation should take into account several factors, including helpfulness, relevance, accuracy, depth, creativity, and level of detail. 

[Question]
{question}
[AI Assistant's Response]
{answer}

Please rate the response on a scale of 1 to 10 for each of the following evaluation factors:

Helpfulness: The degree to which the response addresses the user's question or need.
Relevance: The extent to which the response is related to the user's question or topic.
Accuracy: The correctness of the information provided in the response.
Depth: The level of detail and comprehensiveness of the response.
Creativity: The originality and novelty of the response.
Level of Detail: The amount of information provided in the response.
Formatting and Presentation: How well the response was presented to the user. 

Calculate an overall rating based on above factors and also provide an detailed explanation for the overall rating.

Only respond in json format as follows:
{
  "overall_rating": {
    "explanation" : "<explanation>",
    "rating" : "<rating>",
  },
}
Response format should be parsable by json.loads
"""

prompt_hindi = """
As an impartial evaluator, please assess the quality of the AI assistant's response to the user's question below. 
Your evaluation should take into account several factors, including helpfulness, relevance, accuracy, depth, creativity, and level of detail. 

[Question]
{question}
[AI Assistant's Response]
{answer}

Please rate the response on a scale of 1 to 10 for each of the following evaluation factors:


Helpfulness: The degree to which the response addresses the user's question or need.
Relevance: The extent to which the response is related to the user's question or topic.
Accuracy: The correctness of the information provided in the response.
Depth: The level of detail and comprehensiveness of the response.
Creativity: The originality and novelty of the response.
Level of Detail: The amount of information provided in the response.
Formatting and Presentation: How well the response was presented to the user. 

Calculate an overall rating based on above factors and also provide an detailed explanation for the overall rating.
Degree of usage of hindi language. if reponse in english language, overall rating should be zero as response is expected in hindi.

Only respond in json format as follows:
{
  "overall_rating": {
    "explanation" : "<explanation>",
    "rating" : "<rating>",
  },
}
Response format should be parsable by json.loads
"""
# If an evaluation criteria is not valid for a question, add the rating as -1
# "helpfulness": {
#     "explanation" : "<explanation>",
#     "rating" : "<rating>",
#   },
#   "relevance": {
#     "explanation" : "<explanation>",
#     "rating" : "<rating>",
#   },
#   "accuracy": {
#     "explanation" : "<explanation>",
#     "rating" : "<rating>",
#   },
#   "depth": {
#     "explanation" : "<explanation>",
#     "rating" : "<rating>",
#   },
#   "creativity": {
#     "explanation" : "<explanation>",
#     "rating" : "<rating>",
#   },
#   "level_of_detail": {
#     "explanation" : "<explanation>",
#     "rating" : "<rating>",
#   },

# prompt = """
# Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below.
# Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
# Begin your evaluation by providing a short explanation. Be as objective as possible.

# [Question]
# {question}

# [The Start of Assistant's Answer]
# {answer}
# [The End of Assistant's Answer]

# After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format:
# Explanation: <explanation_for_rating>

# Overall Rating: <overall_rating>
# """

rating_pattern = r'Overall Rating: (\d+(?:\.\d+)?)'


def get_rating(output):
    match = re.search(rating_pattern, output)

    # If a match is found, extract the rating
    if match:
        rating = match.group(1)
        return rating
    else:
        raise ValueError()


def get_lm_judge_rating_prompt(question, answer, language):
    if language == "hi":
        prompt_1 = prompt_hindi.replace("{question}", question)
        prompt_1 = prompt_1.replace("{answer}", answer)
    else:
        prompt_1 = prompt.replace("{question}", question)
        prompt_1 = prompt_1.replace("{answer}", answer)
    return prompt_1


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

    ds = load_dataset("manishiitg/llm_judge", split="train")
    ds = ds.filter(lambda x: x["model_name"] == "google/gemma-2b-it").filter(lambda x: x["lang"] == "hi").shuffle().select(range(10))
    final_data = []
    for row in ds:
        final_data.append(row)

    count = 0
    for row in tqdm(final_data):
        if row["judgement_pending"]:
            count = count + 1

    # if count == 0:
    #     return

    # judge_model = "Qwen/Qwen1.5-72B-Chat-AWQ"
    judge_model = "Qwen/Qwen1.5-7B-Chat-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(judge_model)

    print("Loading model and tokenizer vllm awq...")
    model = vllm.LLM(
        model=judge_model,
        tokenizer=judge_model,
        tokenizer_mode="auto",
        tensor_parallel_size=torch.cuda.device_count(),
        # max_num_batched_tokens=4096,
        quantization="AWQ",
        max_model_len=4096,
        dtype="float16",
        gpu_memory_utilization=.8
    )

    prompts = []
    completed_data = []
    pending_data = []
    for row in tqdm(final_data):
        if row["judgement_pending"] or row["rating"] == -1 or row["model_name"] == "google/gemma-2b-it":
            instruction = row["simple_prompt"]
            answer = row["response"]
            prompt = get_lm_judge_rating_prompt(
                question=instruction, answer=answer, language=row["lang"])

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)
            pending_data.append(row)
        else:
            completed_data.append(row)

    outputs = eval_hf_model(args, model, tokenizer, prompts)

    for idx, text in enumerate(outputs):
        try:
            if "```" in text:
                text = text.replace("```json", "")
                text = text.replace("```", "")
                text = text.strip()
            try:
                ratings = json.loads(text)
                text = json.dumps(ratings, indent=4)
                rating = ratings["overall_rating"]["rating"]
                pending_data[idx]["judgement"] = text
                pending_data[idx]["rating"] = float(rating)
                pending_data[idx]["judgement_pending"] = False
            except TypeError as e:
                pending_data[idx]["judgement"] = text + "Exception:" + str(e)
                pending_data[idx]["rating"] = -1
                pending_data[idx]["judgement_pending"] = False
                print("text failed type error", text, -1, e)
            except ValueError as e:
                pattern = r'"rating"\s*:\s*(\d+(\.\d+)?)'
                match = re.search(pattern, text)

                if match:
                    rating = float(match.group(1))
                    pending_data[idx]["judgement"] = text
                    pending_data[idx]["rating"] = float(rating)
                    pending_data[idx]["judgement_pending"] = False
                else:
                    pending_data[idx]["judgement"] = text + "Exception:" + str(e)
                    pending_data[idx]["rating"] = -1
                    pending_data[idx]["judgement_pending"] = False
                    print("text failed", text, -1, e)
        except Exception as e:
            print("failed ", e)

        print("--------")
        for k, v in pending_data[idx].items():
            print(k, v)
    os.exit(1)
    final_data = pending_data + completed_data
    dataset = process_and_update_dataset(final_data)
    dataset.push_to_hub("manishiitg/llm_judge", private=False)


def process_and_update_dataset(new_data):
    new_data_formatted = {key: [item[key]
                                for item in new_data] for key in new_data[0].keys()}
    new_dataset_chunk = Dataset.from_dict(new_data_formatted)
    dataset2 = new_dataset_chunk
    return dataset2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--push_output",
        type=str,
        default="manishiitg/llm_judge",
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    args = parser.parse_args()
    main(args)
