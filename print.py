import unicodedata
import os
import json
from datasets import load_dataset


def generateLMJudge():

    ds = load_dataset("manishiitg/llm_judge", split="train")

    def is_hindi(char):
        try:
            return unicodedata.name(char).startswith('DEVANAGARI')
        except ValueError:
            return False

    def contains_hindi(s):
        return any(is_hindi(char) for char in s)

    final_data = []
    for row in ds:
        final_data.append(row)

    scores = {}
    for row in final_data:
        if not row["judgement_pending"] and row["rating"] != -1:
            model_name = row["model_name"]
            if model_name in skip_model or "awq" in model_name:
                continue

            lang = "en"
            if contains_hindi(row["simple_prompt"]):
                lang = "hi"

            if model_name not in scores:
                scores[model_name] = {}
            if lang not in scores[model_name]:
                scores[model_name][lang] = []

            scores[model_name][lang].append(float(row["rating"]))

    # Create a list to hold the model scores for sorting
    model_scores = []

    # Iterate over the models and calculate the average score
    for model_name in scores:
        for lang in scores[model_name]:
            ratings = scores[model_name][lang]
            avg = sum(ratings) / len(ratings)
            model_scores.append((model_name, lang, avg, len(ratings)))

    # Sort the model scores by average score in descending order
    model_scores.sort(key=lambda x: x[2], reverse=True)

    markdown_output = ""
    langs = ["hi", "en"]
    for l in langs:
        # Generate the markdown output
        markdown_output += f"#### LLM Judge Language: {l} \n"
        markdown_output += f"| Model | Language | Score | No# Questions |\n"
        markdown_output += "| --- | --- | --- | --- |\n"

        for model_name, lang, avg, count in model_scores:
            if lang == l:
                markdown_output += f"| {model_name} | {lang} | {avg:.4f} | {count} |\n"

        markdown_output += f"\n\n"
    return markdown_output


# Convert JSON to Markdown table grouped by task and sub-task
markdown_output = generateLMJudge()

# Print the Markdown output
print(markdown_output)
