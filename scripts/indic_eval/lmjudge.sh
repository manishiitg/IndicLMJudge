#!/bin/bash

source ./scripts/indic_eval/common_vars.sh

TASK_NAME=lmjudge



for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}"
    FILE=$FOLDER/lm_judge_predictions.jsonl
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""
    fi

    template_format="eval.templates.create_prompt_by_template"
    if echo "$model_name" | grep -qi "Airavata"; then
        template_format="eval.templates.create_prompt_with_tulu_chat_format"
    fi
    if echo "$model_name" | grep -qi "OpenHathi-7B-Hi-v0.1-Base"; then
        template_format="eval.templates.create_prompt_with_llama2_chat_format"
    fi
    if [ "$check_file_existence" = false ] || [ ! -f "$FILE" ]; then
        python3 -m eval.lm_judge.run_eval \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 1 \
            --use_chat_format \
            --chat_formatting_function $template_format \
            $awq_param
    fi
    
done

echo "starting lmjudge ..."

python3 -m eval.lm_judge.judge
