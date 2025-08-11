#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python test.py \
    --model_name llama-7b\
    --model_name_or_path_baseline meta-llama/Llama-2-7b-hf \
    --model_name_or_path_finetune /efs/shicheng/remax/step1_supervised_finetuning/output/llama-7b/tldr \
    --model_name_or_path_rlhf /efs/shicheng/remax/step3_rlhf_finetuning/output/llama-7b/tldr/actor \
    --model_name_or_path_xrlhf /efs/shicheng/remax/step4_xrlhf_finetuning/output/llama-7b/tldr/policy/actor \
    --data_path /efs/shicheng/remax/dataset/tldr/test.jsonl \
    --data_name tldr \
    --max_new_tokens 512 \
    --output_path /efs/shicheng/remax/step4_xrlhf_finetuning/llama-7b_tldr_test.json\
    --batch_size 8