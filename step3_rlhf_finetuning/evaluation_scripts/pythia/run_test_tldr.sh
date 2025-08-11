#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py \
    --model_name pythia-2.8b\
    --model_name_or_path_baseline EleutherAI/pythia-2.8b \
    --model_name_or_path_finetune /efs/shicheng/remax/step1_supervised_finetuning/output/pythia-2.8b/tldr \
    --model_name_or_path_rlhf output/pythia-2.8b/tldr/actor \
    --data_path /efs/shicheng/remax/dataset/tldr/validation.jsonl \
    --data_name tldr \
    --max_new_tokens 512 \
    --output_path /efs/shicheng/remax/step3_rlhf_finetuning/pythia-2.8b_tldr_validation.json\
    --batch_size 8
    
