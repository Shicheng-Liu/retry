#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py \
    --model_name pythia-2.8b\
    --model_name_or_path_baseline EleutherAI/pythia-2.8b \
    --model_name_or_path_finetune /efs/shicheng/remax/step1_supervised_finetuning/output/pythia-2.8b/full-hh-rlhf \
    --model_name_or_path_rlhf output/pythia-2.8b/full-hh-rlhf/actor \
    --data_path /efs/shicheng/remax/dataset/full-hh-rlhf/validation.json \
    --data_name full-hh-rlhf \
    --max_new_tokens 512 \
    --output_path /efs/shicheng/remax/step3_rlhf_finetuning/pythia-2.8b_full-hh-rlhf_validation.json\
    --batch_size 8
    
