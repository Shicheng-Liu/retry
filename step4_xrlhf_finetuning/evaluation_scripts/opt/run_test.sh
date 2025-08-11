#!/bin/bash

# trained by opt-1.3b reward model


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python test.py \
    --model_name opt-1.3b\
    --model_name_or_path_baseline facebook/opt-1.3b \
    --model_name_or_path_finetune /efs/shicheng/remax/step1_supervised_finetuning/output/opt-1.3b/full-hh-rlhf \
    --model_name_or_path_rlhf /efs/shicheng/remax/step3_rlhf_finetuning/output/opt-1.3b/full-hh-rlhf-one-epoch/actor \
    --model_name_or_path_xrlhf /efs/shicheng/remax/step4_xrlhf_finetuning/output/opt-1.3b/full-hh-rlhf/policy/actor \
    --data_path /efs/shicheng/remax/dataset/full-hh-rlhf/test.json \
    --data_name full-hh-rlhf \
    --max_new_tokens 512 \
    --output_path /efs/shicheng/remax/step4_xrlhf_finetuning/opt-1.3b_full-hh-rlhf_test.json\
    --batch_size 8