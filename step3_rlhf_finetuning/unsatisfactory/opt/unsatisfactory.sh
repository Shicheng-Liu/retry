#!/bin/bash

#MODEL_NAME="OpenAssistant/reward-model-deberta-v3-large-v2"
MODEL_NAME="PKU-Alignment/beaver-7b-v3.0-reward"
#MODEL_NAME=~/workspace/siyuan/ReMax/step2_reward_model_finetuning/output/opt-1.3b/full-hh-rlhf

CUDA_VISIBLE_DEVICES=1 python unsatisfactory.py \
    --data_path opt-1.3b_full-hh-rlhf_validation.json \
    --model_name opt-1.3b \
    --model_name_or_path_reward $MODEL_NAME \
    --data_name full-hh-rlhf
    