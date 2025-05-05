#!/bin/bash

# trained by opt-1.3b reward model

MODEL_NAME="PKU-Alignment/beaver-7b-v3.0-reward"
#MODEL_NAME=~/workspace/siyuan/ReMax/step2_reward_model_finetuning/output/opt-1.3b/full-hh-rlhf

CUDA_VISIBLE_DEVICES=2 python reward_eval.py \
    --data_path "/gpuhome/hbz5148/workspace/siyuan/retry/dataset/tldr/train.json" \
    --new_data_path "/gpuhome/hbz5148/workspace/siyuan/retry/dataset/tldr/test.json" \
    --model_name_or_path_reward $MODEL_NAME 
    
