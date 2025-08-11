#!/bin/bash

# trained by opt-1.3b reward model
#MODEL_NAME="OpenAssistant/reward-model-deberta-v3-large-v2"
#MODEL_NAME="PKU-Alignment/beaver-7b-v3.0-reward"
MODEL_NAME="Skywork/Skywork-Reward-V2-Llama-3.1-8B"
#MODEL_NAME=/efs/shicheng/remax/step2_reward_model_finetuning/output/opt-1.3b/full-hh-rlhf

CUDA_VISIBLE_DEVICES=4 python reward_eval.py \
    --data_path opt-1.3b_full-hh-rlhf_test.json \
    --model_name_or_path_reward $MODEL_NAME \
    --data_name full-hh-rlhf
    
