#!/bin/bash


#MODEL_NAME="OpenAssistant/reward-model-deberta-v3-large-v2"
MODEL_NAME="Skywork/Skywork-Reward-V2-Llama-3.1-8B"

CUDA_VISIBLE_DEVICES=4 python reward_eval.py \
    --data_path pythia-2.8b_tldr_test.json \
    --model_name_or_path_reward $MODEL_NAME \
    --data_name tldr
    
