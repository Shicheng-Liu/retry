#!/bin/bash


MODEL_NAME="PKU-Alignment/beaver-7b-v3.0-reward"

CUDA_VISIBLE_DEVICES=4 python reward_eval.py \
    --data_path pythia-2.8b_full-hh-rlhf_validation.json \
    --model_name_or_path_reward $MODEL_NAME \
    --data_name full-hh-rlhf
    
