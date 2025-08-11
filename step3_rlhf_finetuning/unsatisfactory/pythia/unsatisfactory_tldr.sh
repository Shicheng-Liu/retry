#!/bin/bash

MODEL_NAME="OpenAssistant/reward-model-deberta-v3-large-v2"

CUDA_VISIBLE_DEVICES=1 python unsatisfactory.py \
    --data_path pythia-2.8b_tldr_validation.json \
    --model_name pythia-2.8b \
    --model_name_or_path_reward $MODEL_NAME \
    --data_name tldr
    