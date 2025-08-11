#!/bin/bash


MODEL_NAME=/efs/shicheng/remax/step2_reward_model_finetuning/output/pythia-2.8b/tldr

CUDA_VISIBLE_DEVICES=0 python get_embeddings.py \
    --train_data_path /efs/shicheng/remax/dataset/tldr/train.jsonl \
    --unsatisfactory_data_path /efs/shicheng/remax/step3_rlhf_finetuning/pythia-2.8b_tldr_unsatisfactory.json \
    --model_name_or_path_reward $MODEL_NAME \
    --data_name tldr \
    --max_seq_len 640 \
    --train_output_path pythia-2.8b_tldr_training_embeddings.pt \
    --unsatisfactory_output_path pythia-2.8b_tldr_unsatisfactory_embeddings.pt 
    
