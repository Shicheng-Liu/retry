#!/bin/bash


MODEL_NAME=/efs/shicheng/remax/step2_reward_model_finetuning/output/opt-1.3b/full-hh-rlhf

CUDA_VISIBLE_DEVICES=1 python get_embeddings.py \
    --train_data_path /efs/shicheng/remax/dataset/full-hh-rlhf/train.json \
    --unsatisfactory_data_path /efs/shicheng/remax/step3_rlhf_finetuning/opt-1.3b_full-hh-rlhf_unsatisfactory.json \
    --model_name_or_path_reward $MODEL_NAME \
    --data_name full-hh-rlhf \
    --train_output_path opt-1.3b_full-hh-rlhf_training_embeddings.pt \
    --unsatisfactory_output_path opt-1.3b_full-hh-rlhf_unsatisfactory_embeddings.pt 
    
