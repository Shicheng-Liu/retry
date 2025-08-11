#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python get_explanation.py \
    --num_neighbors 2 \
    --batch_size 128 \
    --unsatisfactory_embedding_path /efs/shicheng/remax/step3_rlhf_finetuning/llama-7b_tldr_unsatisfactory_embeddings.pt \
    --training_embedding_path /efs/shicheng/remax/step3_rlhf_finetuning/llama-7b_tldr_training_embeddings.pt \
    --train_data_path /efs/shicheng/remax/dataset/tldr/train.jsonl \
    --output_explanation_path /efs/shicheng/remax/step3_rlhf_finetuning/llama-7b_tldr_explanation.json