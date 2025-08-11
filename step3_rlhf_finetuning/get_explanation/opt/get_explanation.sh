#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python get_explanation.py \
    --num_neighbors 1 \
    --batch_size 128 \
    --unsatisfactory_embedding_path /efs/shicheng/remax/step3_rlhf_finetuning/opt-1.3b_full-hh-rlhf_unsatisfactory_embeddings.pt \
    --training_embedding_path /efs/shicheng/remax/step3_rlhf_finetuning/opt-1.3b_full-hh-rlhf_training_embeddings.pt \
    --train_data_path /efs/shicheng/remax/dataset/full-hh-rlhf/train.json \
    --output_explanation_path /efs/shicheng/remax/step3_rlhf_finetuning/opt-1.3b_full-hh-rlhf_explanation.json