#!/bin/bash

set -x

DEV=0,1,2,3,4,5,6,7
PORT=2742
OUTPUT=$1
ZERO_STAGE=2
DATA_PATH="/efs/shicheng/remax/step3_rlhf_finetuning/pythia-2.8b_full-hh-rlhf_explanation.json"
MODEL_NAME="/efs/shicheng/remax/step2_reward_model_finetuning/output/pythia-2.8b/full-hh-rlhf"
SEED=1234

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/pythia-2.8b/full-hh-rlhf/unlearned_reward
fi
mkdir -p $OUTPUT


(deepspeed --include localhost:$DEV --master_port $PORT \
reward_unlearning.py \
   --data_path $DATA_PATH \
   --data_output_path "/efs/shicheng/remax/output_data/pythia" \
   --data_split 0,10,0 \
   --model_name_or_path $MODEL_NAME \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --max_seq_len 512 \
   --learning_rate 1e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 1 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --print_loss \
   --deepspeed) 2>&1 | tee "$OUTPUT/training.log"