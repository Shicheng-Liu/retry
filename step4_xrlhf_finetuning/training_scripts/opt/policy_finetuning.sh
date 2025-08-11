#!/bin/bash
set -x


DEV=0,1,2,3,4,5,6,7
PORT=1236
DATA_PATH="/efs/shicheng/remax/step3_rlhf_finetuning/opt-1.3b_full-hh-rlhf_unsatisfactory.json"
ACTOR_MODEL_PATH=/efs/shicheng/remax/step3_rlhf_finetuning/output/opt-1.3b/full-hh-rlhf-one-epoch/actor
REWARD_MODEL_PATH=/efs/shicheng/remax/step4_xrlhf_finetuning/output/opt-1.3b/full-hh-rlhf/unlearned_reward
#REWARD_MODEL_PATH=/efs/shicheng/remax/step2_reward_model_finetuning/output/opt-1.3b/full-hh-rlhf
ACTOR_ZERO_STAGE=2
REWARD_ZERO_STAGE=3
REFERENCE_ZERO_STAGE=3
OUTPUT=$1
SEED=2023

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/opt-1.3b/full-hh-rlhf/policy
fi

mkdir -p $OUTPUT


ACTOR_LR=1e-6

(deepspeed --include localhost:$DEV --master_port $PORT \
policy_finetuning.py \
   --algo "remax" \
   --data_path $DATA_PATH \
   --data_output_path "/tmp/data_files/opt" \
   --data_split 0,0,10 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --reward_model_name_or_path $REWARD_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --generation_batches 2 \
   --ppo_epochs 1 \
   --max_answer_seq_len 512 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate ${ACTOR_LR} \
   --actor_weight_decay 0.1 \
   --num_train_epochs 2 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --disable_actor_dropout \
   --disable_reward_dropout \
   --num_warmup_steps 0 \
   --kl_ctl 0.05 \
   --gamma 0.99 \
   --deepspeed \
   --seed $SEED \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --reward_zero_stage $REWARD_ZERO_STAGE \
   --reference_zero_stage $REFERENCE_ZERO_STAGE \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --enable_hybrid_engine \
   --print_answers \
   --save_answers \
   --deepspeed --output_dir $OUTPUT) 2>&1 | tee $OUTPUT/training.log


#--enable_hybrid_engine \