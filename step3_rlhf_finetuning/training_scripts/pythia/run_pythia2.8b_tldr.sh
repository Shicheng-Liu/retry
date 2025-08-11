#!/bin/bash
set -x
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export OMP_NUM_THREADS=4
# export MKL_NUM_THREADS=4
# export NUMEXPR_NUM_THREADS=4
# export OPENBLAS_NUM_THREADS=4
# export RAYON_NUM_THREADS=20
# export TOKENIZERS_PARALLELISM=False


DEV=0,1,2,3,4,5,6,7
PORT=1236
DATA_PATH="/efs/shicheng/remax/dataset/tldr"
ACTOR_MODEL_PATH=/efs/shicheng/remax/step1_supervised_finetuning/output/pythia-2.8b/tldr
REWARD_MODEL_PATH=/efs/shicheng/remax/step2_reward_model_finetuning/output/pythia-2.8b/tldr
ACTOR_ZERO_STAGE=2
REWARD_ZERO_STAGE=3
REFERENCE_ZERO_STAGE=3
OUTPUT=$1
SEED=2023

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/pythia-2.8b/tldr
fi

mkdir -p $OUTPUT


ACTOR_LR=1e-6

(deepspeed --include localhost:$DEV --master_port $PORT \
main.py \
   --algo "remax" \
   --data_path $DATA_PATH \
   --data_output_path "/tmp/data_files/pythia" \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --reward_model_name_or_path $REWARD_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 512 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate ${ACTOR_LR} \
   --actor_weight_decay 0.1 \
   --num_train_epochs 1 \
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
   --print_answers \
   --save_answers \
   --deepspeed --output_dir $OUTPUT) 2>&1 | tee $OUTPUT/training.log


#--enable_hybrid_engine \