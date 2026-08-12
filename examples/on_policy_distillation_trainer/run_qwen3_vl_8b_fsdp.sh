#!/usr/bin/env bash
# Standard OPD from "Visual-Advantage On-Policy Distillation for Vision-Language Models"
# Qwen3-VL-8B teacher -> Qwen3-VL-2B student | exact full-vocabulary reverse KL | 8 GPUs

set -xeuo pipefail

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-/home/bmm-system/data/private/xuyingjia/src/Qwen3-VL-2B-Instruct}
TEACHER_MODEL=${TEACHER_MODEL:-/home/bmm-system/data/private/xuyingjia/src/Qwen3-VL-8B-Instruct}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
full_vocab_chunk_size=${FULL_VOCAB_CHUNK_SIZE:-256}

train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-5120}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
sp_size=${SP_SIZE:-8}
rollout_n=${ROLLOUT_N:-4}
val_n=${VAL_N:-8}

total_epochs=${TOTAL_EPOCHS:-5}
save_freq=${SAVE_FREQ:-200}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_distill_geo3k}
experiment_name=${EXPERIMENT_NAME:-standard_opd_qwen3vl_8b_to_2b_geo3k_v2_4096_testGPUmem}
# ---- end user-adjustable ----

# train_file=${TRAIN_FILE:-/home/bmm-system/data/private/xuyingjia/datasets_support/geo3k/train.parquet}
# val_file=${VAL_FILE:-/home/bmm-system/data/private/xuyingjia/datasets_support/geo3k/test.parquet}


train_file=${TRAIN_FILE:-/home/bmm-system/data/private/xuyingjia/datasets_support/geo3k_direct_observe_v2/train.parquet}
val_file=${VAL_FILE:-/home/bmm-system/data/private/xuyingjia/datasets_support/geo3k_direct_observe_v2/test.parquet}




max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$train_file']"
    data.val_files="['$val_file']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.image_key=images
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${sp_size}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${sp_size}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.n=${val_n}
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=True
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=0
    distillation.nnodes=0
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.distillation_loss.loss_mode=full_reverse_kl
    distillation.distillation_loss.full_vocab_chunk_size=${full_vocab_chunk_size}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=False
    distillation.distillation_loss.loss_max_clamp=null
    distillation.distillation_loss.log_prob_min_clamp=null
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
