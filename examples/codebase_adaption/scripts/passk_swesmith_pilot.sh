#!/usr/bin/env bash
# SWE-smith 试点 pass@k(avg@2)采集: LR=0 权重零更新, 轨迹即 baseline(2026-07-20)。
# 1262 单题 episode(memory 天然为空, 量纯题目难度), N_SAMPLES=2, batch 64 -> 20 rollout 全覆盖。
# 4 卡即可跑(便于塞进 preempt 的 mix 节点), 8 卡更快。
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export CODEBASE_TRAIN_TASK=swe_smith
export CODEBASE_TRAIN_DATASET=data/swe_smith/pilot.jsonl   # 相对 CLBENCH_ROOT
export CODEBASE_TRAIN_ACT_ONLY=1
export CODEBASE_LR=0.0
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

export CODEBASE_RUN_ID="${CODEBASE_RUN_ID:-swesmith-passk-pilot-avg2}"
export CODEBASE_NUM_ROLLOUT=20
export CODEBASE_ROLLOUT_BATCH_SIZE=64
export CODEBASE_N_SAMPLES=2
export CODEBASE_GLOBAL_BATCH_SIZE=16
export CODEBASE_EVAL_INTERVAL=off
export CODEBASE_SAVE_INTERVAL=0
export CODEBASE_SEQ_LEN=24576
export CODEBASE_MAX_TOK_PER_GPU=20480
export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"
export CODEBASE_PROMPT_DATA="${SD}/data/swesmith_passk_pilot.jsonl"
export CODEBASE_BASELINE_ARTIFACT="${SD}/data/baseline_4b_textfmt.json"  # 无关但需存在
NGPU_DETECTED=$(nvidia-smi -L 2>/dev/null | wc -l); export CODEBASE_NGPU=${CODEBASE_NGPU:-${NGPU_DETECTED:-4}}
export CODEBASE_TP=2

NODE_RUN_DIR="/tmp/qixinx/runs/${CODEBASE_RUN_ID}"
export CODEBASE_SAVE_DIR="${NODE_RUN_DIR}/ckpt"
export CODEBASE_TRAJ_DIR="${NODE_RUN_DIR}/traj"
export CODEBASE_TORCH_DIST="/tmp/qixinx/models/Qwen3.5-4B_torch_dist"
mkdir -p "${NODE_RUN_DIR}"
source "${SD}/scripts/cluster_orchard_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((CODEBASE_NGPU-1)))}"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
SCR="${CODEBASE_SCR:-/tmp/qixinx}"
mkdir -p "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/tmp"
export APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache" TMPDIR="$SCR/tmp"
export APPTAINERENV_APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINERENV_APPTAINER_CACHEDIR="$SCR/apptainer_cache" APPTAINERENV_TMPDIR="$SCR/tmp"

bash "${SD}/scripts/ensure_torch_dist_local.sh"

cd "$SD"
echo "=== SWE-smith pass@k RUN=${CODEBASE_RUN_ID} NGPU=${CODEBASE_NGPU} on $(hostname) ==="
bash ./launch_codebase_adaption_apptainer.sh
