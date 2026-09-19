#!/usr/bin/env bash
# v3-curriculum + step-grace reward experiment (2026-07-24, user-approved):
#   - identical to the v3curr recipe (gated_downstream + WRITE prompt v3 +
#     WRITE thinking + v3 three-tier curriculum consumed sequentially), plus:
#   - CODEBASE_STEP_GRACE=12: solves within 12 steps score full 1.0; regret
#     accrues only beyond the grace window (shifted linear, no cliff). Tests
#     whether verification behavior survives once the step discount stops
#     pricing it out (case study SS5.1 pricing analysis).
#   - CODEBASE_DROP_ZERO_STD_GROUPS=1: zero-variance sibling groups are
#     physically dropped before the dynamic-gbs computation.
# Counterpart control: babel's v3curr run (same data/prompt, original reward).
# Usage (flame-31): nohup bash scripts/train_4b_v3curr_grace12.sh > logs/smith-4b-v3curr-grace12.console.log 2>&1 &
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export CODEBASE_WRITE_REWARD_MODE=gated_downstream
export CODEBASE_WRITE_FORMAT_BONUS=0.1
export CODEBASE_WRITE_PROMPT_V3=1
export CODEBASE_WRITE_THINKING=1
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

export CODEBASE_TRAIN_TASK=swe_smith
export CODEBASE_TRAIN_DATASET=data/swe_smith/top53.jsonl
export CODEBASE_ROLLOUT_SHUFFLE=0
export CODEBASE_STEP_GRACE=12
export CODEBASE_DROP_ZERO_STD_GROUPS=1

export CODEBASE_RUN_ID="smith-4b-v3curr-grace12"
export CODEBASE_WANDB_RUN_ID="${CODEBASE_RUN_ID}"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swesmith-4b"

export CODEBASE_NUM_ROLLOUT=120
export CODEBASE_EVAL_INTERVAL=8
# Save every 4 rollouts (2026-07-27): same ratchet-guarantee as actonly —
# node deaths must not roll progress back a full 8-step window.
# Tightened to 2 (2026-07-27): three consecutive lives died inside the
# r104-107 window before reaching the next save; bank progress every 2 steps.
export CODEBASE_SAVE_INTERVAL=2
export CODEBASE_SEQ_LEN=24576
export CODEBASE_MAX_TOK_PER_GPU=20480   # proven-stable value (24576 OOMed twice on think-class runs)
export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"

NODE_RUN_DIR="/tmp/qixinx/runs/${CODEBASE_RUN_ID}"
export CODEBASE_SAVE_DIR="${NODE_RUN_DIR}/ckpt"
export CODEBASE_TRAJ_DIR="${NODE_RUN_DIR}/traj"
export CODEBASE_TORCH_DIST="/tmp/qixinx/models/Qwen3.5-4B_torch_dist"
mkdir -p "${NODE_RUN_DIR}"

source "${SD}/scripts/cluster_orchard_env.sh"
export CODEBASE_PROMPT_DATA="${SD}/data/swesmith_episodes_v3_curriculum.jsonl"
export CODEBASE_BASELINE_ARTIFACT="/project/flame/qixinx/swe_smith/passk_v1/baseline.json"
export CODEBASE_NGPU=8 CODEBASE_TP=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SCR="${CODEBASE_SCR:-/tmp/qixinx}"
mkdir -p "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/tmp"
export APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache" TMPDIR="$SCR/tmp"
export APPTAINERENV_APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINERENV_APPTAINER_CACHEDIR="$SCR/apptainer_cache" APPTAINERENV_TMPDIR="$SCR/tmp"

bash "${SD}/scripts/ensure_torch_dist_local.sh"

cd "$SD"
echo "=== SMITH v3-CURRICULUM RUN=${CODEBASE_RUN_ID} on $(hostname) (ckpt/traj: ${NODE_RUN_DIR}) ==="
bash ./launch_codebase_adaption_apptainer.sh
