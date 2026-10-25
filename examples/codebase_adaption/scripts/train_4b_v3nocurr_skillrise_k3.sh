#!/usr/bin/env bash
# SkillRise baseline (arXiv 2607.26784, "Agentic RL for Cross-Task Skill Evolution",
# github.com/Within-yao/SkillRise; user 2026-10-21) on our co-train setup, from the base
# model for 350 steps. Copy of train_4b_v3nocurr_cod_k3.sh (same data, curriculum order,
# thinking, seq len, batch, shuffle seed, save/eval cadence); the deltas are:
#   - Returns (CODEBASE_WRITE_REWARD_MODE=skillrise):
#       ACT trial j:      R = r_j                                  (own task score only)
#       WRITE after k:    R = sum_{j=k+1}^{k+3} 0.6^(j-k) * r_j    (K=3 window, user 2026-10-21)
#     No format term, no LLM judge (ACT exploration reward OFF, beta=0).
#   - Advantage: R minus the per-position group mean (stock grouping by role x group x
#     position), no std normalization (--disable-grpo-std-normalization = paper's mean_norm).
#   - Loss: stock PPO clip with eps 0.2/0.2 and KL coefficient 0 (paper: use_kl_loss=False).
#   - Prompts (CODEBASE_PROMPT_SET=skillrise): the WRITE instruction asks for a task-agnostic
#     SKILL DOCUMENT (paper's curate rules, outcome stated explicitly) and ACT injects it under
#     the paper's solve-prompt header; message structure stays ours (WRITE v3 framing).
# Smoke test: SMOKE_ROLLOUTS=2 bash scripts/train_4b_v3nocurr_skillrise_k3.sh
# Usage: nohup bash scripts/train_4b_v3nocurr_skillrise_k3.sh > /tmp/qixinx/smith-4b-v3nocurr-skillrise-k3.console.log 2>&1 &
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Ray stability mitigation (proven on flame-20: extends node lifespan ~3x).
export RAY_task_events_report_interval_ms=0
export RAY_health_check_failure_threshold=10
export RAY_health_check_period_ms=30000
# Keepalive against idle-TCP reclaim (GCP ~10min): keeps actor gRPC channels warm.
export RAY_grpc_client_keepalive_time_ms=60000
export RAY_grpc_client_keepalive_timeout_ms=30000

# --- SkillRise reward / algorithm / prompts (the experiment) ---
export CODEBASE_WRITE_REWARD_MODE=skillrise
export CODEBASE_WRITE_WINDOW=3
export CODEBASE_SKILLRISE_GAMMA=0.6
export CODEBASE_PROMPT_SET=skillrise
export CODEBASE_ACT_EXPLORE_BETA=0
export CODEBASE_KL=0
export CODEBASE_TRAIN_EXTRA_ARGS="--disable-grpo-std-normalization --eps-clip 0.2 --eps-clip-high 0.2"

# --- unchanged from explore-gwin-think ---
export CODEBASE_ACT_THINKING=1
export CODEBASE_WRITE_PROMPT_V3=1
export CODEBASE_WRITE_THINKING=1
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

export CODEBASE_TRAIN_TASK=swe_smith
export CODEBASE_TRAIN_DATASET=data/swe_smith/top53.jsonl
# CODEBASE_ROLLOUT_SHUFFLE intentionally NOT set: default 1 = shuffled.
export CODEBASE_DROP_ZERO_STD_GROUPS=1
export CODEBASE_MULTIBLOCK_FEEDBACK=1

export CODEBASE_RUN_ID="smith-4b-v3nocurr-skillrise-k3"
# Date-stamped, decoupled from RUN_ID: an existing wandb id would resume into the old run.
export CODEBASE_WANDB_RUN_ID="smith-4b-v3nocurr-skillrise-k3-1021"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swesmith-4b"

export CODEBASE_NUM_ROLLOUT=350
export CODEBASE_EVAL_INTERVAL=8
export CODEBASE_SAVE_INTERVAL=4
export CODEBASE_SEQ_LEN=20480
export CODEBASE_MAX_TOK_PER_GPU=20480
export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"

# Smoke mode (SMOKE_ROLLOUTS=N): own run dir, no wandb, N rollouts, save every rollout.
if [ -n "${SMOKE_ROLLOUTS:-}" ]; then
  export CODEBASE_RUN_ID="${CODEBASE_RUN_ID}-smoke"
  export CODEBASE_USE_WANDB=
  export CODEBASE_NUM_ROLLOUT="${SMOKE_ROLLOUTS}"
  export CODEBASE_SAVE_INTERVAL=1
  export CODEBASE_EVAL_INTERVAL=0
fi
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
echo "=== V3NOCURR SKILLRISE-K3 RUN=${CODEBASE_RUN_ID} on $(hostname) (ckpt/traj: ${NODE_RUN_DIR}) ==="
bash ./launch_codebase_adaption_apptainer.sh
