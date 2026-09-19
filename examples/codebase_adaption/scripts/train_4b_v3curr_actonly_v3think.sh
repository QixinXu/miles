#!/usr/bin/env bash
# v3-curriculum ACT-only ablation (2026-07-23, user-specified):
#   - data: v3 three-tier curriculum (pure_easy 63 -> easy_mid 138 -> mixed 181),
#     sequential consumption (CODEBASE_ROLLOUT_SHUFFLE=0)
#   - WRITE prompt v3 + WRITE-only thinking stay ON: memory is still written and
#     consumed every trial, but CODEBASE_TRAIN_ACT_ONLY=1 keeps WRITE samples
#     out of training (no reward, no gradient) — only ACT trains.
#   - everything else identical to the smith runs (MAX_TOK 20480 proven value).
# Counterpart experiment: babel runs the same data WITH WRITE training (gated).
# Usage (flame-20, node held by job 126648's sleep):
#   nohup bash scripts/train_4b_v3curr_actonly_v3think.sh > logs/smith-4b-v3curr-actonly.console.log 2>&1 &
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Ray stability mitigation (2026-07-25): flame-20 hit 6 Actor-RPC/GCS deaths in
# ~2 days (accelerating; latest gap ~2h). Known-issue levers: stop the task-event
# flood into GCS/dashboard (main memory-pressure source for long many-task jobs)
# and tolerate transient health-check misses instead of cascading raylet exit.
# grace12 on flame-31 intentionally NOT changed — serves as A/B control.
export RAY_task_events_report_interval_ms=0
export RAY_health_check_failure_threshold=10
export RAY_health_check_period_ms=30000

export CODEBASE_TRAIN_ACT_ONLY=1
export CODEBASE_WRITE_PROMPT_V3=1
export CODEBASE_WRITE_THINKING=1
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

export CODEBASE_TRAIN_TASK=swe_smith
export CODEBASE_TRAIN_DATASET=data/swe_smith/top53.jsonl
export CODEBASE_ROLLOUT_SHUFFLE=0

export CODEBASE_RUN_ID="smith-4b-v3curr-actonly-v3think"
export CODEBASE_WANDB_RUN_ID="${CODEBASE_RUN_ID}"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swesmith-4b"

export CODEBASE_NUM_ROLLOUT=120
export CODEBASE_EVAL_INTERVAL=8
# Save every 4 rollouts (2026-07-25): flame-20 lives can be shorter than the
# old 8-step save interval, which would ratchet-lock progress at the last
# ckpt forever. Denser saves guarantee forward progress; eval stays at 8.
# Tightened to 2 (2026-07-27): flame-20 lives now often shorter than 4
# rollouts; bank progress every 2 steps like grace12.
export CODEBASE_SAVE_INTERVAL=2
export CODEBASE_SEQ_LEN=24576
export CODEBASE_MAX_TOK_PER_GPU=20480
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
echo "=== V3CURR ACT-ONLY (+v3think WRITE untrained) RUN=${CODEBASE_RUN_ID} on $(hostname) (ckpt/traj: ${NODE_RUN_DIR}) ==="
bash ./launch_codebase_adaption_apptainer.sh
