#!/usr/bin/env bash
# AgentCL coding protocol eval of a frozen ckpt (orchard).
#
# Default (PG only, clbench-style): one launch of stage 1 = pass-1 stream
# episodes (memory read/write) + memoryless baselines, then metrics.
#   EVAL_CKPT_DIR=<ckpt dir> EVAL_TAG=fmtonly-iter300 bash scripts/agentcl/eval_agentcl_orchard.sh
#   python scripts/agentcl/agentcl_metrics.py --traj /tmp/qixinx/runs/agentcl-<tag>-stage1/traj \
#     --per-problem agentcl-<tag>-per_problem.csv
#
# Optional (SG/GG, off by default): stage 2 = pass 2 + held-out with memory
# frozen at the end of each pass-1 run (data built from stage-1 trajectories).
#   EVAL_STAGE=2 EVAL_CKPT_DIR=<same> EVAL_TAG=<same> bash scripts/agentcl/eval_agentcl_orchard.sh
#   then pass both stage1 and stage2 traj dirs to agentcl_metrics.py.
#   GG also needs memoryless held-out scores: run stage 1 with AGENTCL_HELDOUT_BASELINE=1.
#
# Knobs: AGENTCL_RUNS (pass-1 runs per stream, default 3), AGENTCL_BASELINE_SAMPLES
# (default 3), AGENTCL_STREAMS (default "bcb_compositional bcb_naive").
# ACT/WRITE scaffold flags default to the thinking arms' training values
# (ACT thinking, WRITE prompt v3 + WRITE thinking, multiblock feedback); override
# via the same CODEBASE_* env vars. Same engine setup as eval_ckpt_19q_orchard.sh
# (weights-only load, eval before train, one harmless waste train step).
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${EVAL_STAGE:-1}"
CKPT="${EVAL_CKPT_DIR:?EVAL_CKPT_DIR required (dir containing latest_checkpointed_iteration.txt)}"
TAG="${EVAL_TAG:-iter$(cat "$CKPT/latest_checkpointed_iteration.txt")}"
DATA_DIR="${SD}/data/agentcl/${TAG}"
mkdir -p "$DATA_DIR"
STAGE1_TRAJ="/tmp/qixinx/runs/agentcl-${TAG}-stage1/traj/eval/heldout"

case "$STAGE" in
  1)
    python3 "${SD}/scripts/agentcl/gen_agentcl_eval_data.py" stage1 \
      --out "${DATA_DIR}/stage1.jsonl" \
      --runs "${AGENTCL_RUNS:-3}" --baseline-samples "${AGENTCL_BASELINE_SAMPLES:-3}" \
      $([[ "${AGENTCL_HELDOUT_BASELINE:-0}" == "1" ]] && echo --heldout-baseline) \
      --streams ${AGENTCL_STREAMS:-bcb_compositional bcb_naive}
    ;;
  2)
    [[ -d "$STAGE1_TRAJ" ]] || { echo "stage-1 trajectories not found: $STAGE1_TRAJ" >&2; exit 1; }
    python3 "${SD}/scripts/agentcl/gen_agentcl_eval_data.py" stage2 \
      --out "${DATA_DIR}/stage2.jsonl" \
      --stage1-traj "$STAGE1_TRAJ"/rollout_* \
      --streams ${AGENTCL_STREAMS:-bcb_compositional bcb_naive}
    ;;
  *) echo "unknown EVAL_STAGE=$STAGE" >&2; exit 1;;
esac
export CODEBASE_EVAL_PROMPT_DATA="${DATA_DIR}/stage${STAGE}.jsonl"
export CODEBASE_N_EVAL_SAMPLES=1   # repeats are explicit rows in the data file

# Scaffold flags of the thinking arms (overridable).
export CODEBASE_ACT_THINKING="${CODEBASE_ACT_THINKING:-1}"
export CODEBASE_WRITE_PROMPT_V3="${CODEBASE_WRITE_PROMPT_V3:-1}"
export CODEBASE_WRITE_THINKING="${CODEBASE_WRITE_THINKING:-1}"
export CODEBASE_MULTIBLOCK_FEEDBACK="${CODEBASE_MULTIBLOCK_FEEDBACK:-1}"

export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"
export CODEBASE_SAVE_DIR="$CKPT"
export CODEBASE_SAVE_INTERVAL=0
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export CODEBASE_TRAIN_EXTRA_ARGS="--no-load-optim --no-load-rng --start-rollout-id 0"
export CODEBASE_NUM_ROLLOUT=1
export CODEBASE_EVAL_BEFORE_TRAIN=1
export CODEBASE_EVAL_INTERVAL=1000000
export CODEBASE_SEQ_LEN=24576 CODEBASE_MAX_TOK_PER_GPU=20480
export CODEBASE_NGPU=8 CODEBASE_TP=2
export CODEBASE_SGLANG_CONCURRENCY="${CODEBASE_SGLANG_CONCURRENCY:-4}"
export CODEBASE_MAX_RESP=32768
# Waste-step constraint: train_iters = 1*1*4//4 = 1, gbs % DP == 0.
export CODEBASE_ROLLOUT_BATCH_SIZE=1 CODEBASE_N_SAMPLES=4
export CODEBASE_GLOBAL_BATCH_SIZE=4
export CODEBASE_PROMPT_DATA="${SD}/data/episodes_6p6_hard.jsonl"

export CODEBASE_RUN_ID="agentcl-${TAG}-stage${STAGE}"
export CODEBASE_USE_WANDB=
NODE_RUN_DIR="/tmp/qixinx/runs/${CODEBASE_RUN_ID}"
export CODEBASE_TRAJ_DIR="${NODE_RUN_DIR}/traj"
export CODEBASE_TORCH_DIST="/tmp/qixinx/models/Qwen3.5-4B_torch_dist"
mkdir -p "${NODE_RUN_DIR}"

source "${SD}/scripts/cluster_orchard_env.sh"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SCR="${CODEBASE_SCR:-/tmp/qixinx}"
mkdir -p "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/tmp" "$SCR/agentcl_ws"
export APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache" TMPDIR="$SCR/tmp"
export APPTAINERENV_APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINERENV_APPTAINER_CACHEDIR="$SCR/apptainer_cache" APPTAINERENV_TMPDIR="$SCR/tmp"
# Per-problem agent workspaces (node-local; visible inside the miles SIF via /tmp).
export APPTAINERENV_AGENTCL_WORK_ROOT="$SCR/agentcl_ws"

bash "${SD}/scripts/ensure_torch_dist_local.sh"

cd "$SD"
echo "=== agentcl eval STAGE=$STAGE CKPT=$CKPT TAG=$TAG RUN=$CODEBASE_RUN_ID data=$CODEBASE_EVAL_PROMPT_DATA on $(hostname) ==="
bash ./launch_codebase_adaption_apptainer.sh
