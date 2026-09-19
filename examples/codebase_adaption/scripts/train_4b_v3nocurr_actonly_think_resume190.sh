#!/usr/bin/env bash
# Resume the ACT-only arm from its own iter_0000190 and extend 191 -> 248 rollouts
# (2026-09-01, user-specified). Purpose: compute-matched comparison against
# co-train. Over 191 rollouts act-only spent 98.8 GPU-hours vs co-train's 128.2
# (0.771x) because it trains only the ACT head and its episodes are shorter, so
# an equal-wall-clock budget buys act-only 191/0.771 = 248 rollouts.
#
# What this restart is and is not:
#   - Weights: exact (HF iter_0000190 from Racktic/swecl-qwen35-ckpt, converted
#     back to torch_dist on the node; the local iter dirs were deleted after the
#     upload was byte-verified).
#   - Data order: exact and NOT fabricated. The run's own
#     ckpt/rollout/global_dataset_state_dict_190.pt survived the ckpt cleanup
#     (it lives outside the iter_* dirs) and reads
#     {sample_offset: 382, epoch_id: 0, sample_group_index: 382, sample_index: 3056}.
#     382 == dataset size, i.e. exactly one epoch consumed; the framework wraps to
#     epoch 1 on the next fetch and reshuffles with seed(rollout_seed + 1).
#   - Optimizer: LOST (Adam moments were only in torch_dist). --no-load-optim
#     warm restart; expect a brief update-scale transient after rollout 191.
#   - Everything else is a VERBATIM copy of train_4b_v3nocurr_actonly_think.sh
#     except CODEBASE_NUM_ROLLOUT 191 -> 248.
# 191 is both a save point (191 % 4 == 3) and an eval point (191 % 8 == 7), so the
# first rollout after the restart checkpoints and evaluates immediately.
# The seeding block runs ONLY when the ckpt dir is not already seeded; after the
# first save (>= 191) a crash restart follows the normal full-resume path with no
# extra args -- the standing rule "never pass --start-rollout-id on crash
# restarts" is preserved.
# Usage: nohup bash scripts/train_4b_v3nocurr_actonly_think_resume190.sh > /tmp/qixinx/smith-4b-v3nocurr-actonly-think.console.log 2>&1 &
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Ray stability mitigation (proven on flame-20: extends node lifespan ~3x).
export RAY_task_events_report_interval_ms=0
export RAY_health_check_failure_threshold=10
export RAY_health_check_period_ms=30000
export RAY_grpc_client_keepalive_time_ms=60000
export RAY_grpc_client_keepalive_timeout_ms=30000

export CODEBASE_TRAIN_ACT_ONLY=1
export CODEBASE_ACT_THINKING=1
export CODEBASE_MULTIBLOCK_FEEDBACK=1
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

export CODEBASE_RUN_ID="smith-4b-v3nocurr-actonly-think"
export CODEBASE_WANDB_RUN_ID="${CODEBASE_RUN_ID}-0818"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swesmith-4b"

# 191 rollouts x 2 episodes = 382 = exactly one full epoch of the v3 pool
# (user decision 2026-07-30): the shuffled arm consumes the same data
# multiset as a full curriculum pass would — order is the only variable.
# At 120 the shuffled arm would eat 3x more hard instances than v3curr's
# first 240 episodes did (15.8% vs 5.4%), confounding reward comparisons.
export CODEBASE_NUM_ROLLOUT=248
export CODEBASE_EVAL_INTERVAL=8
# Interval 8 (user decision 2026-07-30); tighten only if a crash-loop reappears.
export CODEBASE_SAVE_INTERVAL=4
export CODEBASE_SEQ_LEN=20480
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


# ---------------- resume-from-HF seeding (idempotent) ----------------
RESUME_ITER=190
START_ROLLOUT=$((RESUME_ITER + 1))
HF_REPO="Racktic/swecl-qwen35-ckpt"
HF_SUBDIR="${CODEBASE_RUN_ID}/iter_0000190"
HFDL="/tmp/qixinx/hfdl/${CODEBASE_RUN_ID}-iter190"
SIF="${CODEBASE_MILES_SIF:-/project/flame/qixinx/images/miles_dev-202606081341.sif}"
STATE="${CODEBASE_SAVE_DIR}/rollout/global_dataset_state_dict_${RESUME_ITER}.pt"
TRACKER="${CODEBASE_SAVE_DIR}/latest_checkpointed_iteration.txt"

NEED_SEED=1
if [ -d "${CODEBASE_SAVE_DIR}/iter_$(printf %07d ${START_ROLLOUT})" ] || \
   { [ -f "${TRACKER}" ] && [ "$(cat "${TRACKER}" 2>/dev/null || echo 0)" -ge "${START_ROLLOUT}" ] 2>/dev/null; }; then
  NEED_SEED=0
  echo "[resume190] a post-resume save exists: normal resume path"
fi

if [ "${NEED_SEED}" = "1" ]; then
  if [ ! -f "${CODEBASE_SAVE_DIR}/.seed190_done" ]; then
    # The real dataset state must be present -- we do NOT fabricate it.
    [ -f "${STATE}" ] || { echo "[resume190] FATAL: missing ${STATE}"; exit 1; }
    echo "[resume190] dataset state found: ${STATE}"

    # HF hub cache must live on node disk: ~/.cache sits under the 25G home quota.
    export HF_HOME="/tmp/qixinx/hf_home"
    mkdir -p "${HF_HOME}"
    export HF_TOKEN="${HF_TOKEN:-$(grep HF_TOKEN /home/qixinx/miles/.env | cut -d= -f2)}"
    [ -n "${HF_TOKEN}" ] || { echo "[resume190] FATAL: no HF_TOKEN"; exit 1; }

    echo "[resume190] downloading ${HF_REPO}/${HF_SUBDIR} -> ${HFDL}"
    mkdir -p "${HFDL}"
    python3 - "$HF_REPO" "$HF_SUBDIR" "$HFDL" <<'PYEOF'
import os, sys, shutil
from huggingface_hub import snapshot_download
repo, subdir, dst = sys.argv[1:4]
path = snapshot_download(repo, allow_patterns=[f"{subdir}/*"], token=os.environ["HF_TOKEN"])
src = os.path.join(path, subdir)
for f in os.listdir(src):
    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
print("downloaded:", sorted(os.listdir(dst)))
PYEOF
    N_ST=$(ls "${HFDL}"/*.safetensors 2>/dev/null | wc -l)
    [ "${N_ST}" -ge 1 ] || { echo "[resume190] FATAL: no safetensors in ${HFDL}"; exit 1; }

    echo "[resume190] converting HF -> torch_dist into ${CODEBASE_SAVE_DIR}"
    source /home/qixinx/apps/apptainer/env.sh
    mkdir -p "${CODEBASE_SAVE_DIR}"
    apptainer exec --nv --bind /project/flame,/home/qixinx,/tmp "${SIF}" bash -c '
      cd /home/qixinx/miles && source scripts/models/qwen3.5-4B.sh &&
      PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint '"${HFDL}"' \
        --save '"${CODEBASE_SAVE_DIR}"
    [ -f "${TRACKER}" ] || { echo "[resume190] FATAL: converter left no tracker"; exit 1; }
    # The converter rewrote the tracker to its own iteration; the real dataset
    # state file is untouched and is what --start-rollout-id keys off.
    [ -f "${STATE}" ] || { echo "[resume190] FATAL: converter clobbered ${STATE}"; exit 1; }

    touch "${CODEBASE_SAVE_DIR}/.seed190_done"
    rm -rf "${HFDL}"
  else
    echo "[resume190] seed already done (marker present), launching"
  fi
  export CODEBASE_TRAIN_EXTRA_ARGS="${CODEBASE_TRAIN_EXTRA_ARGS:-} --no-load-optim --no-load-rng --start-rollout-id ${START_ROLLOUT}"
fi

cd "$SD"
echo "=== V3NOCURR ACT-ONLY RESUME@191 -> 248 RUN=${CODEBASE_RUN_ID} on $(hostname) (ckpt/traj: ${NODE_RUN_DIR}) ==="
bash ./launch_codebase_adaption_apptainer.sh
