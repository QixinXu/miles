#!/usr/bin/env bash
# Group B of the act/write decoupling study (user, 2026-10-01): WRITE frozen at co-train303.
# Coupled baseline: train_4b_gwin303_fmtonly_resume.sh (smith-4b-gwin303-fmtonly-resume-0919),
# one policy for ACT and WRITE, seeded from gwin-think iter_0000303.
# This run differs from the baseline ONLY in:
#   - WRITE calls go to a separate frozen SGLang model "writer" serving the HF iter_0000303
#     weights (update_weights=false, CODEBASE_WRITE_MODEL=writer); ACT calls keep the trained
#     policy's router.
#   - WRITE samples are not trained (CODEBASE_TRAIN_ACT_ONLY=1); ACT reward unchanged
#     (downstream + exploration, beta 0.3).
#   - Engine layout: ACT engines on GPUs [0, ACT_GPUS), writer engines on the rest; Megatron
#     still spans all 8 GPUs (colocated). The writer engines are offloaded during training like
#     the ACT engines, so they keep a CPU weight backup (enable_weights_cpu_backup) to come back
#     with the frozen weights intact.
# Everything else (env, seed, data-order reconstruction, start rollout 304, NUM_ROLLOUT 450) is
# the baseline script verbatim.
# Usage: nohup bash scripts/train_4b_gwin303_frozenwrite.sh > /tmp/qixinx/smith-4b-gwin303-frozenwrite.console.log 2>&1 &
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Ray stability mitigation (proven on flame-20: extends node lifespan ~3x).
export RAY_task_events_report_interval_ms=0
export RAY_health_check_failure_threshold=10
export RAY_health_check_period_ms=30000
# Keepalive against idle-TCP reclaim (GCP ~10min): keeps actor gRPC channels warm.
export RAY_grpc_client_keepalive_time_ms=60000
export RAY_grpc_client_keepalive_timeout_ms=30000
# OOM investigation (2026-08-13). Both runs died at iteration 0 inside the
# ref-logprob forward (model.py:265), requesting 11.37 GiB on both nodes.
# That size is the fp32 full-vocab logits tensor for a packed micro-batch at
# the cap: ~20090 tok * 151936 vocab * 4 B = 11.37 GiB. It is allocated by the
# model forward itself, BEFORE any chunked log-prob code runs.
# Tried and REVERTED, neither works:
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -- sglang's
#     TorchMemorySaver refuses to run with it and kills the rollout engine.
#   - CODEBASE_LOGP_CHUNK=256 -- math_utils.py:881 only chunks a logits.clone()
#     that is already bounded to chunk_size rows (~155 MB at 512), so it cannot
#     touch the 11.37 GiB forward allocation.
# Context: seq_length is 24576 (below) while max_tokens_per_gpu is 20480, so a
# sample in the 20480..24576 band bypasses the first-fit cap (utils/data.py:265)
# and forms its own oversized micro-batch. Explore-arm lengths (mean 16406,
# max 23213) are the first to reach that band.

export CODEBASE_WRITE_REWARD_MODE=format_only
export CODEBASE_WRITE_WINDOW=3
# --- ACT thinking mode (2026-08-16) ---
export CODEBASE_ACT_THINKING=1
export CODEBASE_WRITE_FORMAT_BONUS=0.1
export CODEBASE_WRITE_PROMPT_V3=1
export CODEBASE_WRITE_THINKING=1
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# --- ACT exploration reward (unchanged from explore-mbfb) ---
export CODEBASE_ACT_EXPLORE_BETA=0.3
# Judge key comes from the repo-level .env (OPENAI_API_KEY -> gpt-5-mini default).
if [ -f "/home/qixinx/miles/.env" ]; then
  set -a; source /home/qixinx/miles/.env; set +a
fi
export CODEBASE_JUDGE_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY missing from /home/qixinx/miles/.env}"
export CODEBASE_JUDGE_API_BASE="https://api.openai.com/v1"
export CODEBASE_JUDGE_MODEL="gpt-5-mini"
export CODEBASE_JUDGE_TIMEOUT=60
export CODEBASE_JUDGE_CONCURRENCY=64

export CODEBASE_TRAIN_TASK=swe_smith
export CODEBASE_TRAIN_DATASET=data/swe_smith/top53.jsonl
# CODEBASE_ROLLOUT_SHUFFLE intentionally NOT set: default 1 = shuffled.
export CODEBASE_DROP_ZERO_STD_GROUPS=1

# --- Group B: frozen writer ---
export CODEBASE_TRAIN_ACT_ONLY=1
export CODEBASE_WRITE_MODEL=writer
ACT_GPUS="${ACT_GPUS:-6}"
export CODEBASE_MULTIBLOCK_FEEDBACK=1

export CODEBASE_RUN_ID="smith-4b-gwin303-frozenwrite"
# Date-stamped, decoupled from RUN_ID: an existing wandb id would resume into the old run.
export CODEBASE_WANDB_RUN_ID="smith-4b-gwin303-frozenwrite-1001"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swesmith-4b"

export CODEBASE_NUM_ROLLOUT=450   # train through rollout 449 (user: "训练到450步")
export CODEBASE_EVAL_INTERVAL=8
export CODEBASE_SAVE_INTERVAL=4
# 2026-08-16 12:45: SEQ_LEN=28800 OOMed at the rollout-1 train step (fp32 logits of a
# 27.6k-token sample that bypasses the 20480 first-fit cap: "Tried to allocate 13.38 GiB",
# 12.47 GiB free). Reverted to the proven 24576 per the user's standing OOM instruction;
# MAX_TOK_PER_GPU untouched.
# 2026-08-16 19:57: OOMed again at 24576 (rollout 12: "Tried to allocate 11.37 GiB", 12.28 GiB
# free; response_length_max 23467 -- thinking lengthens ACT samples, more of them land in the
# 20480..24576 band that bypasses the first-fit cap). Per the same authorization SEQ_LEN is
# now pinned to MAX_TOK_PER_GPU (20480): no sample can form an oversized micro-batch; longer
# ACT samples are tail-truncated to 20480 by the existing seq_length truncation.
export CODEBASE_SEQ_LEN=20480
export CODEBASE_MAX_TOK_PER_GPU=20480   # proven-stable value (24576 OOMed twice on think-class runs)
export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"

# Smoke mode (SMOKE_ROLLOUTS=N): own run dir, no wandb, N rollouts from 304, save every rollout.
if [ -n "${SMOKE_ROLLOUTS:-}" ]; then
  export CODEBASE_RUN_ID="${CODEBASE_RUN_ID}-smoke"
  export CODEBASE_USE_WANDB=
  export CODEBASE_NUM_ROLLOUT=$((304 + SMOKE_ROLLOUTS))
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

# ---------------- resume-from-HF seeding (idempotent) ----------------
RESUME_ITER=303
START_ROLLOUT=$((RESUME_ITER + 1))
HF_REPO="Racktic/swecl-qwen35-ckpt"
HF_SUBDIR="smith-4b-v3nocurr-explore-gwin-think/iter_0000303"   # source run, not this run
HFDL="/tmp/qixinx/hf_models/gwin-think-iter303"   # kept: the frozen writer serves it
SIF="${CODEBASE_MILES_SIF:-/project/flame/qixinx/images/miles_dev-202606081341.sif}"

TRACKER="${CODEBASE_SAVE_DIR}/latest_checkpointed_iteration.txt"
NEED_SEED=1
if [ -f "${TRACKER}" ]; then
  CUR_IT="$(cat "${TRACKER}" 2>/dev/null || echo 0)"
  # A post-resume save exists (or the seed completed earlier): normal full
  # resume, no seeding and no start-rollout override.
  if [ "${CUR_IT}" -ge "${START_ROLLOUT}" ] 2>/dev/null; then
    NEED_SEED=0
    echo "[gwin303-frozenwrite] ckpt tracker at iter ${CUR_IT} >= ${START_ROLLOUT}: normal resume path"
  fi
fi

if [ "${NEED_SEED}" = "1" ]; then
  if [ ! -f "${CODEBASE_SAVE_DIR}/.seed303_done" ]; then
    # HF hub cache must live on the node disk: ~/.cache/huggingface sits under
    # the 25G home quota and a 8G ckpt download blows it (seen 2026-08-19).
    export HF_HOME="/tmp/qixinx/hf_home"
    mkdir -p "${HF_HOME}"
    export HF_TOKEN="${HF_TOKEN:-$(grep HF_TOKEN /home/qixinx/miles/.env | cut -d= -f2)}"
    [ -n "${HF_TOKEN}" ] || { echo "[gwin303-frozenwrite] FATAL: no HF_TOKEN"; exit 1; }

    echo "[gwin303-frozenwrite] downloading ${HF_REPO}/${HF_SUBDIR} -> ${HFDL}"
    mkdir -p "${HFDL}"
    python3 - "$HF_REPO" "$HF_SUBDIR" "$HFDL" <<'PY'
import os, sys, shutil
from huggingface_hub import snapshot_download
repo, subdir, dst = sys.argv[1:4]
path = snapshot_download(repo, allow_patterns=[f"{subdir}/*"], token=os.environ["HF_TOKEN"])
src = os.path.join(path, subdir)
for f in os.listdir(src):
    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
print("downloaded:", sorted(os.listdir(dst)))
PY
    # Weights sanity: the 4B ckpt ships 2 safetensors shards.
    N_ST=$(ls "${HFDL}"/*.safetensors 2>/dev/null | wc -l)
    [ "${N_ST}" -ge 1 ] || { echo "[gwin303-frozenwrite] FATAL: no safetensors in ${HFDL}"; exit 1; }

    echo "[gwin303-frozenwrite] converting HF -> torch_dist into ${CODEBASE_SAVE_DIR}"
    source /home/qixinx/apps/apptainer/env.sh
    mkdir -p "${CODEBASE_SAVE_DIR}"
    apptainer exec --nv --bind /project/flame,/home/qixinx,/tmp "${SIF}" bash -c '
      cd /home/qixinx/miles && source scripts/models/qwen3.5-4B.sh &&
      PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint '"${HFDL}"' \
        --save '"${CODEBASE_SAVE_DIR}"
    [ -f "${TRACKER}" ] || { echo "[gwin303-frozenwrite] FATAL: converter left no tracker file"; exit 1; }

    # Reconstruct the dataset-consumption state the lost
    # global_dataset_state_dict_107.pt held (values derived in the header).
    mkdir -p "${CODEBASE_SAVE_DIR}/rollout"
    apptainer exec --bind /home/qixinx,/tmp "${SIF}" python3 - "${CODEBASE_SAVE_DIR}/rollout/global_dataset_state_dict_${RESUME_ITER}.pt" <<'PY'
import sys, torch
torch.save(
    {"sample_offset": 226, "epoch_id": 1, "sample_group_index": 608,
     "sample_index": 4864, "metadata": {}},
    sys.argv[1],
)
print("wrote", sys.argv[1])
PY
    touch "${CODEBASE_SAVE_DIR}/.seed303_done"
  else
    echo "[gwin303-frozenwrite] seed already done (marker present), launching"
  fi
  # Converted ckpt carries no optimizer/rng state and its tracker says
  # iteration 0, so the start rollout must be forced exactly once here --
  # mirrors the sanctioned eval-flow flag set (eval_ckpt_19q_orchard.sh).
  export CODEBASE_TRAIN_EXTRA_ARGS="${CODEBASE_TRAIN_EXTRA_ARGS:-} --no-load-optim --no-load-rng --start-rollout-id ${START_ROLLOUT}"
fi

# ---------------- frozen writer model (every launch, incl. crash resumes) ----------------
[ "$(ls "${HFDL}"/*.safetensors 2>/dev/null | wc -l)" -ge 1 ] || {
  echo "[gwin303-frozenwrite] FATAL: writer weights missing in ${HFDL} (seed step downloads them)"; exit 1; }
WRITER_GPUS=$((CODEBASE_NGPU - ACT_GPUS))
[ "${ACT_GPUS}" -ge 1 ] && [ "${WRITER_GPUS}" -ge 1 ] || { echo "[gwin303-frozenwrite] FATAL: bad ACT_GPUS=${ACT_GPUS}"; exit 1; }
SGLANG_CFG="${NODE_RUN_DIR}/sglang_frozenwrite.yaml"
cat > "${SGLANG_CFG}" <<EOF
# Generated by train_4b_gwin303_frozenwrite.sh. First model = the trained ACT policy (default router).
sglang:
  - name: default
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: ${ACT_GPUS}
  - name: writer
    model_path: ${HFDL}
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: ${WRITER_GPUS}
        overrides:
          enable_weights_cpu_backup: true
EOF
export CODEBASE_TRAIN_EXTRA_ARGS="${CODEBASE_TRAIN_EXTRA_ARGS:-} --sglang-config ${SGLANG_CFG}"

cd "$SD"
echo "=== GWIN303 FROZEN-WRITER @${START_ROLLOUT} RUN=${CODEBASE_RUN_ID} act_gpus=${ACT_GPUS} writer_gpus=${WRITER_GPUS} on $(hostname) (ckpt/traj: ${NODE_RUN_DIR}) ==="
bash ./launch_codebase_adaption_apptainer.sh
