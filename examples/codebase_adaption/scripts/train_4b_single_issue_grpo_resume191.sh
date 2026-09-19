#!/usr/bin/env bash
# Extension (2026-09-17, user request): continue the finished 191-step run (SUCCEEDED
# 2026-09-16 21:30, tracker at iter 190, Adam moments on disk) to 300 rollouts. Clean
# full resume like the fmtonly resume scripts: no --start-rollout-id (Megatron reads
# iteration 190 from the tracker in --load), no HF round-trip. Only NUM_ROLLOUT, the
# scheduler override and the console log path differ from the original wrapper.
# Single-issue GRPO baseline (2026-09-12, user-specified). No sequence, no
# memory, no WRITE, no exploration reward -- plain GRPO on one issue at a time,
# with the usual ACT reward r = 1[tests pass] * (1 - (t-1)/T), T=40.
#
# Exactly matched to the 12-issue episode arms, by construction:
#   - data   swesmith_issues_v3_flat.jsonl is swesmith_episodes_v3_curriculum
#            flattened 1:1 -- 382 x 12 = 4584 rows, one issue each, same issue
#            multiset in the same order, zero duplicates.
#   - steps  4584 / 24 = 191 rollouts = 1 epoch   (episode arms: 382 / 2 = 191)
#   - samples 24 prompts x G=8 = 192 per step     (episode arms: 2 x 8 x 12 = 192)
#   - concurrency SGLANG_CONCURRENCY=2 -> 2 x 8 GPUs = 16 concurrent episodes,
#            the same in-flight count the episode arms run at, so the 192 issues
#            are worked in 12 waves of 16 exactly as they are today.
#
# CODEBASE_SKIP_WRITE=1 reuses the existing early-continue branch that
# CODEBASE_NO_MEMORY uses, so the rewrite step is never generated. It does NOT
# take the NO_MEMORY path itself: that one also accumulates the whole episode
# into the ACT context (the ICL-memory baseline) and is eval-only.
# Usage: nohup bash scripts/train_4b_single_issue_grpo_resume191.sh > /tmp/qixinx/smith-4b-single-issue-grpo-ACTthinking.console.resume191.log 2>&1 &
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

# --- ACT thinking mode (2026-08-16) ---
export CODEBASE_SKIP_WRITE=1        # no memory rewrite at all
# Reword the two spots in the clbench ACT prompt that design or imply a
# sequence: the brief's objective loses "while reusing knowledge from earlier
# issues", its cross-issue-transfer constraint is dropped, and the header
# "--- Issue k/N ---" loses the counter. Everything else in the prompt is
# byte-identical, including the plain-text ACT system template that miles
# monkey-patches in at runtime.
export CODEBASE_SINGLE_ISSUE_PROMPT=1
export CODEBASE_ACT_THINKING=1
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600


export CODEBASE_TRAIN_TASK=swe_smith
export CODEBASE_TRAIN_DATASET=data/swe_smith/top53.jsonl
# CODEBASE_ROLLOUT_SHUFFLE intentionally NOT set: default 1 = shuffled.
export CODEBASE_DROP_ZERO_STD_GROUPS=1
export CODEBASE_MULTIBLOCK_FEEDBACK=1

export CODEBASE_RUN_ID="smith-4b-single-issue-grpo-ACTthinking"
# Date-stamped, decoupled from RUN_ID: an existing wandb id would resume into the old run.
export CODEBASE_WANDB_RUN_ID="smith-4b-single-issue-grpo-ACTthinking-0912"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swesmith-4b"

export CODEBASE_ROLLOUT_BATCH_SIZE=24   # 24 distinct issues per step
export CODEBASE_N_SAMPLES=8             # G=8 -> 192 samples/step (same as the episode arms)
export CODEBASE_SGLANG_CONCURRENCY=2    # 2 x 8 GPUs = 16 concurrent episodes, same as the episode arms
export CODEBASE_NUM_ROLLOUT=300   # rollouts 0..299; was 191 (0..190)
# Raising NUM_ROLLOUT changes the LR scheduler's lr_decay_steps (191*16 -> 300*16) and
# Megatron's OptimizerParamScheduler.load_state_dict asserts on it unless overridden.
export CODEBASE_TRAIN_EXTRA_ARGS="${CODEBASE_TRAIN_EXTRA_ARGS:-} --override-opt-param-scheduler"
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

NODE_RUN_DIR="/tmp/qixinx/runs/${CODEBASE_RUN_ID}"
export CODEBASE_SAVE_DIR="${NODE_RUN_DIR}/ckpt"
export CODEBASE_TRAJ_DIR="${NODE_RUN_DIR}/traj"
export CODEBASE_TORCH_DIST="/tmp/qixinx/models/Qwen3.5-4B_torch_dist"
mkdir -p "${NODE_RUN_DIR}"

source "${SD}/scripts/cluster_orchard_env.sh"
export CODEBASE_PROMPT_DATA="${SD}/data/swesmith_issues_v3_flat.jsonl"
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
echo "=== SINGLE-ISSUE GRPO RUN=${CODEBASE_RUN_ID} on $(hostname) (ckpt/traj: ${NODE_RUN_DIR}) ==="
bash ./launch_codebase_adaption_apptainer.sh
