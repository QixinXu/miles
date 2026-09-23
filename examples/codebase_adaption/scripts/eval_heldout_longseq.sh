#!/usr/bin/env bash
# held-out 长序列评测(SWE-smith,2026-09-21)。结构照抄 eval_ckpt_19q.sh 的"Megatron 加载式 eval",
# 只改三处:
#   1. 任务/数据集换成 held-out SWE-smith registry(CODEBASE_EVAL_TASK=swe_smith + CODEBASE_EVAL_DATASET)
#   2. 打开 thinking 类训练臂的 scaffold 开关(ACT thinking / WRITE v3 / WRITE thinking / multiblock)
#   3. eval prompt 数据由外部给(EVAL_DATA),replace 与 stateless 的 N_EVAL_SAMPLES 不同
#
# 口径与当年 9+10 完全一致:
#   replace   : 整条 sequence、带记忆(WRITE 每题覆写),5 个 run 已展开成 5 组行,N_EVAL_SAMPLES=1
#   stateless : 单题 no_memory,N_EVAL_SAMPLES=4 得 avg@4
#
# 用法:
#   EVAL_MODE=replace|stateless EVAL_DATA=<eval prompt jsonl> \
#   EVAL_CKPT_DIR=<含 latest_checkpointed_iteration.txt 的 torch_dist> EVAL_TAG=<名字> \
#     bash scripts/eval_heldout_longseq.sh
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${EVAL_MODE:?EVAL_MODE=replace|stateless}"
DATA="${EVAL_DATA:?EVAL_DATA required (eval prompt jsonl)}"
CKPT="${EVAL_CKPT_DIR:?EVAL_CKPT_DIR required}"
TAG="${EVAL_TAG:-iter$(cat "$CKPT/latest_checkpointed_iteration.txt")}"
[ -f "$DATA" ] || { echo "EVAL_DATA 不存在: $DATA" >&2; exit 1; }
[ -f "$CKPT/latest_checkpointed_iteration.txt" ] || { echo "CKPT 缺 tracker: $CKPT" >&2; exit 1; }

export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"
export CODEBASE_HF_CKPT="/data/user_data/qixinx/Qwen3.5-4B"
export CODEBASE_TORCH_DIST="/data/user_data/qixinx/Qwen3.5-4B_torch_dist"
export CODEBASE_SAVE_DIR="$CKPT"
export CODEBASE_SAVE_INTERVAL=0
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export CODEBASE_TRAIN_EXTRA_ARGS="--no-load-optim --no-load-rng --start-rollout-id 0"
export CODEBASE_NUM_ROLLOUT=1
export CODEBASE_EVAL_BEFORE_TRAIN=1
export CODEBASE_EVAL_INTERVAL=1000000
export CODEBASE_SEQ_LEN=24576 CODEBASE_MAX_TOK_PER_GPU=24576
export CODEBASE_NGPU=8 CODEBASE_TP=2
export CODEBASE_SGLANG_CONCURRENCY="${CODEBASE_SGLANG_CONCURRENCY:-4}"
export CODEBASE_MAX_RESP=32768
export CODEBASE_ROLLOUT_BATCH_SIZE=1 CODEBASE_N_SAMPLES=4
export CODEBASE_GLOBAL_BATCH_SIZE=4

# ── thinking 臂的 scaffold(fmtonly-191 / co-train-251 都是 thinking 组) ──
export CODEBASE_ACT_THINKING="${CODEBASE_ACT_THINKING:-1}"
export CODEBASE_WRITE_PROMPT_V3="${CODEBASE_WRITE_PROMPT_V3:-1}"
export CODEBASE_WRITE_THINKING="${CODEBASE_WRITE_THINKING:-1}"
export CODEBASE_MULTIBLOCK_FEEDBACK="${CODEBASE_MULTIBLOCK_FEEDBACK:-1}"
# 判分/命令内存护栏(与训练同):astroid 类判分内存炸弹的防线
export CODEBASE_CMD_MEM_LIMIT_KB="${CODEBASE_CMD_MEM_LIMIT_KB:-16777216}"
export CODEBASE_GRADE_MEM_LIMIT_KB="${CODEBASE_GRADE_MEM_LIMIT_KB:-33554432}"

# ── held-out SWE-smith 数据源 ──
export CODEBASE_EVAL_TASK=swe_smith
export CODEBASE_EVAL_DATASET="${CODEBASE_EVAL_DATASET:-/data/user_data/qixinx/swe_smith/heldout_v4/heldout_registry.jsonl}"
export CODEBASE_EVAL_PROMPT_DATA="$DATA"
# 废步(eval 之后那个无害训练 rollout)的训练数据:也指向 held-out,避免触碰别的 SIF 库。
# 驱动脚本会在 eval 轨迹落盘后直接杀掉进程,这一步正常不会真的跑完。
export CODEBASE_TRAIN_TASK=swe_smith
export CODEBASE_TRAIN_DATASET="$CODEBASE_EVAL_DATASET"
export CODEBASE_PROMPT_DATA="${CODEBASE_PROMPT_DATA:-$SD/data/ho_2x13_26issues_replace.jsonl}"

case "$MODE" in
  replace)   export CODEBASE_N_EVAL_SAMPLES=1 ;;
  stateless) export CODEBASE_N_EVAL_SAMPLES=4 ;;
  *) echo "unknown EVAL_MODE=$MODE" >&2; exit 1;;
esac

export CODEBASE_RUN_ID="hoeval-${TAG}-${MODE}-$(basename "$DATA" .jsonl | sed 's/^ho_//; s/_replace$//; s/_stateless$//')"
export CODEBASE_USE_WANDB=

source "${SD}/scripts/cluster_babel_env.sh"   # CLBENCH_SIF_DIR / SAVE_DIR 等 babel 路径
export CODEBASE_SAVE_DIR="$CKPT"              # cluster_babel_env 会按 RUN_ID 推 SAVE_DIR,这里必须回到 ckpt
SCR="${CODEBASE_SCR:-/scratch/qixinx}"
mkdir -p "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/tmp"
export APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache" TMPDIR="$SCR/tmp"
export APPTAINERENV_APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINERENV_APPTAINER_CACHEDIR="$SCR/apptainer_cache" APPTAINERENV_TMPDIR="$SCR/tmp"
export CODEBASE_APPTAINER_RUNTIME_ROOT="/tmp/cb-hoeval-${TAG}-${MODE}-${SLURM_JOB_ID:-$$}"
NODE_RUN_DIR="/data/group_data/rl/yuxiaoq/qixinx/codebase_adaption_runs/${CODEBASE_RUN_ID}"
export CODEBASE_TRAJ_DIR="${NODE_RUN_DIR}/traj"
mkdir -p "$NODE_RUN_DIR"

cd "$SD"
echo "=== eval_heldout_longseq MODE=$MODE TAG=$TAG RUN=$CODEBASE_RUN_ID"
echo "    data=$DATA ckpt=$CKPT dataset=$CODEBASE_EVAL_DATASET sif=$CLBENCH_SIF_DIR on $(hostname) ==="
bash ./launch_codebase_adaption_apptainer.sh
