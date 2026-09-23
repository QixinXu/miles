#!/usr/bin/env bash
# held-out 长序列评测的 API 模型版(GPT-5.5 等,2026-09-22)。
# 数据 / 判分 / 内存护栏 / 轨迹目录与 eval_heldout_longseq.sh 完全一致,只把模型换成 OpenAI API:
# 驱动是 scripts/agentcl/run_api_agentcl.py --task swe_smith(复用 miles 的 _make_task / ACT / WRITE 构造)。
#
# 不占 GPU。但要在 miles SIF 里跑(import miles + clbench,且要嵌套 apptainer 起每道题的 SIF),
# 所以借 launch_codebase_adaption_apptainer.sh 进容器,容器内再执行本脚本自己(内层分支)。
#
# 用法(在 allocation 的 srun step 里):
#   EVAL_MODE=replace|stateless EVAL_DATA=<eval prompt jsonl> \
#   [EVAL_TAG=gpt55] [API_MODEL=gpt-5.5] [API_CONCURRENCY=12] [API_EXTRA_ARGS="--limit 2"] \
#     bash scripts/eval_heldout_api.sh
# 轨迹 -> codebase_adaption_runs/hoeval-<TAG>-<MODE>-<数据>/traj/eval/heldout/rollout_0/ep_*.json
# 中断后原命令重跑即可(已完成的 episode 跳过)。
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${EVAL_MODE:?EVAL_MODE=replace|stateless}"
DATA="${EVAL_DATA:?EVAL_DATA required (eval prompt jsonl)}"
TAG="${EVAL_TAG:-gpt55}"
RUN_ID="hoeval-${TAG}-${MODE}-$(basename "$DATA" .jsonl | sed 's/^ho_//; s/_replace$//; s/_stateless$//')"
OUT="/data/group_data/rl/yuxiaoq/qixinx/codebase_adaption_runs/${RUN_ID}/traj/eval/heldout/rollout_0"
case "$MODE" in
  replace)   NS=1 ;;
  stateless) NS=4 ;;
  *) echo "unknown EVAL_MODE=$MODE" >&2; exit 1;;
esac

if [ -z "${APPTAINER_CONTAINER:-}" ]; then
  # ── 宿主侧:与 eval_heldout_longseq.sh 相同的路径 / 护栏,然后进 miles SIF ──
  unset SSL_CERT_FILE SSL_CERT_DIR CURL_CA_BUNDLE REQUESTS_CA_BUNDLE NODE_EXTRA_CA_CERTS GIT_SSL_CAINFO PIP_CERT
  [ -f "$DATA" ] || { echo "EVAL_DATA 不存在: $DATA" >&2; exit 1; }
  export CODEBASE_CMD_MEM_LIMIT_KB="${CODEBASE_CMD_MEM_LIMIT_KB:-16777216}"
  export CODEBASE_GRADE_MEM_LIMIT_KB="${CODEBASE_GRADE_MEM_LIMIT_KB:-33554432}"
  source "${SD}/scripts/cluster_babel_env.sh"   # CLBENCH_SIF_DIR / CLBENCH_ROOT / CODEBASE_PYDEPS 等
  SCR="${CODEBASE_SCR:-/scratch/qixinx}"
  mkdir -p "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/tmp"
  export APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache" TMPDIR="$SCR/tmp"
  export APPTAINERENV_APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINERENV_APPTAINER_CACHEDIR="$SCR/apptainer_cache" APPTAINERENV_TMPDIR="$SCR/tmp"
  export CODEBASE_APPTAINER_RUNTIME_ROOT="/tmp/cb-${RUN_ID}-${SLURM_JOB_ID:-$$}"   # 按 RUN_ID 隔离: 两组 replace 并发时共用会撞 cp
  export CODEBASE_INNER_SCRIPT="${SD}/scripts/eval_heldout_api.sh"
  export EVAL_MODE EVAL_DATA EVAL_TAG="$TAG"
  echo "=== eval_heldout_api MODE=$MODE TAG=$TAG RUN=$RUN_ID model=${API_MODEL:-gpt-5.5} n_samples=$NS"
  echo "    data=$DATA sif=$CLBENCH_SIF_DIR on $(hostname) ==="
  exec bash "${SD}/launch_codebase_adaption_apptainer.sh"
fi

# ── 容器内(miles SIF):与 run_codebase_adaption_qwen3.5_4B.sh 相同的 clbench 环境 ──
unset SSL_CERT_FILE SSL_CERT_DIR CURL_CA_BUNDLE REQUESTS_CA_BUNDLE NODE_EXTRA_CA_CERTS GIT_SSL_CAINFO PIP_CERT
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
export CLBENCH_ROOT="${CLBENCH_ROOT:-/home/qixinx/continual-learning-bench}"
export CLBENCH_CONTAINER_BACKEND="${CLBENCH_CONTAINER_BACKEND:-singularity}"
export MSWEA_SINGULARITY_EXECUTABLE="${MSWEA_SINGULARITY_EXECUTABLE:-apptainer}"
PYDEPS="${CODEBASE_PYDEPS:-/data/user_data/qixinx/miles_pydeps/codebase_py312_clean}"
export PYTHONPATH="${PYDEPS}:/home/qixinx/miles:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
OPENAI_API_KEY="$(grep -m1 '^OPENAI_API_KEY=' /home/qixinx/miles/.env | cut -d= -f2- | tr -d "\"'")"
export OPENAI_API_KEY
mkdir -p "$OUT"
cd /home/qixinx/miles
# shellcheck disable=SC2086
exec python3 examples/codebase_adaption/scripts/agentcl/run_api_agentcl.py \
  --task swe_smith --data "$DATA" --out "$OUT" \
  --model "${API_MODEL:-gpt-5.5}" --n-samples "$NS" --concurrency "${API_CONCURRENCY:-12}" \
  ${API_EXTRA_ARGS:-}
