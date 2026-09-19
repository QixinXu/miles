#!/usr/bin/env bash
# WRITE=gated + prompt v3 + WRITE 专属 thinking(orchard 首个正式组, 2026-07-19)。
# 与 train_4b_write_gated_6p6.sh 的差异, 仅三点:
#   1. CODEBASE_WRITE_PROMPT_V3=1  —— 三明治框架语+成败反思引导(离线 A/B: 串台
#      12-16/30→2-4/30, 严格合规 16-17/30→23-24/30, 见 notes/WRITE_PROMPT_V3_AB_0719.md);
#   2. CODEBASE_WRITE_THINKING=1   —— WRITE 专属 think 块, 只训不外泄, 未闭合回退旧记忆;
#   3. 存储全走节点本地 /tmp(orchard: /project/flame 满 + /home 25G 配额):
#      ckpt/HF 导出/traj 均在 /tmp/qixinx/runs/<RUN_ID>/ 下, 依赖 28 天占位 job
#      锁住本节点; 节点重启即丢, 属已接受风险(占位在, ckpt 可断点续训)。
# 用法(在占位节点上, 如 orchard-flame-31):
#   cd ~/miles/examples/codebase_adaption
#   nohup bash scripts/train_4b_write_gated_v3think_6p6.sh > logs/swecl-4b-write-gated-v3think-6p6.console.log 2>&1 &
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export CODEBASE_WRITE_REWARD_MODE=gated_downstream
export CODEBASE_WRITE_FORMAT_BONUS=0.1
export CODEBASE_WRITE_PROMPT_V3=1
export CODEBASE_WRITE_THINKING=1
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1        # 手动 nohup 同样受 dashboard 抖动威胁; 顺带实弹验收合并后的监督修复
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

export CODEBASE_RUN_ID="swecl-4b-write-gated-v3think-6p6"
export CODEBASE_WANDB_RUN_ID="${CODEBASE_RUN_ID}"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swecl-4b-textfmt"

export CODEBASE_NUM_ROLLOUT=120
export CODEBASE_EVAL_INTERVAL=8
export CODEBASE_SAVE_INTERVAL=8
export CODEBASE_SEQ_LEN=24576
# 24576→20480(2026-07-19): rollout_9 反向 OOM(GPU7 差 ~3.3G)——thinking 拉长
# WRITE 样本, 6p6 原值在 v3think 下过线。降 1/6 换显存余量, 代价是略多的梯度累积。
export CODEBASE_MAX_TOK_PER_GPU=20480
export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"

# 节点本地大文件落点(先于 cluster env 设置, 覆盖其 /home 缺省)
NODE_RUN_DIR="/tmp/qixinx/runs/${CODEBASE_RUN_ID}"
export CODEBASE_SAVE_DIR="${NODE_RUN_DIR}/ckpt"
export CODEBASE_TRAJ_DIR="${NODE_RUN_DIR}/traj"
export CODEBASE_TORCH_DIST="/tmp/qixinx/models/Qwen3.5-4B_torch_dist"
mkdir -p "${NODE_RUN_DIR}"

source "${SD}/scripts/cluster_orchard_env.sh"   # 其余 orchard 路径(SIF/pydeps/HF_CKPT/CLBENCH)
export CODEBASE_PROMPT_DATA="${SD}/data/episodes_6p6_hard.jsonl"
export CODEBASE_BASELINE_ARTIFACT="${SD}/data/baseline_4b_textfmt.json"
export CODEBASE_NGPU=8 CODEBASE_TP=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SCR="${CODEBASE_SCR:-/tmp/qixinx}"
mkdir -p "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/tmp"
export APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache" TMPDIR="$SCR/tmp"
export APPTAINERENV_APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINERENV_APPTAINER_CACHEDIR="$SCR/apptainer_cache" APPTAINERENV_TMPDIR="$SCR/tmp"

# torch_dist 现场生成(幂等, 已存在则秒过)
bash "${SD}/scripts/ensure_torch_dist_local.sh"

cd "$SD"
echo "=== GATED+V3+THINK 6p6 RUN=${CODEBASE_RUN_ID} on $(hostname) (code: $SD; ckpt/traj: ${NODE_RUN_DIR}) ==="
bash ./launch_codebase_adaption_apptainer.sh
