#!/bin/bash
# 给评测链造一个 "ckpt 占位目录" (shim): EVAL_CKPT_DIR 需要 latest_checkpointed_iteration.txt + iter 目录。
# 用法: make_shim.sh <torch_dist ckpt 目录 (含 latest_checkpointed_iteration.txt)> <shim 输出目录>
#   ACT 走 API 时, 评测链路仍要把 Megatron + sglang 起起来, 本地模型不会被调用, 随便给一个能加载的 ckpt 即可
#   (例如 Qwen3.5-4B base 的 torch_dist 转换结果)。shim 只放符号链接, 不复制权重。
set -euo pipefail
TD="$1"; SH="$2"
[ -f "$TD/latest_checkpointed_iteration.txt" ] || { echo "!!! $TD 没有 latest_checkpointed_iteration.txt"; exit 1; }
N=$(cat "$TD/latest_checkpointed_iteration.txt")
mkdir -p "$SH"; ln -sfn "$TD/$N" "$SH/$N"; echo -n "$N" > "$SH/latest_checkpointed_iteration.txt"
echo "shim=$SH -> $TD/$N"
