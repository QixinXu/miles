#!/usr/bin/env bash
# held-out 长序列评测的驱动 v2(2026-09-21)。v1 的问题: 组合之间只等 ~50s, 上一批的 Ray/SGLang 没退干净,
# 下一批抢不到 GPU 直接崩(cotrain251 那次 0/208)。v2 在每个组合开始前硬等 GPU 空闲。按 (ckpt, mode, data) 逐个跑,每个都:
#   1. 后台起 eval_heldout_longseq.sh
#   2. 轮询轨迹目录,等到期望条数(replace=行数;stateless=行数×4)
#   3. 落盘后杀掉整棵进程树 —— 不让 eval 之后那个无害训练 rollout 真的跑(照 sbatch_reeval_*.sbatch 的做法)
# 已完成的组合会跳过(轨迹已达期望条数),所以中断后原命令重跑即可。
#
# 用法(在 8 卡 allocation 的 srun step 里):
#   bash scripts/run_heldout_eval_babel.sh [JOB_SPEC ...]
#   JOB_SPEC = "<tag>:<mode>:<data 文件名(在 data/ 下)>"
#   不给参数则跑默认全集:两个 ckpt × 两份数据 × replace/stateless = 8 个组合
set -uo pipefail
unset SSL_CERT_FILE SSL_CERT_DIR CURL_CA_BUNDLE REQUESTS_CA_BUNDLE NODE_EXTRA_CA_CERTS GIT_SSL_CAINFO PIP_CERT
SD=/home/qixinx/miles/examples/codebase_adaption
CKROOT=/data/group_data/rl/yuxiaoq/qixinx/agentcl/ckpt
RUNROOT=/data/group_data/rl/yuxiaoq/qixinx/codebase_adaption_runs
declare -A CKPT=(
  [fmtonly191]="$CKROOT/explore-fmtonly-scaffoldv2-ACTthinking-iter191/ckpt"
  [cotrain251]="$CKROOT/explore-gwin-think-iter251/ckpt"
  [actonly239]="$CKROOT/actonly-think-iter239/ckpt"
  [qwen4b]="$CKROOT/qwen35-4b-base/ckpt"          # Qwen3.5-4B 原始权重(release -> Qwen3.5-4B_torch_dist)
)
POLL=${POLL_SEC:-60}
MAXMIN=${MAX_MIN:-600}

DEFAULT_JOBS=(
  "fmtonly191:replace:ho_2x13_26issues_replace.jsonl"
  "cotrain251:replace:ho_2x13_26issues_replace.jsonl"
  "fmtonly191:replace:ho_3x9_27issues_replace.jsonl"
  "cotrain251:replace:ho_3x9_27issues_replace.jsonl"
  "fmtonly191:stateless:ho_2x13_26issues_stateless.jsonl"
  "cotrain251:stateless:ho_2x13_26issues_stateless.jsonl"
  "fmtonly191:stateless:ho_3x9_27issues_stateless.jsonl"
  "cotrain251:stateless:ho_3x9_27issues_stateless.jsonl"
)
JOBS=("$@"); [ ${#JOBS[@]} -eq 0 ] && JOBS=("${DEFAULT_JOBS[@]}")

wait_gpu_clean () {   # 等到没有残留的 sglang/ray 且显存释放,最多等 $1 分钟
  local maxmin=${1:-10} i=0
  while [ $i -lt $((maxmin * 6)) ]; do
    local apps mem
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
    if [ "$apps" -eq 0 ] && [ "${mem:-0}" -lt 2000 ]; then
      [ $i -gt 0 ] && echo "  GPU 已空闲(等了 $((i*10))s)"
      sleep 10; return 0
    fi
    [ $((i % 6)) -eq 0 ] && echo "  等 GPU 释放… 占卡进程=$apps 显存=${mem}MiB"
    pkill -f "sglang" 2>/dev/null; pkill -f "ray::" 2>/dev/null; pkill -f "raylet" 2>/dev/null
    sleep 10; i=$((i+1))
  done
  echo "  ⚠️ 等了 ${maxmin}min GPU 仍未空闲,继续尝试启动(可能失败)"
  return 1
}

run_one () {
  local tag=$1 mode=$2 dataname=$3
  local data="$SD/data/$dataname"
  local ck="${CKPT[$tag]:-}"
  [ -n "$ck" ] || { echo "❌ 未知 tag=$tag"; return 1; }
  [ -f "$data" ] || { echo "❌ 数据不存在 $data"; return 1; }
  local rows; rows=$(grep -c . "$data")
  local expect=$rows
  [ "$mode" = stateless ] && expect=$((rows * 4))
  local suffix; suffix=$(basename "$dataname" .jsonl | sed 's/^ho_//; s/_replace$//; s/_stateless$//')
  local run_id="hoeval-${tag}-${mode}-${suffix}"
  local trajdir="$RUNROOT/$run_id/traj/eval/heldout/rollout_0"
  local have; have=$(ls "$trajdir"/ep_*.json 2>/dev/null | wc -l)
  echo "════════ $(date +%H:%M) $run_id  期望 $expect 份,已有 $have ════════"
  if [ "$have" -ge "$expect" ]; then echo "  已完成,跳过"; return 0; fi
  wait_gpu_clean 10

  EVAL_MODE=$mode EVAL_DATA="$data" EVAL_CKPT_DIR="$ck" EVAL_TAG="$tag" \
    setsid bash "$SD/scripts/eval_heldout_longseq.sh" > "$SD/logs/${run_id}.log" 2>&1 &
  local pid=$! pgid waited=0
  pgid=$(ps -o pgid= -p $pid 2>/dev/null | tr -d ' ')
  echo "  launcher pid=$pid pgid=${pgid:-?} log=logs/${run_id}.log"
  while :; do
    sleep "$POLL"; waited=$((waited + (POLL + 59) / 60))   # 分钟; 旧写法 +POLL/60+1 每轮记 2 分钟, MAX_MIN 实际只剩一半(9/22 actonly 3x9 因此被提前中止)
    have=$(ls "$trajdir"/ep_*.json 2>/dev/null | wc -l)
    if [ $((waited % 5)) -eq 0 ]; then echo "  [$(date +%H:%M)] $have/$expect"; fi
    if [ "$have" -ge "$expect" ]; then echo "  ✅ $(date +%H:%M) 轨迹齐了($have/$expect),等 90s 落盘后收工"; sleep 90; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "  ⚠️ launcher 已退出,轨迹 $have/$expect,尾部日志:"; tail -6 "$SD/logs/${run_id}.log"; break; fi
    if [ "$waited" -ge "$MAXMIN" ]; then echo "  ⏰ 超时 ${MAXMIN}min,轨迹 $have/$expect,中止本组合"; break; fi
  done
  [ -n "${pgid:-}" ] && kill -TERM -"$pgid" 2>/dev/null
  sleep 20
  [ -n "${pgid:-}" ] && kill -KILL -"$pgid" 2>/dev/null
  pkill -f "sglang" 2>/dev/null; pkill -f "ray::" 2>/dev/null; pkill -f "raylet" 2>/dev/null
  sleep 60
  have=$(ls "$trajdir"/ep_*.json 2>/dev/null | wc -l)
  echo "  收工:$run_id 轨迹 $have/$expect"
}

echo "=== $(date) host=$(hostname) 待跑 ${#JOBS[@]} 个组合 ==="
for spec in "${JOBS[@]}"; do
  IFS=: read -r tag mode dataname <<<"$spec"
  run_one "$tag" "$mode" "$dataname"
done
echo "=== $(date) 全部组合结束 ==="
