#!/usr/bin/env bash
# Sequential 19q eval of actonly iter_63 and iter_87 (replace + stateless each).
# Pattern: pointer dirs pin the iteration (symlink + latest_checkpointed_iteration.txt),
# eval_ckpt_19q_orchard.sh does a weights-only eval-before-train run per mode.
# Usage (flame-20): setsid nohup bash scripts/eval_actonly_63_87.sh > logs/eval-actonly-63-87.driver.log 2>&1 &
set -uo pipefail
SD=/home/qixinx/miles/examples/codebase_adaption
SRC=/tmp/qixinx/runs/smith-4b-v3curr-actonly-v3think/ckpt
PTR=/tmp/qixinx/eval_ptr

for it in 63 87; do
  d=$PTR/actonly_iter$it
  mkdir -p "$d"
  ln -sfn "$SRC/iter_$(printf %07d "$it")" "$d/iter_$(printf %07d "$it")"
  echo "$it" > "$d/latest_checkpointed_iteration.txt"
done

cleanup() {
  ray stop --force >/dev/null 2>&1
  pkill -9 -f "sglang::schedule[r]" 2>/dev/null
  pkill -9 -f "gcs_serve[r]" 2>/dev/null
  pkill -9 -f "rayle[t]" 2>/dev/null
  sleep 3
  rm -rf /tmp/qixinx/tmp/ray
}

for it in 63 87; do
  for mode in replace stateless; do
    echo "=== START iter$it $mode $(date) ==="
    cleanup
    EVAL_MODE=$mode EVAL_CKPT_DIR=$PTR/actonly_iter$it EVAL_TAG=actonly-iter$it \
      bash "$SD/scripts/eval_ckpt_19q_orchard.sh" > "$SD/logs/eval-actonly-iter$it-$mode.log" 2>&1
    echo "=== DONE iter$it $mode rc=$? $(date) ==="
  done
done
cleanup
echo "=== ALL_EVALS_DONE ==="
