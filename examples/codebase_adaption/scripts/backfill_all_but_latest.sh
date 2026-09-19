#!/usr/bin/env bash
# Start detached hf_backfill_nice.sh workers for a run, covering every local
# ckpt except the current latest (which the live run may still need) and
# skipping iterations already on the Hub. Iterations are dealt round-robin to
# JOBS parallel workers (each keeps the nice/ionice + page-cache eviction +
# cgroup high-water guard of hf_backfill_nice.sh); all workers append to LOG.
# Refuses to start if a backfill for this run is already running.
# Usage (on the node holding the run):
#   bash scripts/backfill_all_but_latest.sh <RUN_ID> <LOG> [JOBS=1]
set -uo pipefail
R="${1:?RUN_ID}"; LOG="${2:?log path}"; JOBS="${3:-1}"
if pgrep -f "[h]f_backfill_nice.sh ${R} " >/dev/null; then
  echo "backfill for ${R} already running; not starting another"; exit 1
fi
L=$(cat "/tmp/qixinx/runs/${R}/ckpt/latest_checkpointed_iteration.txt")
mapfile -t ALL < <(ls -d /tmp/qixinx/runs/"${R}"/ckpt/iter_* | xargs -n1 basename | grep -v "iter_0*${L}$" | sort -r)

export HF_TOKEN=$(grep HF_TOKEN /home/qixinx/miles/.env | cut -d= -f2)
mapfile -t DONE < <(python3 - "Racktic/swecl-qwen35-ckpt" "${R}" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, run = sys.argv[1], sys.argv[2]
files = HfApi(token=os.environ["HF_TOKEN"]).list_repo_files(repo)
for it in sorted({f.split("/")[1] for f in files if f.startswith(run + "/") and f.endswith(".safetensors")}):
    print(it)
PY
)
TODO=()
for it in "${ALL[@]}"; do [[ " ${DONE[*]} " == *" ${it} "* ]] || TODO+=("${it}"); done
echo "latest=${L} excluded; ${#ALL[@]} local, ${#DONE[@]} on Hub, ${#TODO[@]} to do, ${JOBS} worker(s)"
[ "${#TODO[@]}" -eq 0 ] && exit 0

cd "$(dirname "$0")/.."
for ((w = 0; w < JOBS; w++)); do
  part=()
  for ((i = w; i < ${#TODO[@]}; i += JOBS)); do part+=("${TODO[$i]}"); done
  [ "${#part[@]}" -eq 0 ] && continue
  echo "worker ${w}: ${part[*]}"
  setsid nohup bash scripts/hf_backfill_nice.sh "${R}" "${part[@]}" >> "${LOG}" 2>&1 < /dev/null &
done
