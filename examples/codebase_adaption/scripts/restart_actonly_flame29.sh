#!/usr/bin/env bash
# Crash recovery for the actonly+explore arm on orchard-flame-29 (2026-09-13 01:21
# SIGKILL; job cgroup memory.current pinned at the 1814G limit by ckpt page cache).
# Run ON flame-29 inside the existing allocation (pam_slurm_adopt joins the job cgroup):
#   ssh orchard-flame-29 bash /home/qixinx/miles/examples/codebase_adaption/scripts/restart_actonly_flame29.sh
# Same launch command as the original run; no --start-rollout-id (resumes from
# latest_checkpointed_iteration.txt = 103 automatically).
set -u
RUN=smith-4b-v3nocurr-explore-actonly-scaffoldv2-ACTthinking
LOG=/tmp/qixinx/$RUN.console.log
CG=/sys/fs/cgroup/system.slice/slurmstepd.scope/s8FXM24FE64B00

echo "== 1. stop the stale launcher chain + orphaned Ray head =="
# Bracketed first chars so this script's own command line never matches itself.
for pat in "train_4b_v3nocurr_explore_[a]ctonly_think.sh" "run_codebase_adaption_qwen3.5_[4]B.sh" \
           "apptainer/libexec/[s]tarter" "gcs_[s]erver" "/[r]aylet " "dist-packages/[r]ay/" "[r]ay::"; do
  pids=$(pgrep -u qixinx -f "$pat" || true)
  [ -n "$pids" ] && { echo "  kill ($pat): $pids"; kill -TERM $pids 2>/dev/null; }
done
sleep 5
left=$(pgrep -u qixinx -f "gcs_[s]erver|/[r]aylet |dist-packages/[r]ay/|[r]ay::|apptainer/libexec/[s]tarter" || true)
[ -n "$left" ] && { echo "  still alive, SIGKILL: $left"; kill -KILL $left 2>/dev/null; sleep 2; }
echo "  my procs now: $(ps -u qixinx --no-headers | wc -l); gpu mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"

echo "== 2. rotate the crashed console log =="
if [ -f "$LOG" ]; then
  mv "$LOG" "/tmp/qixinx/$RUN.console.crash1-r105-pagecache.log" && echo "  -> $RUN.console.crash1-r105-pagecache.log"
fi

echo "== 3. cgroup memory before launch =="
awk '{printf "  memory.current = %.0f G (limit 1814 G)\n", $1/1e9}' "$CG/memory.current"

echo "== 4. relaunch (original command) =="
cd /home/qixinx/miles/examples/codebase_adaption || exit 1
setsid nohup bash scripts/train_4b_v3nocurr_explore_actonly_think.sh > "$LOG" 2>&1 < /dev/null &
disown
sleep 15
ps -u qixinx -o pid,etimes,args | grep -E "explore_[a]ctonly_think.sh|run_codebase_adaption_qwen3.5_[4]B" | cut -c1-100
echo "== log head =="
grep -v '^\s*$' "$LOG" | head -8 | cut -c1-150
echo "done; watch: tail -f $LOG"
