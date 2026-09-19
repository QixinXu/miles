#!/bin/bash
# SWE-smith v3-curriculum training on a flame node (queued behind account limit).
# Runs the full experiment; if training exits (success or crash) the job keeps
# the node alive with a sleep so the allocation is not lost — inspect, rsync
# checkpoints, or relaunch manually via ssh before releasing.
#SBATCH -A aviralku
#SBATCH -p flame
#SBATCH -q flame-64gpu_qos
#SBATCH -J swecl-4b-smith-v3
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=0
#SBATCH --time=28-00:00:00
#SBATCH --output=/home/qixinx/miles/examples/codebase_adaption/logs/%x-%j.out
set -uo pipefail
export PYTHONUNBUFFERED=1
SD=/home/qixinx/miles/examples/codebase_adaption
echo "=== $(date) job=$SLURM_JOB_ID node=$(hostname) SMITH v3-CURRICULUM ==="

source /home/qixinx/.wandb.env 2>/dev/null || true
bash "$SD/scripts/train_4b_smith_v3curr.sh" \
  > "$SD/logs/smith-4b-v3curr-gated-v3think.console.log" 2>&1
rc=$?
echo "=== $(date) training exited rc=$rc — holding node (sleep) for inspection/relaunch ==="
sleep infinity
