#!/bin/bash
# SWE-smith pass@k 采集 job(preempt, 4 卡可跑; 被抢占会 REQUEUE, LR0 幂等无损)。
#SBATCH -A aviralku
#SBATCH -p preempt
#SBATCH -q preempt_qos
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/home/qixinx/miles/examples/codebase_adaption/logs/%x-%j.out
set -uo pipefail
export PYTHONUNBUFFERED=1
SD=/home/qixinx/miles/examples/codebase_adaption
echo "=== $(date) job=$SLURM_JOB_ID node=$(hostname) SWE-smith pass@k ==="
cd "$SD"
bash scripts/passk_swesmith_pilot.sh
