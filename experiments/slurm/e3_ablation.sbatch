#!/bin/bash
# =============================================================================
# E3 ablation as a Slurm job on the iknos cluster. Submit from the repository root:
#
#   Task A (10 seeds, confirmatory run DTI_v2, on the RTX 3090 node)
#     sbatch --job-name=e3A --gres=gpu:rtx3090:1 \
#            --export=ALL,ABL_TASK=A,ABL_VERSION=v2,ABL_RUNS=10 experiments/slurm/e3_ablation.sbatch
#
#   Task B (5 seeds, TREATS_v1, on the RTX 5000 Ada node)
#     sbatch --job-name=e3B --gres=gpu:rtx5000ada:1 \
#            --export=ALL,ABL_TASK=B,ABL_VERSION=v1 experiments/slurm/e3_ablation.sbatch
#
# Why the GPU type is pinned: a version must run on one kind of hardware from start to end. A job in
# the low queue can be preempted and requeued, and without the pin it could restart on the other node.
#
# Why ABL_VERSION is always given: when the job is requeued, or resubmitted after a stop, the script
# resumes exactly that version, skipping the variants already complete. Resubmit with the SAME command.
#
# Where the output goes: everything in experiments/ablation/<TASK>_<version>/ (logs, checkpoints,
# summary, manifest, driver.log). This file's own .out only holds the first lines.
# =============================================================================
#SBATCH --job-name=e3
#SBATCH --output=experiments/slurm/%x-%j.out
#SBATCH --error=experiments/slurm/%x-%j.out
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --requeue

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

source /home/apps/iknos-env.sh
iknos_activate gnn
export PKT_SKIP_CONDA=1          # keep experiments/config.sh from switching to another conda env

: "${ABL_TASK:?set ABL_TASK=A or B with --export}"
: "${ABL_VERSION:?set ABL_VERSION (e.g. v2) with --export, so that a requeue resumes the same version}"
export PROTOCOL=v2

echo "[job] $(date '+%F %T') $SLURM_JOB_ID on $(hostname) | task $ABL_TASK version $ABL_VERSION runs ${ABL_RUNS:-5}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash experiments/e3_ablation.sh all
