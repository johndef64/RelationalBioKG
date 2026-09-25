#!/bin/bash
# =============================================================================
# E3 ablation as a Slurm job on the iknos cluster. Submit from the repository root:
#
#   Task A (10 seeds, confirmatory run DTI_v2)
#     sbatch --job-name=e3A --export=ALL,ABL_TASK=A,ABL_VERSION=v2,ABL_RUNS=10 experiments/slurm/e3_ablation.sbatch
#
#   Task B (5 seeds, TREATS_v1)
#     sbatch --job-name=e3B --export=ALL,ABL_TASK=B,ABL_VERSION=v1 experiments/slurm/e3_ablation.sbatch
#
# GPU: any free one (--gres=gpu:1). The cluster types, to pin one add e.g. --gres=gpu:rtx3090:1:
#     gpu:rtx3090:1      iknos-gpu1, RTX 3090, 24 GB
#     gpu:rtx5000ada:1   iknos-gpu2, RTX 5000 Ada, 32 GB
# A job in the low queue can be preempted and requeued, possibly on the other node: every launch
# writes its GPU into the version's MANIFEST.md, so a mixed-hardware version is visible there.
#
# Why ABL_VERSION is always given: when the job is requeued, or resubmitted after a stop, the script
# resumes exactly that version, skipping the variants already complete. Resubmit with the SAME command.
#
# Where the output goes: everything in experiments/ablation/<TASK>_<version>/ (logs, checkpoints,
# summary, manifest, driver.log). This file's own .out only holds the first lines. At the end the job
# packs that folder (checkpoints excluded) and its .out into experiments/slurm/e3_<TASK>_<version>-<jobid>.zip.
# =============================================================================
#SBATCH --job-name=e3
#SBATCH --output=experiments/slurm/%x-%j.out
#SBATCH --error=experiments/slurm/%x-%j.out
#SBATCH --qos=low
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:rtx3090:1      # iknos-gpu1, RTX 3090, 24 GB
##SBATCH --gres=gpu:rtx5000ada:1   # iknos-gpu2, RTX 5000 Ada, 32 GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
rc=0
bash experiments/e3_ablation.sh all || rc=$?

# everything to download in one file: the whole version folder (logs, summary, manifest, driver.log,
# rankings), checkpoints excluded, plus this job's .out. Written also when the run stopped early.
case "$ABL_TASK" in A|a|DTI) T=DTI ;; *) T=TREATS ;; esac
V="${ABL_ROOT:-experiments/ablation}/${T}_${ABL_VERSION}"
python experiments/slurm/zip_results.py "experiments/slurm/e3_${T}_${ABL_VERSION}-${SLURM_JOB_ID}.zip" \
  "$V" "experiments/slurm/${SLURM_JOB_NAME:-e3}-${SLURM_JOB_ID}.out" || true
exit $rc
