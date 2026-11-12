#!/bin/bash
# =============================================================================
# Block 1, prerequisite of X0a: rebuild the per-seed test ranks of the six E1 runs from their
# checkpoints, with experiments/dump_test_ranks.py (release v1.0.0, unchanged). GPU job.
#
# Submit from the repository root, then chain X0a on it:
#     J=$(sbatch --parsable experiments/block1/slurm/x0a_dump_ranks.sbatch)
#     sbatch --dependency=afterok:$J experiments/block1/slurm/x0a_distance_curve.sbatch
#
# Needs: <E1 folder>/<model>_run0.pt ... _run11.pt and <model>_params.json in the six folders below.
# Ranks already present are skipped (--skip_existing), so a resubmission resumes.
# Output (git-ignored, the same place as in E5):
#     <E1 folder>/ranks/test_ranks_<TASK>_run<i>.csv     each one checks its own E1 MRR
#     experiments/block1/cache/dump_logs/<TASK>_<model>.log
#     experiments/block1/slurm/x0a_dump-<jobid>.zip      ranks + logs + this .out (no checkpoints)
# Time: Task A minutes, Task B about 3-4 h (the three models one after the other).
# =============================================================================
#SBATCH --job-name=x0a_dump
#SBATCH --output=experiments/block1/slurm/%x-%j.out
#SBATCH --error=experiments/block1/slurm/%x-%j.out
#SBATCH --qos=low
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:rtx3090:1      # 24 GB
##SBATCH --gres=gpu:rtx5000ada:1   # 32 GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f experiments/dump_test_ranks.py ] || { echo "[dump] submit from the repository root"; exit 2; }
if [ -f experiments/block1/slurm/site_env.sh ]; then
  source experiments/block1/slurm/site_env.sh
else
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gnn
fi
export PKT_SKIP_CONDA=1 PYTHONHASHSEED=0
# scored (triple, candidate) pairs per GPU call; results do not depend on it (as in stratified_taskB.sbatch)
export PKT_EVAL_BATCH_ROWS=${PKT_EVAL_BATCH_ROWS:-1000000}

JOB=${SLURM_JOB_ID:-local}
LOGS=experiments/block1/cache/dump_logs
mkdir -p "$LOGS"
echo "[dump] $(date '+%F %T') job $JOB on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# the six E1 folders read by experiments/block1/x0a_distance_curve.py
RUNS="DTI:rgcn:models/dti_pkt_taskA_dti.tsv_20260918_130755
DTI:compgcn:models/dti_pkt_taskA_dti.tsv_20260918_135629
DTI:distmult:models/dti_pkt_taskA_dti.tsv_20260918_153101
TREATS:rgcn:models/treats_pkt_taskB_treats.tsv_20260918_151621
TREATS:compgcn:models/treats_pkt_taskB_treats.tsv_compgcn_20260920_123201
TREATS:distmult:models/treats_pkt_taskB_treats.tsv_distmult_20260920_212827"

# 1. requirements: 12 checkpoints + params in every folder, before any GPU work
ok=1
while IFS=: read -r task model folder; do
  n=$(ls "$folder"/${model}_run*.pt 2>/dev/null | wc -l)
  p=$(ls "$folder"/${model}_params.json 2>/dev/null | wc -l)
  echo "[dump] $task $model: $n checkpoints, $p params in $folder"
  { [ "$n" -ge 12 ] && [ "$p" -eq 1 ]; } || { echo "[dump] FAIL: $folder needs ${model}_run0..11.pt and ${model}_params.json"; ok=0; }
done <<< "$RUNS"
[ $ok -eq 1 ] || { echo "[dump] requirements not met: nothing computed"; exit 1; }

# 2. ranks, skipping those already present
rc=0
while IFS=: read -r task model folder; do
  tsv=dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip
  [ "$task" = TREATS ] && tsv=dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip
  log="$LOGS/${task}_${model}.log"
  echo "[dump] $(date '+%F %T') $task $model -> $folder/ranks"
  python -u experiments/dump_test_ranks.py --model_folder "$folder" --tsv "$tsv" --task "$task" \
    --run all --skip_existing --quiet >> "$log" 2>&1 || { echo "[dump] FAIL $task $model (see $log)"; rc=1; }
  grep -E "\-\-\-\- run|MRR from|Error|MISMATCH|nothing to do" "$log" | tail -14
  echo "[dump] $task $model: $(ls "$folder"/ranks/test_ranks_${task}_run*.csv 2>/dev/null | wc -l) rank files"
done <<< "$RUNS"
echo "[dump] $(date '+%F %T') done, exit code $rc"

python experiments/slurm/zip_results.py "experiments/block1/slurm/x0a_dump-${JOB}.zip" "$LOGS" \
  $(while IFS=: read -r t m f; do echo "$f/ranks"; done <<< "$RUNS") \
  "experiments/block1/slurm/${SLURM_JOB_NAME:-x0a_dump}-${JOB}.out"
exit $rc
