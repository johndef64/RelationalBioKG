#!/bin/bash
# =============================================================================
# Stratified analysis of Task B over all 12 E1 seeds of each model. Submit from the repository root:
#
#     sbatch experiments/slurm/stratified_taskB.sbatch
#
# Needs the E1 Task B checkpoints in their folders (<model>_run0.pt ... _run11.pt):
#     models/treats_pkt_taskB_treats.tsv_20260918_151621            R-GCN
#     models/treats_pkt_taskB_treats.tsv_compgcn_20260920_123201    CompGCN
#     models/treats_pkt_taskB_treats.tsv_distmult_20260920_212827   DistMult
# The models are ranked one after the other (about 9 GB of RAM each). Output:
#     <model folder>/ranks/test_ranks_TREATS_run<i>.csv   (per-seed rank dumps; each checks its E1 MRR)
#     experiments/logs/v2/dump_ranks_TREATS_allseeds_<model>.log
#     experiments/logs/v2/stratified_multiseed_TREATS.md (+ _per_seed.csv)
#     experiments/slurm/stratB-<jobid>.zip   ALL of the above plus this job's .out, to download
#                                            (checkpoints excluded)
# =============================================================================
#SBATCH --job-name=stratB
#SBATCH --output=experiments/slurm/%x-%j.out
#SBATCH --error=experiments/slurm/%x-%j.out
#SBATCH --qos=low
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:rtx3090:1      # iknos-gpu1, RTX 3090, 24 GB
##SBATCH --gres=gpu:rtx5000ada:1   # iknos-gpu2, RTX 5000 Ada, 32 GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source /home/apps/iknos-env.sh
iknos_activate gnn
export PKT_SKIP_CONDA=1 PYTHONHASHSEED=0
# scored (triple, candidate) pairs per GPU call. The default 250k is sized for small GPUs; 1M uses about
# 5-6 GB for DistMult/R-GCN at dim 400 and fits both cluster GPUs (24 and 32 GB). The results do not
# depend on it: each triple is scored on its own.
export PKT_EVAL_BATCH_ROWS=1000000

TSV=dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip
declare -A F=(
  [rgcn]=models/treats_pkt_taskB_treats.tsv_20260918_151621
  [compgcn]=models/treats_pkt_taskB_treats.tsv_compgcn_20260920_123201
  [distmult]=models/treats_pkt_taskB_treats.tsv_distmult_20260920_212827
)
echo "[stratB] $(date '+%F %T') job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for m in rgcn compgcn distmult; do
  n=$(ls "${F[$m]}"/${m}_run*.pt 2>/dev/null | wc -l)
  echo "[stratB] $m: $n checkpoints in ${F[$m]}"
  [ "$n" -gt 0 ] || { echo "[stratB] ERROR: no checkpoints for $m"; exit 1; }
  python -u experiments/dump_test_ranks.py --model_folder "${F[$m]}" --tsv "$TSV" --task TREATS \
    --run all --skip_existing --quiet > "experiments/logs/v2/dump_ranks_TREATS_allseeds_${m}.log" 2>&1
  grep -E "\-\-\-\- run|MRR from|Error|MISMATCH" "experiments/logs/v2/dump_ranks_TREATS_allseeds_${m}.log"
done

python -u experiments/stratified_multiseed.py --task TREATS --tsv "$TSV" \
  --model "R-GCN=${F[rgcn]}" --model "CompGCN=${F[compgcn]}" --model "DistMult=${F[distmult]}" \
  --reference DistMult --out experiments/logs/v2/stratified_multiseed_TREATS.md \
  | grep -v " run .*: MRR"
echo "[stratB] $(date '+%F %T') done"

# everything to download in one file (checkpoints excluded)
python experiments/slurm/zip_results.py "experiments/slurm/stratB-${SLURM_JOB_ID}.zip" \
  experiments/logs/v2/stratified_multiseed_TREATS.md experiments/logs/v2/stratified_multiseed_TREATS_per_seed.csv \
  experiments/logs/v2/dump_ranks_TREATS_allseeds_rgcn.log experiments/logs/v2/dump_ranks_TREATS_allseeds_compgcn.log \
  experiments/logs/v2/dump_ranks_TREATS_allseeds_distmult.log \
  "${F[rgcn]}/ranks" "${F[compgcn]}/ranks" "${F[distmult]}/ranks" \
  "experiments/slurm/${SLURM_JOB_NAME:-stratB}-${SLURM_JOB_ID}.out"
