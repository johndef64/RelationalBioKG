#!/bin/bash
# =============================================================================
# Minimal job to check that the cluster works for this project (about 2-3 minutes).
# Submit from the repository root:
#
#     sbatch experiments/slurm/hello.sbatch
#     squeue -u $USER
#     cat experiments/slurm/hello-<jobid>.out
#
# It checks: environment activation, visible GPU, torch + PyG on CUDA, dataset present,
# and a 5-epoch training on Task A (dry run: it saves nothing). Last line: ALL OK or FAILED.
# =============================================================================
#SBATCH --job-name=hello
#SBATCH --output=experiments/slurm/%x-%j.out
#SBATCH --error=experiments/slurm/%x-%j.out
#SBATCH --qos=low
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:rtx3090:1      # iknos-gpu1, RTX 3090, 24 GB
##SBATCH --gres=gpu:rtx5000ada:1   # iknos-gpu2, RTX 5000 Ada, 32 GB
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
ok=1
step () { echo; echo "---- $1"; }

step "1. node and GPU"
echo "job $SLURM_JOB_ID on $(hostname), $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || ok=0

step "2. environment"
source /home/apps/iknos-env.sh && iknos_activate gnn || ok=0
export PKT_SKIP_CONDA=1 PYTHONHASHSEED=0
which python

step "3. torch and PyG on CUDA"
python -c "import torch, torch_geometric as g; x=torch.randn(1000,1000,device='cuda'); print('torch', torch.__version__, '| PyG', g.__version__, '| cuda ok, matmul sum', float((x@x).sum()))" || ok=0

step "4. dataset"
ls -lh dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip || ok=0

step "5. training, 5 epochs, dry run"
python train_and_eval.py --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task DTI --model rgcn \
  --config PKT-DTI-best-v2b --runs 1 --epochs 5 --dry_run --negative_sampling filtered --eval_filtered \
  --split_seed 42 --select_metric mixed --deterministic 2>&1 | grep -E "GPU:|Deterministic|\[val\]|Test MRR|Error|error" \
  || ok=0

echo
[ "$ok" -eq 1 ] && echo "ALL OK" || echo "FAILED (look at the step above the error)"
