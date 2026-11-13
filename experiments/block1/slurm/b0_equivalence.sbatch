#!/bin/bash
# =============================================================================
# Block 1, B0: prerequisite of phase B. train_b1.py must train bit for bit what train_and_eval.py
# (v1.0.0) trains: E1 R-GCN, seed 42, E1 flags, --deterministic; three arms per task (v1, b1, b1reg).
# Plan: future_works/04_TODO_blocco1.md, section 4. Code: experiments/block1/test_equivalence.py
#
# Submit from the repository root:
#     sbatch --export=ALL,B0_MODE=check experiments/block1/slurm/b0_equivalence.sbatch   # requirements only
#     sbatch experiments/block1/slurm/b0_equivalence.sbatch                               # run, version v1
#     sbatch --export=ALL,B0_TASKS=DTI experiments/block1/slurm/b0_equivalence.sbatch     # one task
#
# Needs: the E1 R-GCN folders (params + metrics), the task graphs, a GPU.
# Cost: 3 trainings per task, E1 length (Task A ~4 min per training on the E1 GPU, Task B ~22 min,
#       more under --deterministic).
# Output: experiments/block1/results/B0_<version>/summary/b0_<task>.{md,json} (tracked), runs/ (logs,
#         metrics; checkpoints git-ignored), MANIFEST.md, COMPLETE only if every task PASSES;
#         experiments/block1/slurm/b0-<jobid>.zip (git-ignored, no .pt)
# =============================================================================
#SBATCH --job-name=b0
#SBATCH --output=experiments/block1/slurm/%x-%j.out
#SBATCH --error=experiments/block1/slurm/%x-%j.out
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f experiments/block1/test_equivalence.py ] || { echo "[b0] submit from the repository root"; exit 2; }
if [ -f experiments/block1/slurm/site_env.sh ]; then
  source experiments/block1/slurm/site_env.sh
else
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gnn
fi
export PKT_SKIP_CONDA=1 PYTHONHASHSEED=0

MODE=${B0_MODE:-run}
VERSION=${B0_VERSION:-v1}
TASKS=${B0_TASKS:-"DTI TREATS"}
OUT=experiments/block1/results/B0_${VERSION}
JOB=${SLURM_JOB_ID:-local}
echo "[b0] $(date '+%F %T') job $JOB on $(hostname) | mode $MODE | version $VERSION | tasks $TASKS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

rc=0
if [ "$MODE" = "check" ]; then
  for t in $TASKS; do python -u experiments/block1/test_equivalence.py --task "$t" --check || rc=$?; done
  [ $rc -eq 0 ] && echo "[b0] CHECK: ALL OK" || echo "[b0] CHECK: FAILED (see [FAIL] lines)"
  exit $rc
fi
if [ -f "$OUT/COMPLETE" ]; then
  echo "[b0] $OUT is COMPLETE: refusing to overwrite. Use B0_VERSION=v<N+1> for a new version."
  exit 3
fi
mkdir -p "$OUT"
{
  echo "# B0 ${VERSION}"
  echo
  echo "- started: $(date '+%F %T'), job $JOB on $(hostname)"
  echo "- git commit: $(git rev-parse HEAD 2>/dev/null) ($(git status --porcelain --untracked-files=no 2>/dev/null | wc -l) modified tracked files)"
  echo "- gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "- tasks: $TASKS"
  echo "- code: experiments/block1/test_equivalence.py, train_b1.py, b1_models.py (E1 flags, seed 42, --deterministic)"
} > "$OUT/MANIFEST.md"
for t in $TASKS; do
  rt=0
  python -u experiments/block1/test_equivalence.py --task "$t" --out_dir "$OUT" || rt=$?
  echo "- task $t: exit code ${rt} ($([ $rt -eq 0 ] && echo PASS || echo FAIL))" >> "$OUT/MANIFEST.md"
  [ $rt -eq 0 ] || rc=$rt
done
echo "- finished: $(date '+%F %T')" >> "$OUT/MANIFEST.md"
[ $rc -eq 0 ] && date '+%F %T' > "$OUT/COMPLETE"
echo "[b0] $(date '+%F %T') done, exit code $rc"
python experiments/slurm/zip_results.py "experiments/block1/slurm/b0-${JOB}.zip" \
  "$OUT" "experiments/block1/slurm/${SLURM_JOB_NAME:-b0}-${JOB}.out"
exit $rc
