#!/bin/bash
# =============================================================================
# Block 1, X0b: smoothing of relatives in the embeddings of every E1 checkpoint (relative contrast in
# the decoder space, Dirichlet energy, effective rank). One forward pass per checkpoint, no training.
# Plan: future_works/04_TODO_blocco1.md, point 0b. Code: experiments/block1/x0b_embedding_smoothing.py
#
# Submit from the repository root:
#     sbatch experiments/block1/slurm/x0b_embedding_smoothing.sbatch                               # run, version v1
#     sbatch --export=ALL,X0B_MODE=check experiments/block1/slurm/x0b_embedding_smoothing.sbatch   # requirements only
#     sbatch --export=ALL,X0B_VERSION=v2 experiments/block1/slurm/x0b_embedding_smoothing.sbatch   # a new version
#
# Needs: the 12 E1 checkpoints + params + rank dumps of the six E1 folders, the task graphs;
#        optional: the Mondo cache written by X0a (adds the Mondo parent/child relatives on Task B).
# Output: experiments/block1/results/X0b_<version>/summary/ (tracked), MANIFEST.md, COMPLETE;
#         experiments/block1/slurm/x0b-<jobid>.zip (git-ignored)
# =============================================================================
#SBATCH --job-name=x0b
#SBATCH --output=experiments/block1/slurm/%x-%j.out
#SBATCH --error=experiments/block1/slurm/%x-%j.out
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f experiments/block1/x0b_embedding_smoothing.py ] || { echo "[x0b] submit from the repository root"; exit 2; }
if [ -f experiments/block1/slurm/site_env.sh ]; then
  source experiments/block1/slurm/site_env.sh
else
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gnn
fi
export PKT_SKIP_CONDA=1 PYTHONHASHSEED=0

MODE=${X0B_MODE:-run}
VERSION=${X0B_VERSION:-v1}
TASKS=${X0B_TASKS:-"DTI TREATS"}
OUT=experiments/block1/results/X0b_${VERSION}
JOB=${SLURM_JOB_ID:-local}
echo "[x0b] $(date '+%F %T') job $JOB on $(hostname) | mode $MODE | version $VERSION | tasks $TASKS"

rc=0
if [ "$MODE" = "check" ]; then
  for t in $TASKS; do python -u experiments/block1/x0b_embedding_smoothing.py --task "$t" --check || rc=$?; done
  [ $rc -eq 0 ] && echo "[x0b] CHECK: ALL OK" || echo "[x0b] CHECK: FAILED (see [FAIL] lines)"
  exit $rc
fi
if [ -f "$OUT/COMPLETE" ]; then
  echo "[x0b] $OUT is COMPLETE: refusing to overwrite. Use X0B_VERSION=v<N+1> for a new version."
  exit 3
fi
mkdir -p "$OUT"
{
  echo "# X0b ${VERSION}"
  echo
  echo "- started: $(date '+%F %T'), job $JOB on $(hostname)"
  echo "- git commit: $(git rev-parse HEAD 2>/dev/null) ($(git status --porcelain --untracked-files=no 2>/dev/null | wc -l) modified tracked files)"
  echo "- tasks: $TASKS"
  echo "- code: experiments/block1/x0b_embedding_smoothing.py (default parameters; see summary/*_provenance.json)"
} > "$OUT/MANIFEST.md"
for t in $TASKS; do
  rt=0
  python -u experiments/block1/x0b_embedding_smoothing.py --task "$t" --out_dir "$OUT" || rt=$?
  echo "- task $t: exit code ${rt}" >> "$OUT/MANIFEST.md"
  [ $rt -eq 0 ] || rc=$rt
done
echo "- finished: $(date '+%F %T')" >> "$OUT/MANIFEST.md"
[ $rc -eq 0 ] && date '+%F %T' > "$OUT/COMPLETE"
echo "[x0b] $(date '+%F %T') done, exit code $rc"
python experiments/slurm/zip_results.py "experiments/block1/slurm/x0b-${JOB}.zip" \
  "$OUT" "experiments/block1/slurm/${SLURM_JOB_NAME:-x0b}-${JOB}.out"
exit $rc
