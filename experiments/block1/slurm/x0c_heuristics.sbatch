#!/bin/bash
# =============================================================================
# Block 1, X0c: classical link-prediction heuristics (popularity, CN, AA, RA, Katz-3, PPR, typed
# "relatives of the known targets") under the E1 protocol. No training; GPU used for propagation.
# Plan: future_works/04_TODO_blocco1.md, point 0c. Code: experiments/block1/x0c_heuristics.py
#
# Submit from the repository root:
#     sbatch experiments/block1/slurm/x0c_heuristics.sbatch                               # run, version v1
#     sbatch --export=ALL,X0C_MODE=check experiments/block1/slurm/x0c_heuristics.sbatch   # requirements only
#     sbatch --export=ALL,X0C_VERSION=v2 experiments/block1/slurm/x0c_heuristics.sbatch   # a new version
#     sbatch --export=ALL,X0C_TASKS=DTI  experiments/block1/slurm/x0c_heuristics.sbatch   # one task only
#
# Needs: the E1 rank dumps (for alignment and strata; see x0a_dump_ranks.sbatch) and params, the task graphs.
# Output: experiments/block1/results/X0c_<version>/summary/ (tracked), ranks/ (per-triple ranks, tracked),
#         MANIFEST.md, COMPLETE; experiments/block1/slurm/x0c-<jobid>.zip (git-ignored)
# =============================================================================
#SBATCH --job-name=x0c
#SBATCH --output=experiments/block1/slurm/%x-%j.out
#SBATCH --error=experiments/block1/slurm/%x-%j.out
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f experiments/block1/x0c_heuristics.py ] || { echo "[x0c] submit from the repository root"; exit 2; }
if [ -f experiments/block1/slurm/site_env.sh ]; then
  source experiments/block1/slurm/site_env.sh
else
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gnn
fi
export PKT_SKIP_CONDA=1 PYTHONHASHSEED=0

MODE=${X0C_MODE:-run}
VERSION=${X0C_VERSION:-v1}
TASKS=${X0C_TASKS:-"DTI TREATS"}
OUT=experiments/block1/results/X0c_${VERSION}
JOB=${SLURM_JOB_ID:-local}
echo "[x0c] $(date '+%F %T') job $JOB on $(hostname) | mode $MODE | version $VERSION | tasks $TASKS"

rc=0
if [ "$MODE" = "check" ]; then
  for t in $TASKS; do python -u experiments/block1/x0c_heuristics.py --task "$t" --check || rc=$?; done
  [ $rc -eq 0 ] && echo "[x0c] CHECK: ALL OK" || echo "[x0c] CHECK: FAILED (see [FAIL] lines)"
  exit $rc
fi
if [ -f "$OUT/COMPLETE" ]; then
  echo "[x0c] $OUT is COMPLETE: refusing to overwrite. Use X0C_VERSION=v<N+1> for a new version."
  exit 3
fi
mkdir -p "$OUT"
{
  echo "# X0c ${VERSION}"
  echo
  echo "- started: $(date '+%F %T'), job $JOB on $(hostname)"
  echo "- git commit: $(git rev-parse HEAD 2>/dev/null) ($(git status --porcelain --untracked-files=no 2>/dev/null | wc -l) modified tracked files)"
  echo "- tasks: $TASKS"
  echo "- code: experiments/block1/x0c_heuristics.py (default parameters; see summary/*_provenance.json)"
} > "$OUT/MANIFEST.md"
for t in $TASKS; do
  rt=0
  python -u experiments/block1/x0c_heuristics.py --task "$t" --out_dir "$OUT" || rt=$?
  echo "- task $t: exit code ${rt}" >> "$OUT/MANIFEST.md"
  [ $rt -eq 0 ] || rc=$rt
done
echo "- finished: $(date '+%F %T')" >> "$OUT/MANIFEST.md"
[ $rc -eq 0 ] && date '+%F %T' > "$OUT/COMPLETE"
echo "[x0c] $(date '+%F %T') done, exit code $rc"
python experiments/slurm/zip_results.py "experiments/block1/slurm/x0c-${JOB}.zip" \
  "$OUT" "experiments/block1/slurm/${SLURM_JOB_NAME:-x0c}-${JOB}.out"
exit $rc
