#!/bin/bash
# =============================================================================
# Block 1, X0a: MRR by distance from the supervision (contains step 0 of X7). CPU only, no training.
# Plan: future_works/04_TODO_blocco1.md, point 0a. Code: experiments/block1/x0a_distance_curve.py
#
# Submit from the repository root:
#     sbatch experiments/block1/slurm/x0a_distance_curve.sbatch                               # run, version v1
#     sbatch --export=ALL,X0A_MODE=check experiments/block1/slurm/x0a_distance_curve.sbatch   # requirements only
#     sbatch --export=ALL,X0A_VERSION=v2 experiments/block1/slurm/x0a_distance_curve.sbatch   # a new version
#     sbatch --export=ALL,X0A_TASKS=DTI  experiments/block1/slurm/x0a_distance_curve.sbatch   # one task only
#
# Needs (checked by the script before computing anything):
#   * the E1 rank dumps models/<E1 folder>/ranks/test_ranks_<TASK>_run0..11.csv and the *_params.json
#     of the six E1 folders listed in the script (checkpoints .pt NOT needed);
#   * dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip and pkt_taskB_treats.tsv.zip;
#   * Task B: dataset/PKT/edges.zip (python analysis/pkt_source.py downloads it; compute nodes may have
#     no internet, so do it beforehand from the login node), read once and cached in
#     experiments/block1/cache/ (git-ignored).
#
# Output (tracked in git, see experiments/block1/README.md):
#     experiments/block1/results/X0a_<version>/summary/x0a_<TASK>{.md,_per_seed.csv,_strata.csv,_provenance.json}
#     experiments/block1/results/X0a_<version>/MANIFEST.md, COMPLETE (when every task succeeded)
#     experiments/block1/slurm/x0a-<jobid>.zip   everything above + this job's .out (git-ignored)
# =============================================================================
#SBATCH --job-name=x0a
#SBATCH --output=experiments/block1/slurm/%x-%j.out
#SBATCH --error=experiments/block1/slurm/%x-%j.out
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
# no --gres: this job needs no GPU (on iknos it then runs on the head node, see the cluster guide)

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f experiments/block1/x0a_distance_curve.py ] || { echo "[x0a] submit from the repository root"; exit 2; }

# environment: experiments/block1/slurm/site_env.sh (git-ignored, from site_env.example.sh) or conda
if [ -f experiments/block1/slurm/site_env.sh ]; then
  source experiments/block1/slurm/site_env.sh
else
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gnn
fi
export PKT_SKIP_CONDA=1 PYTHONHASHSEED=0

MODE=${X0A_MODE:-run}            # run | check
VERSION=${X0A_VERSION:-v1}
TASKS=${X0A_TASKS:-"DTI TREATS"}
OUT=experiments/block1/results/X0a_${VERSION}
JOB=${SLURM_JOB_ID:-local}
echo "[x0a] $(date '+%F %T') job $JOB on $(hostname) | mode $MODE | version $VERSION | tasks $TASKS"
echo "[x0a] git $(git rev-parse --short HEAD 2>/dev/null) $(git status --porcelain --untracked-files=no 2>/dev/null | wc -l) modified tracked files"

rc=0
if [ "$MODE" = "check" ]; then
  for t in $TASKS; do python -u experiments/block1/x0a_distance_curve.py --task "$t" --check || rc=$?; done
  [ $rc -eq 0 ] && echo "[x0a] CHECK: ALL OK" || echo "[x0a] CHECK: FAILED (see [FAIL] lines)"
  exit $rc
fi

if [ -f "$OUT/COMPLETE" ]; then
  echo "[x0a] $OUT is COMPLETE: refusing to overwrite. Use X0A_VERSION=v<N+1> for a new version."
  exit 3
fi
mkdir -p "$OUT"
{
  echo "# X0a ${VERSION}"
  echo
  echo "- started: $(date '+%F %T'), Slurm job $JOB on $(hostname)"
  echo "- git commit: $(git rev-parse HEAD 2>/dev/null) ($(git status --porcelain --untracked-files=no 2>/dev/null | wc -l) modified tracked files)"
  echo "- tasks: $TASKS"
  echo "- code: experiments/block1/x0a_distance_curve.py (default parameters; see summary/*_provenance.json)"
} > "$OUT/MANIFEST.md"

for t in $TASKS; do
  rt=0
  python -u experiments/block1/x0a_distance_curve.py --task "$t" --out_dir "$OUT" || rt=$?
  echo "- task $t: exit code ${rt}" >> "$OUT/MANIFEST.md"
  [ $rt -eq 0 ] || rc=$rt
done
echo "- finished: $(date '+%F %T')" >> "$OUT/MANIFEST.md"
[ $rc -eq 0 ] && date '+%F %T' > "$OUT/COMPLETE"
echo "[x0a] $(date '+%F %T') done, exit code $rc"

# one zip to download (checkpoints excluded by the helper); written even if a task failed
python experiments/slurm/zip_results.py "experiments/block1/slurm/x0a-${JOB}.zip" \
  "$OUT" "experiments/block1/slurm/${SLURM_JOB_NAME:-x0a}-${JOB}.out"
exit $rc
