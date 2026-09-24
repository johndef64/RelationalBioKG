#!/bin/bash
# =============================================================================
# One job that checks the new ablation system before the real runs. Submit from the repository root:
#
#     sbatch experiments/slurm/check_new_system.sbatch
#     tail -f experiments/slurm/check-<jobid>.out          # while it runs
#
# About 40 minutes on one GPU. It writes only inside experiments/slurm/test_<jobid>/, never into
# experiments/ablation/ or models/. The last lines of the .out are the verdict, one PASS/FAIL per test:
#
#   1  fixed-order GPU sum is bit-identical, atomic sum is not        (src/deterministic_ops.py)
#   2  two short trainings with --deterministic are identical          (train_and_eval.py)
#   3  a miniature ablation (13 variants x 2 seeds x 20 epochs) produces the versioned folder:
#      MANIFEST with the GPU, settings.env, PREREGISTRATION.json, 13 logs, 13 model folders,
#      summary, COMPLETE, driver.log; and models/ is left untouched
#   4  comp_full and ctx_full, the same recipe, give identical test metrics seed by seed
#   5  resume skips completed variants; a resume with different settings is refused
#
# Run it on the GPU type the real job will use (edit --gres, or pass it on the command line).
# =============================================================================
#SBATCH --job-name=check
#SBATCH --output=experiments/slurm/%x-%j.out
#SBATCH --error=experiments/slurm/%x-%j.out
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source /home/apps/iknos-env.sh
iknos_activate gnn
export PKT_SKIP_CONDA=1 PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

T="experiments/slurm/test_${SLURM_JOB_ID:-local}"
mkdir -p "$T"
declare -A R
verdict () { R[$1]="$2"; echo "[check] test $1: $2"; }
cnt () { local c; c="$(grep -c -- "$1" "$2" 2>/dev/null)"; echo "${c:-0}"; }   # 0 also if the file is missing

echo "[check] $(date '+%F %T') on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
MODELS_BEFORE="$(ls models/ 2>/dev/null)"

# ---------------------------------------------------------------- test 1
python - > "$T/t1.log" 2>&1 <<'EOF'
import torch
torch.manual_seed(0)
n, E, d = 5000, 400000, 64
idx = torch.randint(0, n, (E,), device='cuda'); src = torch.randn(E, d, device='cuda')
perm = torch.argsort(idx, stable=True); L = torch.bincount(idx, minlength=n)
seg = [torch.segment_reduce(src[perm], 'sum', lengths=L, axis=0) for _ in range(5)]
atm = [torch.zeros(n, d, device='cuda').index_add_(0, idx, src) for _ in range(5)]
same_seg = all(torch.equal(seg[0], s) for s in seg[1:])
same_atm = all(torch.equal(atm[0], a) for a in atm[1:])
diff = (seg[0] - atm[0]).abs().max().item()
print('GPU', torch.cuda.get_device_name(0))
print('segment_reduce identical:', same_seg, '| index_add identical:', same_atm, '| max diff:', diff)
print('RESULT', 'PASS' if same_seg and diff < 1e-3 else 'FAIL')
EOF
cat "$T/t1.log"
verdict 1 "$(grep -o 'RESULT .*' "$T/t1.log" | cut -d' ' -f2)"

# ---------------------------------------------------------------- test 2
A="--tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task DTI --model rgcn --config PKT-DTI-best-v2b \
 --runs 1 --epochs 30 --dry_run --early_stopping --patience 50 --negative_sampling filtered --eval_filtered \
 --oversample_rate 1 --undersample_rate 1.0 --alpha 0.25 --gamma 3.0 --alpha_adv 2.0 --split_seed 42 \
 --select_metric mixed --train_negative_rate 10 --disjoint_supervision 0.3"
python train_and_eval.py $A --deterministic > "$T/det1.log" 2>&1
python train_and_eval.py $A --deterministic > "$T/det2.log" 2>&1
python train_and_eval.py $A                 > "$T/nondet.log" 2>&1
v () { grep "\[val\]" "$1" | sed 's/\[val\] [0-9-]* [0-9:]* //'; }
n2="$(v "$T/det1.log" | wc -l)"
if [ "$n2" -gt 0 ] && diff <(v "$T/det1.log") <(v "$T/det2.log") > /dev/null; then
  verdict 2 "PASS ($n2 evaluations identical)"
else
  verdict 2 "FAIL (evaluations: $n2; see $T/det1.log and det2.log)"
fi
diff <(v "$T/det1.log") <(v "$T/nondet.log") > /dev/null \
  && echo "[check] note: without the flag the run happened to be identical too" \
  || echo "[check] note: without the flag the run differs, as expected"

# ---------------------------------------------------------------- test 3
export ABL_ROOT="$T/ablation" ABL_RUNS=2 ABL_EPOCHS=20 PROTOCOL=v2 ABL_TASK=A ABL_VERSION=v1
bash experiments/e3_ablation.sh all > "$T/t3_console.log" 2>&1
V="$ABL_ROOT/DTI_v1"
miss=""
for f in COMPLETE MANIFEST.md settings.env PREREGISTRATION.json driver.log summary/ablation_summary.md; do
  [ -s "$V/$f" ] || miss="$miss $f"
done
nlog="$(ls "$V"/logs/*.log 2>/dev/null | wc -l)"
nmod="$(ls -d "$V"/models/*/ 2>/dev/null | wc -l)"
gpu_in_manifest="$(cnt 'RTX\|NVIDIA\|Tesla\|A100\|H100' "$V/MANIFEST.md")"
[ "$MODELS_BEFORE" = "$(ls models/ 2>/dev/null)" ] || miss="$miss models/-changed"
[ "$nlog" -eq 13 ] || miss="$miss logs=$nlog"
[ "$nmod" -eq 13 ] || miss="$miss models=$nmod"
[ "$gpu_in_manifest" -gt 0 ] || miss="$miss no-GPU-in-MANIFEST"
[ -z "$miss" ] && verdict 3 "PASS (13 logs, 13 model folders, all files present)" || verdict 3 "FAIL (missing/wrong:$miss)"

# ---------------------------------------------------------------- test 4
m () { grep -o "Test MRR: [0-9.]*" "$V"/logs/e3_"$1"_*.log | tr '\n' ' '; }
echo "[check] comp_full: $(m comp_full)"
echo "[check] ctx_full : $(m ctx_full)"
if [ -n "$(m comp_full)" ] && [ "$(m comp_full)" = "$(m ctx_full)" ]; then
  verdict 4 "PASS (identical seed by seed)"
else
  verdict 4 "FAIL (the same recipe gave different numbers: something is still not deterministic)"
fi

# ---------------------------------------------------------------- test 5
bash experiments/e3_ablation.sh component > "$T/t5_resume.log" 2>&1
skips="$(cnt 'skip' "$T/t5_resume.log")"
ABL_RUNS=3 bash experiments/e3_ablation.sh component > "$T/t5_refuse.log" 2>&1; rc=$?
refused="$(cnt 'different settings' "$T/t5_refuse.log")"
# under Slurm the driver writes to driver.log: look there as well
skips=$(( skips + $(cnt "skip 'comp_" "$V/driver.log") ))
refused=$(( refused + $(cnt 'different settings' "$V/driver.log") ))
if [ "$skips" -ge 6 ] && [ "$refused" -ge 1 ] && [ "$rc" -ne 0 ]; then
  verdict 5 "PASS (6 variants skipped on resume; different settings refused)"
else
  verdict 5 "FAIL (skipped=$skips refused=$refused exit=$rc; see $T/t5_*.log and $V/driver.log)"
fi

echo
echo "================ VERDICT ($(date '+%F %T'), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"
for k in 1 2 3 4 5; do echo "  test $k: ${R[$k]:-not run}"; done
echo "  details in $T/"
