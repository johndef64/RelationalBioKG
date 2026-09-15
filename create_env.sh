#!/usr/bin/env bash
# =============================================================================
# create_env.sh — build the conda environment for RelationalPKT, ready to run.
#
# Reproduces the environment on which the code was verified:
#   Python 3.10 · PyTorch 2.7.0 · PyG 2.7.0 (+ torch_scatter / torch_sparse / pyg_lib ...)
#   + the pinned Python dependencies in requirements.txt
#
# The CUDA build is picked automatically from the NVIDIA driver (nvidia-smi):
#   driver CUDA >= 12.8 -> cu128 · >= 12.6 -> cu126 · >= 11.8 -> cu118 · no GPU -> cpu
#
# Usage (from the repo root):
#   bash create_env.sh                    # create/update env "gnn"
#   CUDA_TAG=cu126 bash create_env.sh     # force a CUDA build (cu128 | cu126 | cu118 | cpu)
#   ENV_NAME=gnn2 bash create_env.sh      # another env name
#   RECREATE=1 bash create_env.sh         # delete and rebuild the env from scratch
#
# Safe to re-run: an existing env is reused and packages are (re)installed at the pinned versions.
# Afterwards: `conda activate gnn` and `wandb login` (needed only for the HPO).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="${ENV_NAME:-gnn}"
PYTHON_VERSION="3.10"
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
TORCHAUDIO_VERSION="2.7.0"
PYG_VERSION="2.7.0"
RECREATE="${RECREATE:-0}"

log () { echo -e "\n[create_env] $*"; }

# ---- conda ------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "[create_env] ERROR: conda not found on PATH (install Miniconda/Anaconda first)." >&2
  exit 1
fi
# conda's shell hooks reference unset variables: relax -u only around conda itself
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u

# ---- CUDA build -------------------------------------------------------------
detect_cuda_tag () {
  if ! command -v nvidia-smi >/dev/null 2>&1; then echo "cpu"; return; fi
  local v
  v="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1 || true)"
  if [ -z "$v" ]; then echo "cpu"; return; fi
  ge () { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }   # $1 >= $2
  if ge "$v" 12.8; then echo "cu128"
  elif ge "$v" 12.6; then echo "cu126"
  elif ge "$v" 11.8; then echo "cu118"
  else echo "cpu"; fi
}
CUDA_TAG="${CUDA_TAG:-$(detect_cuda_tag)}"
case "$CUDA_TAG" in cu128|cu126|cu118|cpu) ;; *)
  echo "[create_env] ERROR: CUDA_TAG must be cu128, cu126, cu118 or cpu (got '$CUDA_TAG')." >&2; exit 1 ;;
esac
log "env=$ENV_NAME python=$PYTHON_VERSION torch=$TORCH_VERSION build=$CUDA_TAG"
[ "$CUDA_TAG" = "cpu" ] && echo "[create_env] WARNING: CPU build — full-graph training needs a GPU."

# ---- environment ------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [ "$RECREATE" = "1" ]; then
    log "removing existing env '$ENV_NAME' (RECREATE=1)"
    set +u; conda deactivate 2>/dev/null || true; set -u
    conda env remove -n "$ENV_NAME" -y
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" pip -y
  else
    log "env '$ENV_NAME' already exists — updating it (RECREATE=1 to rebuild from scratch)"
  fi
else
  log "creating env '$ENV_NAME'"
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" pip -y
fi
set +u; conda activate "$ENV_NAME"; set -u
PY="python"
$PY -m pip install --upgrade pip

# ---- PyTorch ----------------------------------------------------------------
log "installing PyTorch $TORCH_VERSION ($CUDA_TAG)"
$PY -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

# ---- PyG: compiled extensions + torch-geometric (wheels must match torch + CUDA) ----
PYG_INDEX="https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_TAG.html"
log "installing PyG extensions for torch-$TORCH_VERSION+$CUDA_TAG"
$PY -m pip install --no-cache-dir --only-binary=:all: \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f "$PYG_INDEX"
log "installing torch-geometric $PYG_VERSION"
$PY -m pip install "torch-geometric==$PYG_VERSION" -f "$PYG_INDEX"

# ---- everything else (pinned) -----------------------------------------------
log "installing requirements.txt"
$PY -m pip install -r requirements.txt

# ---- verification -----------------------------------------------------------
log "verifying the environment"
$PY - <<'EOF'
import importlib, sys
mods = ["torch", "torch_scatter", "torch_sparse", "torch_geometric", "torcheval", "numpy", "pandas",
        "scipy", "sklearn", "ijson", "wandb", "tqdm", "termcolor", "requests"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  ok  {m:16s} {getattr(mod, '__version__', '')}")
    except Exception as e:
        bad.append(m); print(f"  ERR {m:16s} {e}")
import torch
print(f"  CUDA available: {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
if torch.cuda.is_available():
    # exercise the compiled extensions on the GPU once
    import torch_scatter, torch_sparse
    x = torch.ones(4, 2, device="cuda"); idx = torch.tensor([0, 0, 1, 1], device="cuda")
    assert torch_scatter.scatter_add(x, idx, dim=0).shape == (2, 2)
    i = torch.tensor([[0, 1], [1, 0]], device="cuda"); v = torch.ones(2, device="cuda")
    assert torch_sparse.spmm(i, v, 2, 2, torch.ones(2, 3, device="cuda")).shape == (2, 3)
    print("  torch_scatter / torch_sparse GPU kernels: ok")
# the project's own modules import cleanly
sys.path.insert(0, ".")
import train_and_eval, tuning_hyperparameter  # noqa: F401
print("  project modules (train_and_eval, tuning_hyperparameter): ok")
sys.exit(1 if bad else 0)
EOF

# ---- data check (not downloaded by this script) -----------------------------
for f in dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip; do
  [ -f "$f" ] && echo "  data ok  $f" || echo "  data MISSING  $f  (copy dataset/PKT_subgraphs/ or run: python analysis/06_build_subgraphs.py)"
done

log "done. Next:  conda activate $ENV_NAME  &&  wandb login   (then see TODO_SERVER.md)"
