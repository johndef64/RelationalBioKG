"""
pkt_source.py
-------------
The source graph of every build script: the PheKnowLator (PKT) ontology backbone of KG-TransomicNet,
`dataset/PKT/nodes.zip` and `dataset/PKT/edges.zip`.

`ensure_pkt_files()` is called by the analysis/ scripts before they read those files. If a file is
missing it is downloaded from the Hugging Face dataset `johndef64/KG-TransomicNet` (folder `PKT/`,
public, no login) and its SHA-256 is checked against the release every result of this repository
was computed from. A file that is already there is left untouched and not re-hashed, so nothing is
downloaded twice.

    python analysis/pkt_source.py            # download now (and verify the files already present)
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

PKT_DIR = Path(__file__).resolve().parents[1] / "dataset" / "PKT"
HF_BASE = "https://huggingface.co/datasets/johndef64/KG-TransomicNet/resolve/main/PKT/"
FILES = {  # name -> (size in bytes, SHA-256), the release used for all results (checked 2026-09-28)
    "nodes.zip": (64_673_549, "129f4f1110d56e7695fce0d8e160f1b9c54c5664be8fff5d07f8204a4ccfad7b"),
    "edges.zip": (224_508_605, "24642dec8b220eb9264113c29fc4784a0fea75e70d5bc5c594e1d5ce60fae2cc"),
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(name, dest):
    size, digest = FILES[name]
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"[pkt] {name} not found: downloading {size / 1e6:.0f} MB from {HF_BASE}{name}", flush=True)
    with urllib.request.urlopen(HF_BASE + name) as r, open(part, "wb") as out:
        done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            print(f"\r[pkt]   {done / 1e6:6.0f} / {size / 1e6:.0f} MB", end="", flush=True)
    print()
    got = _sha256(part)
    if got != digest:
        part.unlink()
        raise SystemExit(f"[pkt] {name}: checksum mismatch (got {got[:12]}..., expected {digest[:12]}...). "
                         "The published file differs from the release this code was run on.")
    part.replace(dest)
    print(f"[pkt] {name}: downloaded and verified", flush=True)


def ensure_pkt_files(names=tuple(FILES), pkt_dir=PKT_DIR, verify_existing=False):
    """Make sure the PKT source files exist in `pkt_dir`, downloading the missing ones."""
    pkt_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        dest = pkt_dir / name
        if not dest.exists():
            _download(name, dest)
        elif verify_existing and _sha256(dest) != FILES[name][1]:
            print(f"[pkt] WARNING: {dest} is not the release the results were computed from "
                  "(different SHA-256). Delete it to download the published one.", flush=True)
    return pkt_dir


if __name__ == "__main__":
    ensure_pkt_files(verify_existing=True)
    print(f"[pkt] ready: {', '.join(str(PKT_DIR / n) for n in FILES)}")
    sys.exit(0)
