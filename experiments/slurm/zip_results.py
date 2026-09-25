"""Pack the outputs of a finished job into one .zip to download: logs, summaries, results.

    python experiments/slurm/zip_results.py <out.zip> <file or folder> [<file or folder> ...]

Model checkpoints (*.pt) are left out: they are large and stay on the server. Missing paths are
reported and skipped, so a job that failed half-way still gets its zip. Paths inside the zip are
relative to the repository root, so `unzip` / Expand-Archive from the root puts every file back in its
place.
"""
import os
import sys
import zipfile

SKIP_EXT = (".pt",)

if len(sys.argv) < 3:
    sys.exit(__doc__)
out, paths = sys.argv[1], sys.argv[2:]
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        if not os.path.exists(p):
            print(f"[zip] missing, skipped: {p}")
            continue
        files = [p] if os.path.isfile(p) else \
            [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs]
        for f in files:
            if f.endswith(SKIP_EXT) or os.path.abspath(f) == os.path.abspath(out):
                continue
            z.write(f, os.path.relpath(f))
            n += 1
print(f"[zip] {out}: {n} files, {os.path.getsize(out) / 1e6:.1f} MB (checkpoints *.pt excluded)")
