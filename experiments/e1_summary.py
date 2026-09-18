#!/usr/bin/env python3
"""
E1 summary — thesis table: relational GNN encoders (R-GCN, CompGCN) vs the embedding-only
DistMult baseline, per task, in the format of the PathogenKG model-comparison table.

Reads experiments/logs/v2/e1_<TASK>_<model>_<YYYYmmdd>_<HHMMSS>.log (the latest most-complete log
per task x model) and reports, per metric (rows) and model (columns): mean ± sample std over runs,
best model per row in bold, and for every GNN a Welch t-test against DistMult (two-sided).
Also: best single run (by test M), cold-start-excluded (warm) MRR, mean best epoch.

Output: <out>/e1_summary.md and e1_summary.csv
Usage:  python experiments/e1_summary.py                       # logs in experiments/logs/v2
        python experiments/e1_summary.py --logdir experiments/logs --out experiments/logs   # v1 logs
"""
import argparse
import csv
import glob
import math
import os
import re
import statistics as st

try:
    from scipy import stats as _sp
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

RUN_RE = re.compile(r"Run\s+(\d+)\s*\|\s*Test Auroc:\s*([\d.]+),\s*Test Auprc:\s*([\d.]+),\s*"
                    r"Test MRR:\s*([\d.]+),\s*TEST HITS:\s*(\{[^}]*\})")
WARM_RE = re.compile(r"Run\s+(\d+)\s*\|\s*WARM Test Auroc:\s*([\d.]+),\s*Test Auprc:\s*([\d.]+),\s*"
                     r"Test MRR:\s*([\d.]+)")
# --dedup_eval: test metrics excluding triples whose fact is already in training through a
# near-duplicate ChEBI node. DEDUP = charge/hydration/salt variants (the defensible grouping);
# DEDUP_STEREO additionally treats enantiomers as one agent (sensitivity bound). Absent from
# logs produced without the flag, in which case the columns simply stay empty.
DEDUP_RE = re.compile(r"Run\s+(\d+)\s*\|\s*DEDUP Test Auroc:\s*([\d.]+),\s*Test Auprc:\s*([\d.]+),\s*"
                      r"Test MRR:\s*([\d.]+)")
DEDUP_STEREO_RE = re.compile(r"Run\s+(\d+)\s*\|\s*DEDUP_STEREO Test Auroc:\s*([\d.]+),\s*"
                             r"Test Auprc:\s*([\d.]+),\s*Test MRR:\s*([\d.]+)")
TIME_RE = re.compile(r"Run\s+(\d+)\s*\|\s*best_epoch:\s*(\w+)\s*\|\s*train_time_sec:\s*([\d.]+)")
HITS_RE = re.compile(r"(\d+)\s*:\s*([\d.]+)")
NAME_RE = re.compile(r"^e1_([A-Za-z0-9,]+)_([a-z]+)_(\d{8}_\d{6})\.log$")

METRICS = ["AUROC", "AUPRC", "MRR", "Hits@1", "Hits@3", "Hits@10", "M", "warm_MRR",
           "dedup_MRR", "dedup_stereo_MRR"]
MODEL_ORDER = ["distmult", "rgcn", "compgcn"]
MODEL_LABEL = {"distmult": "DistMult (no GNN)", "rgcn": "R-GCN", "compgcn": "CompGCN"}


def parse(path):
    runs = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            for line in raw.split("\r"):
                if (m := DEDUP_STEREO_RE.search(line)):
                    runs.setdefault(int(m.group(1)), {})["dedup_stereo_MRR"] = float(m.group(4))
                elif (m := DEDUP_RE.search(line)):
                    runs.setdefault(int(m.group(1)), {})["dedup_MRR"] = float(m.group(4))
                elif (m := WARM_RE.search(line)):
                    runs.setdefault(int(m.group(1)), {})["warm_MRR"] = float(m.group(4))
                elif (m := RUN_RE.search(line)):
                    a, p, mrr = float(m.group(2)), float(m.group(3)), float(m.group(4))
                    hits = {int(k): float(v) for k, v in HITS_RE.findall(m.group(5))}
                    runs.setdefault(int(m.group(1)), {}).update({
                        "AUROC": a, "AUPRC": p, "MRR": mrr, "M": 0.2 * a + 0.4 * p + 0.4 * mrr,
                        "Hits@1": hits.get(1, math.nan), "Hits@3": hits.get(3, math.nan),
                        "Hits@10": hits.get(10, math.nan)})
                elif (m := TIME_RE.search(line)):
                    r = runs.setdefault(int(m.group(1)), {})
                    r["best_epoch"] = float(m.group(2)) if m.group(2).isdigit() else math.nan
    return [runs[k] for k in sorted(runs) if "MRR" in runs[k]]


def vals(runs, key):
    return [r[key] for r in runs if key in r and r[key] == r[key]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="experiments/logs/v2")
    ap.add_argument("--out", default=None, help="output folder (default: --logdir)")
    args = ap.parse_args()
    out = args.out or args.logdir

    data = {}
    for path in sorted(glob.glob(os.path.join(args.logdir, "e1_*.log"))):
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        runs = parse(path)
        key = (m.group(1), m.group(2))
        if runs and (key not in data or len(runs) >= len(data[key][0])):
            data[key] = (runs, os.path.basename(path))
    if not data:
        print(f"[e1_summary] no parseable e1_*.log in {args.logdir}")
        return

    md = ["# E1 — relational GNN encoders vs embedding-only DistMult baseline\n",
          "Test set, mean ± sample std over runs (same split, same protocol for all models). "
          "M = 0.2·AUROC + 0.4·AUPRC + 0.4·MRR. Best mean per row in **bold**. "
          "p = Welch two-sided t-test of each GNN vs DistMult. *warm MRR* excludes cold-start test "
          "triples (an endpoint with no edge in the training graph).\n"]
    rows_csv = []
    for task in sorted({k[0] for k in data}):
        models = [mm for mm in MODEL_ORDER if (task, mm) in data] + \
                 sorted(mm for (t, mm) in data if t == task and mm not in MODEL_ORDER)
        gnns = [mm for mm in models if mm != "distmult"]
        base = data.get((task, "distmult"), (None,))[0]

        header = "| Metric | " + " | ".join(f"{MODEL_LABEL.get(mm, mm)} (n={len(data[(task, mm)][0])})" for mm in models)
        if base:
            header += " | " + " | ".join(f"p {MODEL_LABEL.get(g, g)} vs DistMult" for g in gnns)
        md += [f"\n## Task {task}\n", header + " |",
               "|" + "---|" * (1 + len(models) + (len(gnns) if base else 0))]
        for metric in METRICS:
            means = {mm: (st.mean(vals(data[(task, mm)][0], metric)) if vals(data[(task, mm)][0], metric) else None)
                     for mm in models}
            valid = [v for v in means.values() if v is not None]
            best = max(valid) if valid else None
            cells = []
            for mm in models:
                v = vals(data[(task, mm)][0], metric)
                if not v:
                    cells.append("—"); continue
                sd = st.stdev(v) if len(v) > 1 else 0.0
                txt = f"{st.mean(v):.3f} ± {sd:.3f}"
                cells.append(f"**{txt}**" if best is not None and abs(st.mean(v) - best) < 1e-12 else txt)
                rows_csv.append([task, mm, metric, len(v), round(st.mean(v), 5), round(sd, 5)])
            pcells = []
            if base:
                for g in gnns:
                    a, b = vals(base, metric), vals(data[(task, g)][0], metric)
                    p = None
                    if HAVE_SCIPY and len(a) > 1 and len(b) > 1:
                        try:
                            p = float(_sp.ttest_ind(b, a, equal_var=False).pvalue)
                        except Exception:
                            p = None
                    pcells.append("—" if p is None or p != p else (f"{p:.2g}" if p >= 1e-3 else "<0.001"))
            md.append(f"| {metric} | " + " | ".join(cells) + (" | " + " | ".join(pcells) if pcells else "") + " |")

        md.append("")
        for mm in models:
            runs, logname = data[(task, mm)]
            best_run = max(range(len(runs)), key=lambda i: runs[i]["M"])
            r = runs[best_run]
            be = vals(runs, "best_epoch")
            md.append(f"- {MODEL_LABEL.get(mm, mm)}: best run {best_run} → AUROC {r['AUROC']:.3f}, "
                      f"AUPRC {r['AUPRC']:.3f}, MRR {r['MRR']:.3f}, Hits@10 {r['Hits@10']:.3f}, M {r['M']:.3f}"
                      + (f"; mean best epoch {st.mean(be):.0f}" if be else "") + f"  (`{logname}`)")

    if not HAVE_SCIPY:
        md.append("\n_scipy not installed: p-values omitted._")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "e1_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(out, "e1_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "model", "metric", "n", "mean", "std"])
        w.writerows(rows_csv)
    print("\n".join(md))
    print(f"\n[e1_summary] -> {os.path.join(out, 'e1_summary.md')}")


if __name__ == "__main__":
    main()
