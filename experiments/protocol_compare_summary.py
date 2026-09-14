#!/usr/bin/env python3
"""
Summary table for E0 (experiments/e0_protocol_compare.sh): legacy protocol v1 vs consolidated v2.

Reads experiments/logs/e0/e0_<TASK>_<model>_<variant>_<YYYYmmdd>_<HHMMSS>.log and reports, per
task x model, every variant with mean ± sample std over runs of:
  AUROC, AUPRC, MRR, Hits@1/3/10, M = 0.2 AUROC + 0.4 AUPRC + 0.4 MRR,
  warm (cold-start-excluded) MRR and M, best epoch, training time per run.
Deltas and Welch t-tests (unpaired: v1 and the fixed-split variants do not share test sets) are
given vs `v1` and vs `v1_fixsplit`. With 3 runs per variant p-values are only indicative; the
size and consistency of the deltas matter more.

Baselines (popularity, DistMult without message passing) are listed per task.

Output: experiments/protocol_compare_summary.md and .csv
Usage:  python experiments/protocol_compare_summary.py [--logdir experiments/logs/e0] [--out experiments]
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
                     r"Test MRR:\s*([\d.]+),\s*TEST HITS:\s*(\{[^}]*\})")
TIME_RE = re.compile(r"Run\s+(\d+)\s*\|\s*best_epoch:\s*(\w+)\s*\|\s*train_time_sec:\s*([\d.]+)")
COLD_RE = re.compile(r"Run\s+(\d+)\s*\|\s*cold-start test triples:\s*(\d+)/(\d+)")
RTIES_RE = re.compile(r"Run\s+0\s*\|\s*RANDOM-TIES MRR:\s*([\d.]+)")
HITS_RE = re.compile(r"(\d+)\s*:\s*([\d.]+)")
NAME_RE = re.compile(r"^e0_([A-Za-z0-9,]+)_([a-z]+)_(.+)_(\d{8}_\d{6})\.log$")

LADDER = ["v1", "v1_fixsplit", "s1_selectM", "s2_negatives", "s3_fullgraph", "v2"]
LABEL = {
    "v1": "v1 (legacy)",
    "v1_fixsplit": "+ fixed split",
    "s1_selectM": "+ select on val M",
    "s2_negatives": "+ no oversampling, 5 train negatives",
    "s3_fullgraph": "+ full context graph",
    "v2": "+ disjoint supervision = **v2**",
}
COLS = ["AUROC", "AUPRC", "MRR", "Hits@1", "Hits@3", "Hits@10", "M", "warm_MRR", "warm_M",
        "best_epoch", "train_time_sec"]


def M(a, p, m):
    return 0.2 * a + 0.4 * p + 0.4 * m


def parse(path):
    runs = {}

    def rec(i):
        return runs.setdefault(int(i), {})

    rand_ties = None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            for line in raw.split("\r"):
                if (m := WARM_RE.search(line)):
                    r = rec(m.group(1))
                    a, p, mrr = float(m.group(2)), float(m.group(3)), float(m.group(4))
                    r.update(warm_MRR=mrr, warm_M=M(a, p, mrr))
                elif (m := RUN_RE.search(line)):
                    r = rec(m.group(1))
                    a, p, mrr = float(m.group(2)), float(m.group(3)), float(m.group(4))
                    hits = {int(k): float(v) for k, v in HITS_RE.findall(m.group(5))}
                    r.update(AUROC=a, AUPRC=p, MRR=mrr, M=M(a, p, mrr), **{
                        "Hits@1": hits.get(1, math.nan), "Hits@3": hits.get(3, math.nan),
                        "Hits@10": hits.get(10, math.nan)})
                elif (m := TIME_RE.search(line)):
                    r = rec(m.group(1))
                    r["best_epoch"] = float(m.group(2)) if m.group(2).isdigit() else math.nan
                    r["train_time_sec"] = float(m.group(3))
                elif (m := COLD_RE.search(line)):
                    rec(m.group(1))["cold_frac"] = int(m.group(2)) / max(1, int(m.group(3)))
                elif (m := RTIES_RE.search(line)):
                    rand_ties = float(m.group(1))
    complete = [runs[k] for k in sorted(runs) if "MRR" in runs[k]]
    return complete, rand_ties


def agg(runs, key):
    vals = [r[key] for r in runs if key in r and r[key] == r[key]]
    if not vals:
        return None, None, 0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def welch(a_runs, b_runs, key):
    a = [r[key] for r in a_runs if key in r]
    b = [r[key] for r in b_runs if key in r]
    if not HAVE_SCIPY or len(a) < 2 or len(b) < 2:
        return None
    try:
        return float(_sp.ttest_ind(b, a, equal_var=False).pvalue)
    except Exception:
        return None


def fmt(mean, sd):
    return "—" if mean is None else f"{mean:.3f} ± {sd:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="experiments/logs/e0")
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args()

    data = {}   # (task, model, variant) -> (runs, rand_ties, log)
    for path in sorted(glob.glob(os.path.join(args.logdir, "e0_*.log"))):
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        task, model, variant = m.group(1), m.group(2), m.group(3)
        runs, rt = parse(path)
        key = (task, model, variant)
        if runs and (key not in data or len(runs) >= len(data[key][0])):   # latest most-complete log
            data[key] = (runs, rt, os.path.basename(path))
    if not data:
        print(f"[summary] no parseable e0 logs in {args.logdir}")
        return

    csv_rows, md = [], ["# E0 — protocol comparison: v1 (legacy) vs v2 (consolidated)\n",
                        "Mean ± sample std over runs (test set). M = 0.2·AUROC + 0.4·AUPRC + 0.4·MRR. "
                        "*warm* = excluding cold-start test triples. Δ and Welch p on M are vs `v1_fixsplit` "
                        "(same fixed test set as all later variants); `v1` changes split at every run, so "
                        "its test sets differ. Configs are those tuned under v1 for all variants "
                        "(conservative for v2). With ~3 runs, p-values are indicative only.\n"]
    tasks = sorted({k[0] for k in data})
    for task in tasks:
        models = sorted({k[1] for k in data if k[0] == task and k[1] not in ("popularity", "distmult")})
        for model in models:
            md.append(f"\n## {task} — {model}\n")
            md.append("| variant | n | AUROC | AUPRC | MRR | Hits@10 | **M** | warm MRR | ΔM vs fixsplit | p(M) | best ep. | time/run (s) |")
            md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            ref = data.get((task, model, "v1_fixsplit"), (None,))[0]
            for variant in LADDER:
                if (task, model, variant) not in data:
                    continue
                runs = data[(task, model, variant)][0]
                s = {c: agg(runs, c) for c in COLS}
                dM = p = None
                if ref and variant not in ("v1_fixsplit",):
                    dM = s["M"][0] - agg(ref, "M")[0]
                    p = welch(ref, runs, "M")
                md.append("| {} | {} | {} | {} | {} | {} | **{}** | {} | {} | {} | {} | {} |".format(
                    LABEL.get(variant, variant), s["MRR"][2], fmt(*s["AUROC"][:2]), fmt(*s["AUPRC"][:2]),
                    fmt(*s["MRR"][:2]), fmt(*s["Hits@10"][:2]), fmt(*s["M"][:2]), fmt(*s["warm_MRR"][:2]),
                    "—" if dM is None else f"{dM:+.3f}", "—" if p is None else f"{p:.3f}",
                    "—" if s["best_epoch"][0] is None else f"{s['best_epoch'][0]:.0f}",
                    "—" if s["train_time_sec"][0] is None else f"{s['train_time_sec'][0]:.0f}"))
                for c in COLS:
                    mean, sd, n = s[c]
                    csv_rows.append([task, model, variant, c, n, "" if mean is None else round(mean, 5),
                                     "" if sd is None else round(sd, 5),
                                     "" if (c != "M" or dM is None) else round(dM, 5),
                                     "" if (c != "M" or p is None) else round(p, 5),
                                     data[(task, model, variant)][2]])

        base = [(k, v) for k, v in data.items() if k[0] == task and k[1] in ("popularity", "distmult")]
        if base:
            md.append(f"\n## {task} — baselines (fixed split)\n")
            md.append("| baseline | n | AUROC | AUPRC | MRR | Hits@10 | **M** | warm MRR | note |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for (_, model, variant), (runs, rt, logname) in sorted(base):
                s = {c: agg(runs, c) for c in COLS}
                note = (f"MRR with random tie-breaking: {rt:.3f}" if rt is not None else
                        "embeddings only, no message passing (v2 protocol)")
                name = "popularity (node degree)" if model == "popularity" else f"DistMult (no GNN) `{variant}`"
                md.append("| {} | {} | {} | {} | {} | {} | **{}** | {} | {} |".format(
                    name, s["MRR"][2], fmt(*s["AUROC"][:2]), fmt(*s["AUPRC"][:2]), fmt(*s["MRR"][:2]),
                    fmt(*s["Hits@10"][:2]), fmt(*s["M"][:2]), fmt(*s["warm_MRR"][:2]), note))
                for c in COLS:
                    mean, sd, n = s[c]
                    csv_rows.append([task, model, variant, c, n, "" if mean is None else round(mean, 5),
                                     "" if sd is None else round(sd, 5), "", "", logname])

    md.append("\n### How to read it\n")
    md.append("- `v1 → + fixed split`: evaluation change only (same training). A large gap means the v1 "
              "mean±sd was dominated by split variance.")
    md.append("- `+ select on val M`: effect of choosing the checkpoint with the HPO criterion instead of "
              "the validation loss (look at *best ep.*).")
    md.append("- `+ no oversampling`: for R-GCN it should be ~neutral (the loss is a mean over positives and "
              "R-GCN averages per relation); for CompGCN it also removes the duplicated target edges from "
              "the graph.")
    md.append("- `+ full context graph`: effect of keeping the edges (and 1-to-1 bridges) that random "
              "undersampling deleted.")
    md.append("- `+ disjoint supervision`: removes the train/test mismatch of scoring edges that are in the graph.")
    md.append("- A GNN row must beat both baselines; otherwise message passing is not adding information.")
    if not HAVE_SCIPY:
        md.append("\n_scipy not installed: p-values omitted._")

    os.makedirs(args.out, exist_ok=True)
    md_path = os.path.join(args.out, "protocol_compare_summary.md")
    csv_path = os.path.join(args.out, "protocol_compare_summary.csv")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "model", "variant", "metric", "n", "mean", "std", "delta_M_vs_fixsplit",
                    "p_welch_M_vs_fixsplit", "source_log"])
        w.writerows(csv_rows)
    print("\n".join(md))
    print(f"\n[summary] -> {md_path} , {csv_path}")


if __name__ == "__main__":
    main()
