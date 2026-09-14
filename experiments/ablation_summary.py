#!/usr/bin/env python3
"""
Aggregate E3 ablation results into a PUBLICATION-GRADE comparison table.

The ablation runs each tag as a separate `train_and_eval.py` invocation logging to
experiments/logs/e3_<tag>_<timestamp>.log. There is no cross-tag summary out of the
box; this script builds it, with the statistics an ablation table needs for a paper.

For every metric (AUROC, AUPRC, MRR, Hits@1/3/10) and every tag it reports:
  - mean, SAMPLE std (n-1), and SEM (std/sqrt(n)) across runs;
  - Delta vs the full-reference of the same family (comp_full / ctx_full);
  - a PAIRED significance test vs that reference. Runs are paired by index, which
    equals seed order in train_and_eval.py (seed = base + run_i), so run i of a
    variant and run i of the reference share the seed -> legitimate paired samples.
  - both a paired t-test and a Wilcoxon signed-rank test (non-parametric, safer for
    n~12 and bounded metrics), each with raw and HOLM-adjusted p-values (Holm applied
    across the variants within a family x metric group, correcting multiple comparisons).

Significance markers in the markdown table use the Holm-adjusted paired t-test p-value
vs the family reference: ** p<0.01, * p<0.05 (two-sided). Full numbers (both tests,
raw + Holm) are in the CSV.

Read-only on the logs -> safe to run while the ablation is still going (partial tags
show fewer runs; tests need n>=2 paired points, else p is left blank).

Requires scipy for the tests. Without it, means/std/SEM/Delta are still produced and a
note is printed (p-values blank).

Output:
  experiments/ablation_summary.csv   (long format: one row per tag x metric, all stats)
  experiments/ablation_summary.md    (per-family tables, mean+/-std with significance)

Usage:
  python experiments/ablation_summary.py
  python experiments/ablation_summary.py --logdir experiments/logs --out experiments
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

# "Run 0 | Test Auroc: 0.899, Test Auprc: 0.921, Test MRR: 0.385, TEST HITS: {1: 0.006, 3: 0.015, 10: 0.05}"
RUN_RE = re.compile(
    r"Run\s+(\d+)\s*\|\s*Test Auroc:\s*([\d.]+),\s*Test Auprc:\s*([\d.]+),\s*"
    r"Test MRR:\s*([\d.]+),\s*TEST HITS:\s*(\{[^}]*\})"
)
NAME_RE = re.compile(r"^e3_(.+)_\d{8}_\d{6}\.log$")   # tag may contain underscores
HITS_RE = re.compile(r"(\d+)\s*:\s*([\d.]+)")

METRICS = ["AUROC", "AUPRC", "MRR", "Hits@1", "Hits@3", "Hits@10"]


def parse_log(path):
    """Return list of per-run metric dicts, in run (=seed) order."""
    runs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RUN_RE.search(line)
            if not m:
                continue
            hits = {int(k): float(v) for k, v in HITS_RE.findall(m.group(5))}
            runs.append({
                "_run": int(m.group(1)),
                "AUROC": float(m.group(2)),
                "AUPRC": float(m.group(3)),
                "MRR": float(m.group(4)),
                "Hits@1": hits.get(1, float("nan")),
                "Hits@3": hits.get(3, float("nan")),
                "Hits@10": hits.get(10, float("nan")),
            })
    runs.sort(key=lambda r: r["_run"])   # ensure paired-by-seed alignment
    return runs


def family_of(tag):
    if tag.startswith("comp_"):
        return "comp"
    if tag.startswith("ctx_"):
        return "ctx"
    return "other"


def paired(ref_runs, var_runs, metric):
    """Aligned, NaN-free paired value arrays for one metric (paired by run index)."""
    ref = {r["_run"]: r[metric] for r in ref_runs}
    var = {r["_run"]: r[metric] for r in var_runs}
    a, b = [], []
    for k in sorted(set(ref) & set(var)):
        x, y = ref[k], var[k]
        if x == x and y == y:  # drop NaN
            a.append(x); b.append(y)
    return a, b


def holm(pairs):
    """Holm-Bonferroni over [(key, p), ...] with possibly-None p. Returns {key: adj_p}."""
    valid = [(k, p) for k, p in pairs if p is not None and p == p]
    m = len(valid)
    out = {k: None for k, _ in pairs}
    prev = 0.0
    for rank, (k, p) in enumerate(sorted(valid, key=lambda kp: kp[1])):
        adj = min(1.0, (m - rank) * p)
        adj = max(adj, prev)
        prev = adj
        out[k] = adj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="experiments/logs")
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args()

    logs = sorted(glob.glob(os.path.join(args.logdir, "e3_*.log")))
    by_tag = {}  # tag -> (runs, source_log); keep the log with MORE runs on duplicates
    for path in logs:
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        tag, runs = m.group(1), parse_log(path)
        if runs and (tag not in by_tag or len(runs) > len(by_tag[tag][0])):
            by_tag[tag] = (runs, os.path.basename(path))

    if not by_tag:
        print(f"[summary] no parseable e3_*.log with test metrics in {args.logdir}")
        return
    if not HAVE_SCIPY:
        print("[summary] WARNING: scipy not found -> p-values will be blank. `pip install scipy` for tests.")

    # ---- descriptive stats per tag/metric ----
    stats = {}  # tag -> metric -> dict
    for tag, (runs, _src) in by_tag.items():
        stats[tag] = {}
        for k in METRICS:
            vals = [r[k] for r in runs if r[k] == r[k]]
            n = len(vals)
            mean = st.mean(vals) if vals else float("nan")
            sd = st.stdev(vals) if n > 1 else 0.0            # SAMPLE std (n-1)
            sem = sd / math.sqrt(n) if n > 1 else 0.0
            stats[tag][k] = {"n": n, "mean": mean, "std": sd, "sem": sem,
                             "delta": None, "p_t": None, "p_t_holm": None,
                             "p_w": None, "p_w_holm": None}

    # ---- paired tests vs family reference (comp_full / ctx_full) ----
    families = {}
    for tag in by_tag:
        families.setdefault(family_of(tag), []).append(tag)

    ref_missing = []
    for fam, tags in families.items():
        ref_tag = f"{fam}_full"
        if ref_tag not in by_tag:
            if fam != "other":
                ref_missing.append(fam)
            continue
        ref_runs = by_tag[ref_tag][0]
        variants = [t for t in tags if t != ref_tag]
        for k in METRICS:
            p_t_pairs, p_w_pairs = [], []
            for t in variants:
                a, b = paired(ref_runs, by_tag[t][0], k)
                stats[t][k]["delta"] = (st.mean(b) - st.mean(a)) if a else None  # variant - full
                pt = pw = None
                if HAVE_SCIPY and len(a) >= 2 and any(x != y for x, y in zip(a, b)):
                    try:
                        pt = float(_sp.ttest_rel(b, a).pvalue)
                    except Exception:
                        pt = None
                    try:
                        pw = float(_sp.wilcoxon(b, a).pvalue)
                    except Exception:
                        pw = None
                stats[t][k]["p_t"], stats[t][k]["p_w"] = pt, pw
                p_t_pairs.append((t, pt)); p_w_pairs.append((t, pw))
            for t, adj in holm(p_t_pairs).items():
                stats[t][k]["p_t_holm"] = adj
            for t, adj in holm(p_w_pairs).items():
                stats[t][k]["p_w_holm"] = adj

    # ---- CSV (long format) ----
    csv_path = os.path.join(args.out, "ablation_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family", "tag", "is_reference", "metric", "n", "mean", "std", "sem",
                    "delta_vs_full", "p_ttest", "p_ttest_holm", "p_wilcoxon", "p_wilcoxon_holm",
                    "source_log"])
        for tag in sorted(by_tag):
            fam = family_of(tag)
            is_ref = (tag == f"{fam}_full")
            for k in METRICS:
                s = stats[tag][k]
                fmt = lambda x: ("" if x is None else round(x, 6))
                w.writerow([fam, tag, int(is_ref), k, s["n"], round(s["mean"], 4),
                            round(s["std"], 4), round(s["sem"], 4), fmt(s["delta"]),
                            fmt(s["p_t"]), fmt(s["p_t_holm"]), fmt(s["p_w"]), fmt(s["p_w_holm"]),
                            by_tag[tag][1]])

    # ---- Markdown (per-family tables, mean+/-std with significance markers) ----
    def marker(s, is_ref):
        if is_ref:
            return ""
        p = s["p_t_holm"]
        if p is None:
            return ""
        return "**" if p < 0.01 else ("*" if p < 0.05 else "")

    md_path = os.path.join(args.out, "ablation_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# E3 ablation summary\n\n")
        f.write("Mean ± sample-std (n−1) over runs. Significance = Holm-adjusted **paired** "
                "t-test vs the family's `*_full` reference: `**` p<0.01, `*` p<0.05 (two-sided). "
                "Δ and both tests (t-test + Wilcoxon, raw + Holm) are in `ablation_summary.csv`.\n\n")
        for fam, label in (("comp", "Component ablation"), ("ctx", "Relational-context ablation"),
                           ("other", "Other")):
            tags = sorted(families.get(fam, []))
            if not tags:
                continue
            ref_tag = f"{fam}_full"
            f.write(f"## {label}\n\n")
            f.write("| tag | n | " + " | ".join(METRICS) + " |\n")
            f.write("|" + "---|" * (len(METRICS) + 2) + "\n")
            for tag in tags:
                is_ref = (tag == ref_tag)
                name = tag + ("  _(ref)_" if is_ref else "")
                cells = []
                n = stats[tag][METRICS[0]]["n"]
                for k in METRICS:
                    s = stats[tag][k]
                    cells.append(f"{s['mean']:.3f} ± {s['std']:.3f}{marker(s, is_ref)}")
                f.write(f"| {name} | {n} | " + " | ".join(cells) + " |\n")
            f.write("\n")
        if ref_missing:
            f.write(f"_Note: no `*_full` reference found for: {', '.join(ref_missing)} "
                    f"→ no paired tests for that family._\n")
        if not HAVE_SCIPY:
            f.write("_scipy not installed → p-values omitted; only descriptive stats shown._\n")

    # ---- console ----
    print(f"[summary] {len(by_tag)} tag(s) -> {csv_path} , {md_path}\n")
    for fam in ("comp", "ctx", "other"):
        tags = sorted(families.get(fam, []))
        if not tags:
            continue
        print(f"== {fam} ==")
        hdr = f"{'tag':22s} {'n':>3s}  " + "  ".join(f"{k:>9s}" for k in METRICS)
        print(hdr); print("-" * len(hdr))
        for tag in tags:
            is_ref = (tag == f"{fam}_full")
            cells = "  ".join(f"{stats[tag][k]['mean']:6.3f}{marker(stats[tag][k], is_ref):<3s}"
                              for k in METRICS)
            print(f"{tag:22s} {stats[tag][METRICS[0]]['n']:>3d}  {cells}")
        print()


if __name__ == "__main__":
    main()
