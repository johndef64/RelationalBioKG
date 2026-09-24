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

PRE-REGISTRATION (optional). If the version folder holds a PREREGISTRATION.json (or one is given
with --prereg), the tests follow it instead of the defaults above:
  "primary":  {"comp": [...], "ctx": [...]}   Holm is applied among these only; every other variant
                                             is SECONDARY: raw p reported, never starred.
  "alternative": "less" | "two-sided"         direction of the primary tests, declared in advance
                                             ("less" = removing the piece lowers the metric).
  "equivalence": {"margin": {"MRR": 0.02}, "tags": {...}}
                                             TOST equivalence test for the "does not matter" claims:
                                             p < 0.05 means the difference is shown to lie within
                                             +/- margin, which a non-significant test cannot show.
The file must be written before the version's results are looked at; its timestamp and content are
echoed in the summary so a reader can check that. Every Delta also gets a 95% paired t interval.

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


def diff_stats(a, b, margin=None):
    """Paired difference b - a: 95% t interval and, given a margin, the TOST equivalence p."""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    if n < 2 or not HAVE_SCIPY:
        return None, None, None
    mean, sd = st.mean(d), st.stdev(d)
    se = sd / math.sqrt(n)
    tcrit = _sp.t.ppf(0.975, n - 1)
    lo, hi = mean - tcrit * se, mean + tcrit * se
    p_eq = None
    if margin is not None and se > 0:
        # two one-sided tests: difference > -margin and difference < +margin
        p_low = 1 - _sp.t.cdf((mean + margin) / se, n - 1)
        p_high = _sp.t.cdf((mean - margin) / se, n - 1)
        p_eq = max(p_low, p_high)
    return lo, hi, p_eq


def load_prereg(path):
    import json
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        pre = json.load(f)
    pre["_path"] = path
    pre["_mtime"] = __import__("datetime").datetime.fromtimestamp(os.path.getmtime(path)).isoformat(" ", "seconds")
    return pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="experiments/logs")
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--prereg", default=None,
                    help="pre-registration JSON (default: PREREGISTRATION.json next to --logdir)")
    args = ap.parse_args()
    pre = load_prereg(args.prereg or os.path.join(os.path.dirname(os.path.normpath(args.logdir)),
                                                   "PREREGISTRATION.json"))
    # one direction for all families, or {"comp": "less", "ctx": "two-sided"}
    _alt = (pre or {}).get("alternative", "two-sided")
    alt_for = (lambda fam: _alt.get(fam, "two-sided")) if isinstance(_alt, dict) else (lambda fam: _alt)
    alternative = _alt if isinstance(_alt, str) else ", ".join(f"{k}: {v}" for k, v in _alt.items())
    margins = ((pre or {}).get("equivalence") or {}).get("margin", {})
    eq_tags = ((pre or {}).get("equivalence") or {}).get("tags", {})

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
                             "p_w": None, "p_w_holm": None,
                             "ci_lo": None, "ci_hi": None, "p_equiv": None, "role": ""}

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
        # without a pre-registration every variant is tested two-sided and Holm runs over all of
        # them (the historical behaviour); with one, only the declared primaries enter Holm
        primary = set((pre or {}).get("primary", {}).get(fam, variants))
        equiv = set(eq_tags.get(fam, []))
        for k in METRICS:
            p_t_pairs, p_w_pairs = [], []
            for t in variants:
                a, b = paired(ref_runs, by_tag[t][0], k)
                stats[t][k]["delta"] = (st.mean(b) - st.mean(a)) if a else None  # variant - full
                is_primary = t in primary
                stats[t][k]["role"] = ("primary" if is_primary else "secondary") if pre else ""
                alt = alt_for(fam) if (pre and is_primary) else "two-sided"
                pt = pw = None
                if HAVE_SCIPY and len(a) >= 2 and any(x != y for x, y in zip(a, b)):
                    try:
                        pt = float(_sp.ttest_rel(b, a, alternative=alt).pvalue)
                    except Exception:
                        pt = None
                    try:
                        pw = float(_sp.wilcoxon(b, a, alternative=alt).pvalue)
                    except Exception:
                        pw = None
                stats[t][k]["p_t"], stats[t][k]["p_w"] = pt, pw
                lo, hi, peq = diff_stats(a, b, margins.get(k) if t in equiv else None)
                stats[t][k]["ci_lo"], stats[t][k]["ci_hi"], stats[t][k]["p_equiv"] = lo, hi, peq
                if is_primary:
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
                    "delta_vs_full", "ci95_lo", "ci95_hi", "role", "alternative",
                    "p_ttest", "p_ttest_holm", "p_wilcoxon", "p_wilcoxon_holm", "p_equivalence",
                    "source_log"])
        for tag in sorted(by_tag):
            fam = family_of(tag)
            is_ref = (tag == f"{fam}_full")
            for k in METRICS:
                s = stats[tag][k]
                fmt = lambda x: ("" if x is None else round(x, 6))
                alt = (alt_for(fam) if s["role"] == "primary" else "two-sided") if not is_ref else ""
                w.writerow([fam, tag, int(is_ref), k, s["n"], round(s["mean"], 4),
                            round(s["std"], 4), round(s["sem"], 4), fmt(s["delta"]),
                            fmt(s["ci_lo"]), fmt(s["ci_hi"]), s["role"], alt,
                            fmt(s["p_t"]), fmt(s["p_t_holm"]), fmt(s["p_w"]), fmt(s["p_w_holm"]),
                            fmt(s["p_equiv"]), by_tag[tag][1]])

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
        # ---- effects: delta, interval and verdict per variant, for the metrics that carry claims ----
        f.write("## Effects against the reference\n\n")
        if pre:
            f.write(f"Tests follow the pre-registration `{os.path.basename(pre['_path'])}` "
                    f"(last modified {pre['_mtime']}): primary variants are tested "
                    f"**{alternative}** with Holm among primaries only; secondary variants get a "
                    f"two-sided raw p and no star; equivalence is TOST against ±margin.\n\n")
        else:
            f.write("No pre-registration found: every variant is primary, two-sided, Holm over "
                    "the family.\n\n")

        def verdict(s):
            if s["role"] == "secondary":
                if s["p_equiv"] is not None and s["p_equiv"] < 0.05:
                    return "equivalent to reference"
                return "secondary (descriptive)"
            p = s["p_t_holm"]
            if p is not None and p < 0.05:
                return "effect confirmed"
            if s["p_equiv"] is not None and s["p_equiv"] < 0.05:
                return "equivalent to reference"
            return "not shown"

        for k in [m for m in ("MRR", "AUROC") if m in METRICS]:
            f.write(f"### {k}" + (f" (equivalence margin ±{margins[k]})" if k in margins else "") + "\n\n")
            f.write("| tag | role | Δ | 95% CI | p (Holm if primary) | p equivalence | verdict |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for fam in ("comp", "ctx"):
                for tag in sorted(t for t in families.get(fam, []) if t != f"{fam}_full"):
                    s = stats[tag][k]
                    if s["delta"] is None:
                        continue
                    p = s["p_t_holm"] if s["role"] != "secondary" else s["p_t"]
                    ci = (f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]" if s["ci_lo"] is not None else "")
                    pe = "" if s["p_equiv"] is None else f"{s['p_equiv']:.3g}"
                    ps = "" if p is None else f"{p:.3g}"
                    f.write(f"| {tag} | {s['role'] or 'primary'} | {s['delta']:+.3f} | {ci} | {ps} | {pe} | "
                            f"{verdict(s)} |\n")
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
