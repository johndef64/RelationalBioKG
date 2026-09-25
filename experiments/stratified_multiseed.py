"""
stratified_multiseed.py — the stratified analysis over ALL twelve E1 seeds of each model.

stratified_analysis.py looks at one checkpoint per model (the one with the best test MRR). This
script repeats its key strata on every seed and reports mean ± sd across seeds, with Welch's test
against the reference model (the same test as E1), so that each stratified difference carries the
seed variability instead of resting on one checkpoint.

Input: the per-seed rank dumps written by
    PYTHONHASHSEED=0 python experiments/dump_test_ranks.py --model_folder <folder> --tsv ... --task ... --run all
i.e. <folder>/ranks/test_ranks_<TASK>_run<i>.csv. The strata (regime, degree, redundancy) depend on
the split only, which is the same for every seed and model, and this is checked.

Usage:
  python experiments/stratified_multiseed.py --task DTI --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip \
      --model R-GCN=models/dti_pkt_taskA_dti.tsv_20260918_130755 \
      --model CompGCN=models/dti_pkt_taskA_dti.tsv_20260918_135629 \
      --model DistMult=models/dti_pkt_taskA_dti.tsv_20260918_153101 \
      --reference DistMult --out docs/stratified_multiseed_DTI.md
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stratified_analysis import (LABELS_TSV, CTX, load_graph, relative_sets, is_relative,  # noqa: E402
                                 specific_attr, family_key)

REGIMES = ["warm: the drug has known targets", "context but no known target",
           "cold start: an endpoint has no edge at all"]
DEG_BINS = ([-1, 0, 1, 3, 10, 10**9], ["0 (unseen)", "1", "2-3", "4-10", ">10"])
CTX_BINS = ([-1, 0, 10, 50, 200, 10**9], ["0", "1-10", "11-50", "51-200", ">200"])


def strata(df, redundant):
    """Model-independent labels of every test triple (same split for all seeds and models)."""
    s = pd.DataFrame(index=df.index)
    reg = pd.Series(REGIMES[0], index=df.index)
    reg[(df["head_target_deg_train"] == 0) & (~df["cold_start"])] = REGIMES[1]
    reg[df["cold_start"]] = REGIMES[2]
    s["regime"] = reg
    s["tail_deg"] = pd.cut(df["tail_target_deg_train"], DEG_BINS[0], labels=DEG_BINS[1]).astype(str)
    s["head_deg"] = pd.cut(df["head_target_deg_train"], DEG_BINS[0], labels=DEG_BINS[1]).astype(str)
    s["tail_ctx"] = pd.cut(df["tail_ctx_deg"], CTX_BINS[0], labels=CTX_BINS[1]).astype(str)
    s["redundant"] = np.where(redundant, "yes", "no")
    return s


def redundancy(df, graph, rels, task):
    train_targets = defaultdict(set)
    tgt = graph[graph.interaction == task]
    test_pairs = set(zip(df["head"], df["tail"]))
    for h, t in zip(tgt["head"], tgt["tail"]):
        if (h, t) not in test_pairs:
            train_targets[h].add(t)
    return np.array([any(is_relative(rels, t, k) for k in train_targets.get(h, ()))
                     for h, t in zip(df["head"], df["tail"])])


def competitor_criteria(graph, task, rels, df):
    crit = {}
    for rel, (kind, m) in rels.items():
        crit[rel] = (lambda a, b, m=m, kind=kind:
                     (b in m.get(a, ())) if kind == "direct"
                     else bool(m.get(a) and m.get(b) and (m[a] & m[b])))
    if task == "DTI":
        spec_go = specific_attr(graph, "PROTEIN_GO_FUNCTION", 50)
        spec_pw = specific_attr(graph, "PROTEIN_PATHWAY", 50)
        crit["GO function (specific)"] = \
            lambda a, b: bool(spec_go.get(a) and spec_go.get(b) and (spec_go[a] & spec_go[b]))
        crit["pathway (specific)"] = \
            lambda a, b: bool(spec_pw.get(a) and spec_pw.get(b) and (spec_pw[a] & spec_pw[b]))
    labels = {}
    if os.path.exists(LABELS_TSV):
        lab = pd.read_csv(LABELS_TSV, sep="\t", dtype=str).fillna("")
        labels = dict(zip(lab["entity"], lab["label"]))
    pool = sorted(set(graph[graph["interaction"] == task]["tail"]))
    labels.update({e: l for e, l in zip(df["tail"], df["tail_label"].fillna(""))})
    fam = {e: family_key(labels.get(e, "")) for e in set(pool) | set(df["tail"])}
    crit["same family (label)"] = lambda a, b: bool(fam.get(a)) and fam.get(a) == fam.get(b)
    return crit, pool


def enrichment(df, crit, pool, seed, max_rows):
    """Competitor relatedness vs random candidates, one number per criterion, for one dump."""
    beaten = df[(df["rank_tail"] > 1) & df["competitors"].fillna("").ne("")]
    if max_rows and len(beaten) > max_rows:
        beaten = beaten.sample(max_rows, random_state=seed)
    rng = np.random.RandomState(seed)
    out = {}
    comps = [str(c).split(";") for c in beaten["competitors"]]
    rnds = [rng.choice(pool, size=len(c), replace=False) for c in comps]
    for name, fn in crit.items():
        obs = np.mean([np.mean([fn(t, c) for c in cs]) for t, cs in zip(beaten["tail"], comps)])
        base = np.mean([np.mean([fn(t, c) for c in rs]) for t, rs in zip(beaten["tail"], rnds)])
        out[name] = obs / base if base else float("nan")
    return out


def fmt(vals):
    v = np.asarray(vals, dtype=float)
    return f"{np.nanmean(v):.3f} ± {np.nanstd(v, ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def pval(a, b):
    if len(a) < 2 or len(b) < 2:
        return ""
    p = ttest_ind(a, b, equal_var=False).pvalue
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["DTI", "TREATS"])
    p.add_argument("--tsv", required=True)
    p.add_argument("--model", action="append", required=True, help="label=model_folder (repeatable)")
    p.add_argument("--reference", required=True, help="label of the model the others are tested against")
    p.add_argument("--out", required=True, help="output .md; a _per_seed.csv is written next to it")
    p.add_argument("--competitor_rows", type=int, default=0,
                   help="subsample the beaten triples for the competitor enrichment (0 = all)")
    args = p.parse_args()

    graph = load_graph(args.tsv)
    rels = relative_sets(graph, args.task)
    models = dict(m.split("=", 1) for m in args.model)
    if args.reference not in models:
        raise SystemExit(f"--reference {args.reference} is not among {list(models)}")

    dumps = {}
    for label, folder in models.items():
        files = glob.glob(os.path.join(folder, "ranks", f"test_ranks_{args.task}_run*.csv"))
        runs = sorted(int(re.search(r"_run(\d+)\.csv$", f).group(1)) for f in files)
        if not runs:
            raise SystemExit(f"[!] no per-seed dumps in {folder}/ranks/: run dump_test_ranks.py --run all")
        dumps[label] = {r: os.path.join(folder, "ranks", f"test_ranks_{args.task}_run{r}.csv") for r in runs}
        print(f"[i] {label}: {len(runs)} seeds")

    first = pd.read_csv(next(iter(next(iter(dumps.values())).values())))
    red = redundancy(first, graph, rels, args.task)
    S = strata(first, red)
    crit, pool = competitor_criteria(graph, args.task, rels, first)
    groupings = ["regime", "tail_deg", "head_deg", "tail_ctx", "redundant"]

    rows = []  # one row per (model, seed)
    for label, runs in dumps.items():
        for r, path in runs.items():
            df = pd.read_csv(path)
            if len(df) != len(first) or not ((df["head"] == first["head"]) & (df["tail"] == first["tail"])).all():
                raise SystemExit(f"[!] {path} is not aligned with the other dumps: different split?")
            row = {"model": label, "run": r, "MRR": df["rr_mean"].mean()}
            for g in groupings:
                for k, v in df.groupby(S[g])["rr_mean"].mean().items():
                    row[f"{g}={k}"] = v
            for k, v in enrichment(df, crit, pool, r, args.competitor_rows).items():
                row[f"enrich={k}"] = v
            rows.append(row)
            print(f"    {label} run {r}: MRR {row['MRR']:.4f}")
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(args.out.replace(".md", "_per_seed.csv"), index=False)

    labels = list(models)
    others = [m for m in labels if m != args.reference]
    L = [f"# Stratified analysis over seeds — {args.task}\n",
         f"{len(first):,} test triples, the same split for every model and seed. Each cell is the mean ± sd "
         f"over seeds ({', '.join(f'{m}: {len(dumps[m])}' for m in labels)}) of the MRR inside the stratum; "
         f"p is Welch's two-sided test against {args.reference} over seeds, as in E1. "
         f"Generated by `experiments/stratified_multiseed.py`.\n"]

    def block(title, prefix, keys, note=""):
        L.append(f"\n## {title}\n")
        if note:
            L.append(note + "\n")
        head = ["stratum", "n"] + labels + [f"p {m} vs {args.reference}" for m in others]
        L.append("| " + " | ".join(head) + " |")
        L.append("|" + "---|" * len(head))
        for k in keys:
            col = f"{prefix}={k}" if prefix else k
            if col not in per_seed:
                continue
            n = int((S[prefix] == k).sum()) if prefix in S else len(first)
            vals = {m: per_seed.loc[per_seed.model == m, col].dropna().values for m in labels}
            cells = [k, f"{n:,}"] + [fmt(vals[m]) for m in labels] + \
                    [pval(vals[m], vals[args.reference]) for m in others]
            L.append("| " + " | ".join(cells) + " |")

    L.append("\n## Overall\n")
    L.append("| | " + " | ".join(labels) + " |")
    L.append("|" + "---|" * (len(labels) + 1))
    L.append("| MRR | " + " | ".join(fmt(per_seed.loc[per_seed.model == m, "MRR"]) for m in labels) + " |")

    block("1. By what the drug brings (supervision regime)", "regime", REGIMES,
          "The middle regime is where an encoder should win: no known target to memorise, but a "
          "neighbourhood to aggregate over.")
    # seeds on which each encoder beats the reference in the middle regime
    mid = f"regime={REGIMES[1]}"
    for m in others:
        a = per_seed.loc[per_seed.model == m, mid].values
        b = per_seed.loc[per_seed.model == args.reference, mid].values
        L.append(f"\n{m}: lowest seed {a.min():.3f} against the highest {args.reference} seed {b.max():.3f} "
                 f"in the middle regime" + (" (no overlap)." if a.min() > b.max() else "."))
    what = "protein" if args.task == "DTI" else "disease"
    block(f"2. By target-relation degree of the {what}", "tail_deg", DEG_BINS[1])
    block(f"3. By context degree of the {what}", "tail_ctx", CTX_BINS[1])
    block("4. By target-relation degree of the drug", "head_deg", DEG_BINS[1])
    block("5. Redundancy of the held-out edge", "redundant", ["yes", "no"],
          f"Redundant: the drug already has, in training, a {what} related to the held-out one "
          f"({', '.join(CTX[args.task]['relatives'])}).")

    L.append("\n## 6. What beats the true target: enrichment over random candidates\n")
    L.append("Ratio between how often the top-10 competitors are related to the true "
             f"{what} and how often 10 random candidates are. Above 1 = the model prefers relatives.\n")
    head = ["criterion"] + labels + [f"p {m} vs {args.reference}" for m in others]
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))
    for name in crit:
        col = f"enrich={name}"
        vals = {m: per_seed.loc[per_seed.model == m, col].dropna().values for m in labels}
        L.append("| " + " | ".join([name] + [fmt(vals[m]) for m in labels] +
                                    [pval(vals[m], vals[args.reference]) for m in others]) + " |")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[i] written: {args.out} and {args.out.replace('.md', '_per_seed.csv')}")


if __name__ == "__main__":
    main()
