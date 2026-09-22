"""
stratified_analysis.py — where the model works and where it does not (paper Q3, §5.5).

Input: the per-triple rank dump of experiments/dump_test_ranks.py plus the task graph.
No training, no GPU. Output: a markdown report and a CSV per stratum.

Four questions, in the order the paper asks them.

1. DEGREE STRATA. Is the recovered signal concentrated on well-connected nodes? MRR by the
   target-relation degree of the target node (how many drugs already hit that protein / how
   many drugs already treat that disease) and by its context degree (everything else in the
   graph). If performance collapses at degree 1, the model mostly reorders what is already
   well annotated, which is the annotation-bias caveat of the discussion.

2. WHO BEATS THE TRUE TARGET. Tests the reading "the encoder recognises the neighbourhood of
   a target but does not identify the target itself": for each test triple, how many of the
   candidates the model ranks above the true one are its close relatives (PPI partner, same
   pathway, same GO molecular function), against the rate expected from a random candidate.
   An enrichment well above 1 means the competitors are relatives, i.e. the model puts the
   right family first and the wrong member on top.

3. REDUNDANCY. Is a held-out edge easy because the head already has a very similar one in
   training? For Task A: the drug already targets a protein that interacts with, or shares a
   pathway with, the held-out target. For Task B: the drug already treats a disease that
   shares phenotypes with the held-out one -- the graph carries no disease-disease edges, so
   shared phenotypes stand in for the Mondo hierarchy (stated as a proxy, not the real thing).
   MRR is reported on the redundant and non-redundant halves.

4. COLD START. Already in the E1 logs, repeated here so the strata add up.

Usage:
  python experiments/stratified_analysis.py \
      --ranks models/dti_pkt_taskA_dti.tsv_20260918_130755/test_ranks_DTI.csv \
      --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task DTI
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

LABELS_TSV = os.path.join("dataset", "PKT_subgraphs", "node_labels.tsv")

CTX = {
    "DTI": dict(
        relatives={"PPI": ("head", "tail"), "PROTEIN_PATHWAY": ("head", "tail"),
                   "PROTEIN_GO_FUNCTION": ("head", "tail")},
    ),
    "TREATS": dict(
        relatives={"DISEASE_PHENOTYPE": ("head", "tail")},
    ),
}


def load_graph(tsv):
    df = pd.read_csv(tsv, sep="\t", dtype=str,
                     compression="zip" if str(tsv).endswith(".zip") else None)
    return df[["head", "interaction", "tail"]]


def relative_sets(df, task):
    """For every target-side node: the sets it can be 'related' through.

    PPI is symmetric and node-to-node; pathway, GO and phenotype are node-to-attribute, so
    two nodes are relatives when they share an attribute.
    """
    out = {}
    for rel in CTX[task]["relatives"]:
        sub = df[df["interaction"] == rel]
        if rel == "PPI":
            adj = defaultdict(set)
            for h, t in zip(sub["head"], sub["tail"]):
                adj[h].add(t)
                adj[t].add(h)
            out[rel] = ("direct", adj)
        else:
            attr = defaultdict(set)
            for h, t in zip(sub["head"], sub["tail"]):
                attr[h].add(t)
            out[rel] = ("shared", attr)
    return out


def is_relative(rels, a, b):
    """True if a and b are related through any of the relations, with the relation names."""
    hits = []
    for rel, (kind, m) in rels.items():
        if kind == "direct":
            if b in m.get(a, ()):
                hits.append(rel)
        else:
            if m.get(a) and m.get(b) and (m[a] & m[b]):
                hits.append(rel)
    return hits


def specific_attr(df, rel, max_members):
    """Attribute sets keeping only the terms shared by at most `max_members` nodes.

    Sharing a Gene Ontology term is almost uninformative: two random proteins of the pool share
    one about two times out of three, because the annotation is dominated by very general terms.
    Restricting to specific terms is what makes "relative" mean something.
    """
    sub = df[df["interaction"] == rel]
    members = sub.groupby("tail")["head"].nunique()
    keep = set(members[members <= max_members].index)
    attr = defaultdict(set)
    for h, t in zip(sub["head"], sub["tail"]):
        if t in keep:
            attr[h].add(t)
    return attr


def family_key(label):
    """Crude protein/disease family key: the label without its last token and any digits.

    'histone deacetylase 2 (human)' and 'histone deacetylase 4 (human)' collapse to the same key,
    'fibroblast growth factor receptor 1/2/3' likewise. It is a heuristic on names, reported as
    such, and it catches exactly the paralogue case the interpretation is about.
    """
    if not isinstance(label, str) or not label.strip():
        return None
    s = label.replace("(human)", "").strip().lower()
    s = "".join(c for c in s if not c.isdigit())
    toks = [t for t in s.replace(",", " ").replace("-", " ").split() if len(t) > 2]
    return " ".join(toks[:3]) if len(toks) >= 2 else None


def mrr_table(df, col, bins, labels):
    g = pd.cut(df[col], bins=bins, labels=labels, right=True, include_lowest=True)
    t = df.groupby(g, observed=True).agg(n=("rr_mean", "size"), MRR=("rr_mean", "mean"),
                                         hits1=("rank_tail", lambda s: (s <= 1).mean()),
                                         hits10=("rank_tail", lambda s: (s <= 10).mean()))
    return t.reset_index().rename(columns={col: "stratum"})


def md_table(t, first_col):
    cols = list(t.columns)
    out = ["| " + " | ".join([first_col] + cols[1:]) + " |",
           "|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        cells = [str(r[cols[0]])]
        for c in cols[1:]:
            v = r[c]
            cells.append(f"{v:.3f}" if isinstance(v, float) else f"{v:,}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ranks", required=True, help="CSV from dump_test_ranks.py")
    p.add_argument("--tsv", required=True, help="the task graph the model was trained on")
    p.add_argument("--task", required=True, choices=["DTI", "TREATS"])
    p.add_argument("--out", default=None, help="output .md (default: next to --ranks)")
    p.add_argument("--compare", nargs="*", default=[],
                   help="further rank dumps of the SAME split (other models), as label=path, "
                        "compared with --ranks triple by triple")
    args = p.parse_args()

    df = pd.read_csv(args.ranks)
    graph = load_graph(args.tsv)
    rels = relative_sets(graph, args.task)
    L = []  # report lines

    what = "protein" if args.task == "DTI" else "disease"
    L.append(f"# Stratified analysis — {args.task} ({len(df):,} test triples)\n")
    L.append(f"Source: `{args.ranks}`. MRR is the mean of the per-triple reciprocal rank, "
             f"averaged over the two directions exactly as in E1; Hits are on the "
             f"drug$\\rightarrow${what} direction only.\n")
    overall_mrr = df["rr_mean"].mean()
    h1, h10 = (df["rank_tail"] <= 1).mean(), (df["rank_tail"] <= 10).mean()
    L.append(f"**Overall MRR {overall_mrr:.3f}**, Hits@1 {h1:.3f}, Hits@10 {h10:.3f}.\n")

    # ---------- 1. degree strata ----------
    L.append("\n## 1. By degree\n")
    L.append(f"Target-relation degree of the {what} in the training split "
             f"(how many drugs already hit it):\n")
    L.append(md_table(mrr_table(df, "tail_target_deg_train", [-1, 0, 1, 3, 10, 10**9],
                                ["0 (unseen)", "1", "2-3", "4-10", ">10"]), "degree"))
    L.append(f"\nContext degree of the {what} (every edge that is not the target relation):\n")
    L.append(md_table(mrr_table(df, "tail_ctx_deg", [-1, 0, 10, 50, 200, 10**9],
                                ["0", "1-10", "11-50", "51-200", ">200"]), "context edges"))
    L.append("\nTarget-relation degree of the drug (how many targets it already has in training):\n")
    L.append(md_table(mrr_table(df, "head_target_deg_train", [-1, 0, 1, 3, 10, 10**9],
                                ["0 (unseen)", "1", "2-3", "4-10", ">10"]), "degree"))

    # ---------- 2. competitors ----------
    if "competitors" in df.columns and df["competitors"].notna().any():
        L.append("\n## 2. What beats the true target\n")
        pool = sorted(set(graph[graph["interaction"] == args.task]["tail"]))
        rng = np.random.RandomState(0)
        beaten = df[(df["rank_tail"] > 1) & df["competitors"].fillna("").ne("")]

        # criteria, from the loosest to the strictest
        crit = {}
        for rel, (kind, m) in rels.items():
            crit[rel] = (lambda a, b, m=m, kind=kind:
                         (b in m.get(a, ())) if kind == "direct"
                         else bool(m.get(a) and m.get(b) and (m[a] & m[b])))
        if args.task == "DTI":
            spec_go = specific_attr(graph, "PROTEIN_GO_FUNCTION", 50)
            spec_pw = specific_attr(graph, "PROTEIN_PATHWAY", 50)
            crit["GO function (specific, <=50 proteins)"] = \
                lambda a, b: bool(spec_go.get(a) and spec_go.get(b) and (spec_go[a] & spec_go[b]))
            crit["pathway (specific, <=50 proteins)"] = \
                lambda a, b: bool(spec_pw.get(a) and spec_pw.get(b) and (spec_pw[a] & spec_pw[b]))
        labels = {}
        if os.path.exists(LABELS_TSV):
            lab_df = pd.read_csv(LABELS_TSV, sep="\t", dtype=str).fillna("")
            labels = dict(zip(lab_df["entity"], lab_df["label"]))
        labels.update({e: l for e, l in zip(df["tail"], df["tail_label"].fillna(""))})
        fam = {e: family_key(labels.get(e, "")) for e in set(pool) | set(df["tail"])}
        crit["same family (label heuristic)"] = \
            lambda a, b: bool(fam.get(a)) and fam.get(a) == fam.get(b)

        L.append(f"Triples whose true {what} is not ranked first: {len(beaten):,} "
                 f"({len(beaten) / len(df):.1%}). For each of them, the model's top-10 "
                 f"competitors are compared with 10 candidates drawn at random from the pool.\n")
        rows = []
        for name, fn in crit.items():
            obs, base = [], []
            for _, r in beaten.iterrows():
                comp = str(r["competitors"]).split(";")
                t = r["tail"]
                obs.append(np.mean([fn(t, c) for c in comp]))
                rnd = rng.choice(pool, size=len(comp), replace=False)
                base.append(np.mean([fn(t, c) for c in rnd]))
            o, b = float(np.mean(obs)), float(np.mean(base))
            rows.append(dict(criterion=name, competitors=o, random=b,
                             enrichment=(o / b if b else float("nan"))))
        L.append(md_table(pd.DataFrame(rows), "criterion"))
        L.append("\n_Read: an enrichment well above 1 means the candidates the model prefers to the "
                 "true target are its close relatives, which is the behaviour the expert review "
                 "also found. The label heuristic is the closest thing here to a paralogue test; "
                 "the family-key column of the per-triple CSV allows it to be checked by hand._\n")

    # ---------- 3. redundancy ----------
    L.append("\n## 3. Redundancy of the held-out edge\n")
    train_targets = defaultdict(set)
    tgt = graph[graph.interaction == args.task]
    test_pairs = set(zip(df["head"], df["tail"]))
    for h, t in zip(tgt["head"], tgt["tail"]):
        if (h, t) not in test_pairs:
            train_targets[h].add(t)
    red = []
    for _, r in df.iterrows():
        known = train_targets.get(r["head"], set())
        red.append(any(is_relative(rels, r["tail"], k) for k in known))
    df["redundant"] = red
    t = df.groupby("redundant").agg(n=("rr_mean", "size"), MRR=("rr_mean", "mean"),
                                    hits1=("rank_tail", lambda s: (s <= 1).mean()))
    t = t.reset_index()
    t["redundant"] = t["redundant"].map({True: "yes", False: "no"})
    L.append(f"A test triple counts as redundant when the drug already has, in training, a "
             f"{what} related to the held-out one "
             f"({', '.join(CTX[args.task]['relatives'])}).\n")
    L.append(md_table(t, "redundant"))
    if args.task == "TREATS":
        L.append("\n_The graph has no disease-disease edges, so shared phenotypes stand in for the "
                 "Mondo hierarchy: this is a proxy and understates hierarchical redundancy._\n")

    # ---------- 4. cold start ----------
    L.append("\n## 4. Cold start\n")
    t = df.groupby("cold_start").agg(n=("rr_mean", "size"), MRR=("rr_mean", "mean"))
    t = t.reset_index()
    t["cold_start"] = t["cold_start"].map({True: "cold", False: "warm"})
    L.append(md_table(t, "triples"))

    # ---------- 5. supervision available for the drug ----------
    L.append("\n## 5. How much the drug itself brings\n")
    L.append("Three regimes, by what the drug has in the training graph. The middle one is the "
             "case an encoder should win: no known target to memorise, but a neighbourhood to "
             "aggregate over.\n")
    grp = pd.Series("warm: the drug has known targets", index=df.index)
    grp[(df["head_target_deg_train"] == 0) & (~df["cold_start"])] = \
        "context but no known target"
    grp[df["cold_start"]] = "cold start: an endpoint has no edge at all"
    df["regime"] = grp
    others = {}
    for spec in args.compare:
        label, path = spec.split("=", 1)
        o = pd.read_csv(path)
        if len(o) != len(df) or not (o["head"] == df["head"]).all():
            raise SystemExit(f"[!] {path} is not aligned with --ranks: different split?")
        others[label] = o
    tab = pd.DataFrame({"this model": df.groupby(grp)["rr_mean"].mean()})
    for label, o in others.items():
        tab[label] = o.groupby(grp)["rr_mean"].mean()
    tab.insert(0, "n", grp.value_counts())
    L.append(md_table(tab.reset_index().rename(columns={"index": "regime"}), "regime"))
    if others:
        try:
            from scipy.stats import wilcoxon
            sel = grp == "context but no known target"
            lines = []
            for label, o in others.items():
                p_val = wilcoxon(df.loc[sel, "rr_mean"], o.loc[sel, "rr_mean"]).pvalue
                lines.append(f"{label}: p = {p_val:.2g}")
            L.append("\nPaired Wilcoxon on the middle regime, this model against " +
                     "; ".join(lines) + ".\n")
        except ImportError:
            pass

    L.append("\n## Not covered here\n")
    L.append("- understudied proteins (Pharos target development levels) and rare diseases: both "
             "need an external annotation file, which is not in the repository. Context degree "
             "above is the graph-internal proxy.\n")

    out = args.out or os.path.join(os.path.dirname(args.ranks), f"stratified_{args.task}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    df.to_csv(out.replace(".md", "_per_triple.csv"), index=False)
    print("\n".join(L))
    print(f"\n[i] written: {out}")


if __name__ == "__main__":
    main()
