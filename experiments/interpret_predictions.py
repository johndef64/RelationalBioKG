"""
interpret_predictions.py  —  INTERPRETABILITY layer for the novel links a mature model finds.

IMPORTANT — this is interpretability, NOT validation. It explains *why* the model made a
prediction by surfacing supporting structure from the SAME KG the model trained on
(shared pathway / PPI neighbour / GO for Task A; molecular meta-path / phenotype overlap for
Task B). Because the evidence comes from the same graph, it CANNOT independently confirm a
prediction is true — it is circular w.r.t. the model's own signal. The only non-circular
quantity here is `held_out_recovered` (is_test_target): those positives were removed from
training, so recovering them is genuine predictive evidence.

Use it as (a) a held-out recovery report and (b) a KG-grounded triage that ranks candidates
and attaches a rationale, to feed the human expert review (see expert_review_script.py).
For a truly external validation (breaking circularity) use a time-split or an unintegrated
external DB — not this script.

The "tier" here is an AUTOMATIC triage label (not an expert judgement). In PathogenKG there was
no such label at all: the model produced the ranking and a domain expert assigned the three
plausibility tiers from the literature. This one is a KG-grounded triage added on top, and it is
deliberately built from ONE kind of evidence only:
  Tier 1  >=2 independent KG-evidence types
  Tier 2  exactly 1 KG-evidence type
  Tier 3  embedding-only (no in-KG rationale — genuine novel hypothesis)
`held_out_recovered` is kept as a SEPARATE column and is never folded into the tier. Merging the
two would put the only non-circular quantity in the same bucket as the circular one, which is
exactly the distinction this module exists to preserve.

Usage:
  python experiments/interpret_predictions.py \
      --rankings models/<folder>/drug_eval_results/*rankings*.json \
      --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task_type A --topk 20
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_subgraph(tsv):
    df = pd.read_csv(tsv, sep="\t", dtype=str, compression="zip" if str(tsv).endswith(".zip") else None)
    return df[["head", "interaction", "tail"]]


def undirected_adj(df, rels):
    sub = df[df["interaction"].isin(rels)]
    adj = defaultdict(set)
    for h, t in zip(sub["head"], sub["tail"]):
        adj[h].add(t); adj[t].add(h)
    return adj


def neighbors_by(df, rels, key_col, val_col):
    sub = df[df["interaction"].isin(rels)]
    m = defaultdict(set)
    for k, v in zip(sub[key_col], sub[val_col]):
        m[k].add(v)
    return m


# ---------------- Task A: protein targets ----------------
def build_taskA(df):
    return {
        "ppi": undirected_adj(df, {"PPI"}),
        "prot_go": neighbors_by(df, {"PROTEIN_GO_PROCESS", "PROTEIN_GO_FUNCTION",
                                     "PROTEIN_GO_COMPONENT"}, "head", "tail"),
        "prot_pw": neighbors_by(df, {"PROTEIN_PATHWAY"}, "head", "tail"),
        "drug_go": neighbors_by(df, {"COMPOUND_GO"}, "head", "tail"),
        "drug_pw": neighbors_by(df, {"COMPOUND_PATHWAY"}, "head", "tail"),
    }


def evidence_taskA(ctx, drug, pred_protein, known_targets):
    ev = []
    if any(pred_protein in ctx["ppi"].get(kt, ()) for kt in known_targets):
        ev.append("PPI")
    p_pw = ctx["prot_pw"].get(pred_protein, set())
    if p_pw & ctx["drug_pw"].get(drug, set()) or \
       any(p_pw & ctx["prot_pw"].get(kt, set()) for kt in known_targets):
        ev.append("PATHWAY")
    p_go = ctx["prot_go"].get(pred_protein, set())
    if p_go & ctx["drug_go"].get(drug, set()) or \
       any(p_go & ctx["prot_go"].get(kt, set()) for kt in known_targets):
        ev.append("GO")
    return ev


# ---------------- Task B: disease targets ----------------
def build_taskB(df):
    return {
        "drug_prot": neighbors_by(df, {"DTI"}, "head", "tail"),
        "prot_gene": undirected_adj(df, {"GENE_PRODUCT"}),
        "gene_dis": undirected_adj(df, {"GDA", "GDA_DYSFUNCTION"}),
        "dis_phen": neighbors_by(df, {"DISEASE_PHENOTYPE"}, "head", "tail"),
    }


def evidence_taskB(ctx, drug, pred_disease, known_diseases):
    ev = []
    genes_via_drug = set()
    for prot in ctx["drug_prot"].get(drug, ()):
        genes_via_drug |= ctx["prot_gene"].get(prot, set())
    if any(pred_disease in ctx["gene_dis"].get(g, ()) for g in genes_via_drug):
        ev.append("MOL_PATH")
    p_phen = ctx["dis_phen"].get(pred_disease, set())
    if any(p_phen & ctx["dis_phen"].get(kd, set()) for kd in known_diseases):
        ev.append("PHENOTYPE")
    return ev


def tier(ev):
    """KG-evidence triage only. Held-out recovery is reported separately, never merged in here."""
    if len(ev) >= 2:
        return 1
    if len(ev) == 1:
        return 2
    return 3


def annotate(rankings, df, task_type, topk):
    """Return a DataFrame of novel top-k predictions with KG evidence + auto tier."""
    ctx = build_taskA(df) if task_type == "A" else build_taskB(df)
    ev_fn = evidence_taskA if task_type == "A" else evidence_taskB
    rows, heldout = [], 0
    for drug, preds in rankings.items():
        known = {p["tail"] for p in preds if p.get("is_known_positive") or p.get("is_test_target")}
        novel = [p for p in preds if not p.get("is_known_positive")]
        for p in novel[:topk]:
            ev = ev_fn(ctx, drug, p["tail"], known)
            if p.get("is_test_target"):
                heldout += 1
            rows.append({
                "drug": drug, "rank": p["rank"], "prediction": p["tail"],
                "confidence": round(float(p["confidence"]), 4),
                "held_out_recovered": bool(p.get("is_test_target")),
                "kg_evidence": "+".join(ev) if ev else "-",
                "n_evidence": len(ev), "auto_tier": tier(ev),
            })
    return pd.DataFrame(rows), heldout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rankings", required=True, help="drug_eval rankings JSON (glob allowed)")
    ap.add_argument("--tsv", required=True, help="task subgraph TSV(.zip)")
    ap.add_argument("--task_type", choices=["A", "B"], required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rk = sorted(glob.glob(args.rankings))
    if not rk:
        raise SystemExit(f"No rankings file matched: {args.rankings}")
    rk = rk[-1]
    print(f"[i] rankings: {rk}")
    with open(rk) as f:
        rankings = json.load(f)
    print(f"[i] loading subgraph {args.tsv} ...")
    df = load_subgraph(args.tsv)

    res, heldout = annotate(rankings, df, args.task_type, args.topk)
    res = res.sort_values(["auto_tier", "confidence"], ascending=[True, False])
    out_prefix = args.out or str(Path(rk).with_suffix("")) + "_interpreted"
    res.to_csv(out_prefix + ".csv", index=False)

    n = len(res); tc = res["auto_tier"].value_counts().to_dict()
    print(f"[i] wrote {out_prefix}.csv | novel preds: {n} | tiers {tc} | "
          f"held-out recovered in top-{args.topk}: {heldout}")


if __name__ == "__main__":
    main()
