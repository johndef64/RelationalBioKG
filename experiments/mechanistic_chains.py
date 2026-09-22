"""
mechanistic_chains.py — the compound -> protein -> ... -> disease chain behind a predicted target.

A drug--target prediction is a pair. What makes it a *hypothesis* a clinician can act on is the
route from that target to a disease: the model proposes protein P for drug D, the graph says P (or
its gene, or its pathway) is implicated in disease X, and either D already treats X, in which case
the prediction offers a mechanism for an indication we already have, or it does not, in which case
the pair (D, X) is a repurposing candidate with a stated route.

This script extracts those routes for pairs that came out of the expert review (or any list of
pairs), ranks them, and writes them as readable text.

WHERE THE GRAPH COMES FROM. Task A has no diseases in it, so the chains are searched in the Task B
graph, which shares the same molecular core and adds the disease side. The drug--target edge being
explained is the model's prediction, which is in neither graph: it is prepended to the chain, and
marked as predicted in the output.

META-PATHS, shortest first. From the predicted protein P to a disease X:
  1. P <-GENE_PRODUCT- G -GDA-> X                      the protein's gene is associated with X
  2. P <-GENE_PRODUCT- G <-GDA_DYSFUNCTION- X          X has its basis in dysfunction of that gene
  3. P -PROTEIN_PATHWAY-> W <-GENE_PATHWAY- G -GDA-> X the pathway bridge
  4. P -PPI-> Q <-GENE_PRODUCT- G -GDA-> X             an interaction partner carries the association
Chains through hubs explain nothing, so among chains of equal length the ones through the most
specific intermediates come first (a pathway with 12 proteins beats one with 900).

CIRCULARITY, as everywhere in E4: these routes come from the same graph the model was trained on.
They say why a prediction is coherent, not that it is true. What the chain adds over the model's
score is that a human can check each step.

Usage:
  python experiments/mechanistic_chains.py \
      --review models/<run>/drug_eval_results/expert_review_taskA_<ts>_BLIND_compiled.csv \
      --key    models/<run>/drug_eval_results/expert_review_taskA_<ts>_KEY.csv \
      --tsv    dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip

  # or for an arbitrary list of pairs (CSV with columns drug,protein as Type::id)
  python experiments/mechanistic_chains.py --pairs my_pairs.csv --tsv dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip
"""
import argparse
import math
import os
from collections import defaultdict

import pandas as pd

LABELS_TSV = os.path.join("dataset", "PKT_subgraphs", "node_labels.tsv")

# (relation, direction) steps; "<-" means the edge is stored the other way round
META_PATHS = [
    ("gene",         [("GENE_PRODUCT", "<-"), ("GDA", "->")]),
    ("dysfunction",  [("GENE_PRODUCT", "<-"), ("GDA_DYSFUNCTION", "<-")]),
    ("pathway",      [("PROTEIN_PATHWAY", "->"), ("GENE_PATHWAY", "<-"), ("GDA", "->")]),
    ("interactome",  [("PPI", "<->"), ("GENE_PRODUCT", "<-"), ("GDA", "->")]),
]


def load_graph(tsv):
    df = pd.read_csv(tsv, sep="\t", dtype=str,
                     compression="zip" if str(tsv).endswith(".zip") else None)
    fwd, bwd, deg = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set)), defaultdict(int)
    for rel, h, t in zip(df["interaction"], df["head"], df["tail"]):
        fwd[rel][h].add(t)
        bwd[rel][t].add(h)
        deg[h] += 1
        deg[t] += 1
    return df, fwd, bwd, deg


def step(fwd, bwd, rel, direction, node):
    if direction == "->":
        return fwd[rel].get(node, set())
    if direction == "<-":
        return bwd[rel].get(node, set())
    return fwd[rel].get(node, set()) | bwd[rel].get(node, set())


def chains_from(protein, fwd, bwd, deg, max_per_path=200):
    """All chains protein -> ... -> disease, as (kind, [nodes], cost). Cost = hub penalty."""
    out = []
    for kind, path in META_PATHS:
        frontier = [[protein]]
        for rel, direction in path:
            nxt = []
            for chain in frontier:
                for n in step(fwd, bwd, rel, direction, chain[-1]):
                    if n not in chain:
                        nxt.append(chain + [n])
                        if len(nxt) > max_per_path * 20:
                            break
            frontier = nxt
            if not frontier:
                break
        for chain in frontier[:max_per_path * 5]:
            if not chain[-1].startswith("Disease::"):
                continue
            cost = sum(math.log10(deg[n] + 1) for n in chain[1:-1])  # intermediates only
            out.append((kind, chain, cost))
    return out


def best_chains(protein, fwd, bwd, deg, treated, k):
    """Rank: fewest hops first, then the chains that land on a disease the drug already treats,
    then the least hubby intermediates. Length comes first because a long chain explains nothing:
    at four hops most proteins reach most diseases. One chain per disease."""
    seen, ranked = {}, []
    for kind, chain, cost in chains_from(protein, fwd, bwd, deg):
        dis = chain[-1]
        if dis not in seen or (len(chain), cost) < (len(seen[dis][1]), seen[dis][2]):
            seen[dis] = (kind, chain, cost)
    for kind, chain, cost in seen.values():
        known = chain[-1] in treated
        ranked.append((len(chain), 0 if known else 1, cost, kind, chain, known))
    ranked.sort()
    return ranked[:k]


def render(chain, kind, labels, drug):
    arrows = {"gene": ["is product of gene", "associated with"],
              "dysfunction": ["is product of gene", "dysfunction underlies"],
              "pathway": ["participates in", "with gene", "associated with"],
              "interactome": ["interacts with", "is product of gene", "associated with"]}[kind]
    def lab(e):
        return labels.get(e, e.split("::")[-1])
    parts = [f"{lab(drug)} --[predicted target]--> {lab(chain[0])}"]
    for a, n in zip(arrows, chain[1:]):
        parts.append(f"--[{a}]--> {lab(n)}")
    return " ".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", default=os.path.join("dataset", "PKT_subgraphs", "pkt_taskB_treats.tsv.zip"),
                   help="graph to walk: needs the disease side, so Task B")
    p.add_argument("--review", help="filled BLIND sheet from expert_review_script.py")
    p.add_argument("--key", help="its KEY file (gives stratum and the entity ids)")
    p.add_argument("--pairs", help="CSV with columns drug,protein (Type::id) instead of a review")
    p.add_argument("--all-items", action="store_true",
                   help="use every reviewed prediction, not only those rated plausible")
    p.add_argument("--chains", type=int, default=3, help="chains kept per pair")
    p.add_argument("--out", default=None, help="output .md (a .csv is written next to it)")
    args = p.parse_args()

    labels = {}
    if os.path.exists(LABELS_TSV):
        ldf = pd.read_csv(LABELS_TSV, sep="\t", dtype=str).fillna("")
        labels = dict(zip(ldf["entity"], ldf["label"]))

    # --- the pairs to explain ---
    if args.pairs:
        pairs = pd.read_csv(args.pairs, dtype=str).rename(
            columns={"prediction": "protein", "prediction_id": "protein"})
        pairs["tier"] = ""
        out_default = os.path.splitext(args.pairs)[0] + "_chains.md"
    else:
        if not (args.review and args.key):
            raise SystemExit("give --review and --key, or --pairs")
        rev = pd.read_csv(args.review, dtype=str)
        key = pd.read_csv(args.key, dtype=str)
        m = rev.merge(key, on="item_id", validate="one_to_one")
        m = m[m["stratum"] == "topk"]
        if not args.all_items:
            m = m[m["expert_plausible"].fillna("").str.lower().str.startswith("y")]
        pairs = pd.DataFrame(dict(drug=m["drug_id"], protein=m["prediction_id"],
                                  tier=m["expert_tier"].fillna("")))
        out_default = os.path.join(os.path.dirname(args.review), "mechanistic_chains.md")
    pairs = pairs.drop_duplicates(subset=["drug", "protein"])
    print(f"[i] pairs to explain: {len(pairs)}")

    print("[i] loading the graph ...")
    df, fwd, bwd, deg = load_graph(args.tsv)
    treats = defaultdict(set)
    for h, t in zip(df[df["interaction"] == "TREATS"]["head"],
                    df[df["interaction"] == "TREATS"]["tail"]):
        treats[h].add(t)

    rows, md = [], []
    for _, r in pairs.iterrows():
        drug, prot = r["drug"], r["protein"]
        got = best_chains(prot, fwd, bwd, deg, treats.get(drug, set()), args.chains)
        if not got:
            rows.append(dict(drug=drug, drug_label=labels.get(drug, ""), protein=prot,
                             protein_label=labels.get(prot, ""), expert_tier=r["tier"],
                             kind="", disease="", disease_label="", hops="",
                             drug_already_treats="", chain=""))
            continue
        for n_nodes, _, cost, kind, chain, known in got:
            hops = n_nodes - 1          # edges from the predicted protein to the disease
            rows.append(dict(
                drug=drug, drug_label=labels.get(drug, ""),
                protein=prot, protein_label=labels.get(prot, ""),
                expert_tier=r["tier"], kind=kind, disease=chain[-1],
                disease_label=labels.get(chain[-1], ""), hops=hops,
                drug_already_treats=known, chain=render(chain, kind, labels, drug)))

    out = pd.DataFrame(rows)
    md_path = args.out or out_default
    out.to_csv(md_path.replace(".md", ".csv"), index=False)

    # --- report ---
    have = out[out["chain"].astype(str) != ""]
    n_pairs = len(pairs)
    n_with = have[["drug", "protein"]].drop_duplicates().shape[0]
    explained = have[have["drug_already_treats"] == True][["drug", "protein"]].drop_duplicates().shape[0]
    md.append(f"# Mechanistic chains for {n_pairs} predicted drug--target pairs\n")
    md.append(f"Graph: `{args.tsv}`. Chains are ranked by length, then by how specific their "
              f"intermediates are; at most {args.chains} per pair are kept.\n")
    md.append(f"- pairs with at least one chain to a disease: **{n_with}/{n_pairs}**")
    md.append(f"- pairs whose chain reaches a disease the drug already treats: **{explained}** "
              f"(the chain explains an existing indication)")
    md.append(f"- the remaining chains end on a disease the drug does not treat in the graph: those "
              f"are repurposing candidates with a stated route.\n")
    md.append("\nThe evidence is circular with respect to the model, which learned from this same "
              "graph: a chain shows why a prediction is coherent, not that it is correct.\n")
    for (drug, prot), g in have.groupby(["drug", "protein"], sort=False):
        dl = g.iloc[0]["drug_label"] or drug
        pl = g.iloc[0]["protein_label"] or prot
        tier = g.iloc[0]["expert_tier"]
        md.append(f"\n## {dl} -> {pl}" + (f"  ({tier})" if tier else ""))
        for _, c in g.iterrows():
            tag = "explains a known indication" if c["drug_already_treats"] else "new candidate indication"
            md.append(f"- _{tag}_ ({c['kind']}, {c['hops']} hops): {c['chain']}")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[i] written: {md_path} and {md_path.replace('.md', '.csv')}")
    print("\n".join(md[:8]))


if __name__ == "__main__":
    main()
