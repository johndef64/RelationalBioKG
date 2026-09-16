"""
07_build_ablation_subgraphs.py
------------------------------
Generate RELATIONAL-CONTEXT ablation variants of a task subgraph by filtering the
already-built TSV (fast: no re-scan of the 4.6 GB edges.json).

Each variant keeps the TARGET relation and a subset of the context relations, so we can
measure how much each context layer (PPI / GO / pathway / drug-context) contributes to
link prediction — the "topology-only" ablation of the PathogenKG method on PKT.

Variants for Task A (target = 'DTI', the injected drug--target layer):
  full          all relations (== dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip)
  core_ppi      DTI + PPI only                      (bare protein network)
  no_ppi        full minus PPI
  no_go         full minus the 3 protein-GO relations
  no_pathway    full minus pathway relations
  no_drugctx    full minus COMPOUND_GO / COMPOUND_PATHWAY (drug only via DTI)
  no_biochem    full minus CPI_BIOCHEM: does PheKnowLator's biochemistry (substrates,
                cofactors, catalysis) help predict pharmacological targets?

Variants for Task B (target = 'TREATS'), with --task B:
  full          all relations
  no_pharma     full minus DTI: what does the injected pharmacological layer add to
                indication prediction? (before the injection, only 4.9% of the compounds
                with a TREATS edge had any molecular target in the graph)
  no_disease_ctx  full minus DISEASE_PHENOTYPE / GDA / GDA_DYSFUNCTION
  no_ppi / no_go / no_pathway  as above

Run in conda env `gnn`.  Outputs to dataset/PKT_subgraphs/ablation/.
"""
import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

SUB_DIR = Path(__file__).resolve().parents[1] / "dataset" / "PKT_subgraphs"
ABL_DIR = SUB_DIR / "ablation"
ABL_DIR.mkdir(parents=True, exist_ok=True)

GO_RELS = {"PROTEIN_GO_PROCESS", "PROTEIN_GO_FUNCTION", "PROTEIN_GO_COMPONENT"}
PATHWAY_RELS = {"PROTEIN_PATHWAY", "COMPOUND_PATHWAY", "GENE_PATHWAY"}
DRUGCTX_RELS = {"COMPOUND_GO", "COMPOUND_PATHWAY"}
DISEASE_CTX_RELS = {"DISEASE_PHENOTYPE", "GDA", "GDA_DYSFUNCTION"}


def write_variant(df, rels_keep, name, target_rel):
    sub = df[df["interaction"].isin(rels_keep)]
    buf = io.StringIO()
    sub.to_csv(buf, sep="\t", index=False)
    with zipfile.ZipFile(ABL_DIR / f"{name}.tsv.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}.tsv", buf.getvalue())
    n_t = int((sub["interaction"] == target_rel).sum())
    print(f"  {name:30s} edges={len(sub):>9,}  target({target_rel})={n_t:,}")
    return name, len(sub), n_t, sorted(rels_keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="A", choices=["A", "B"], help="task graph to ablate")
    args = ap.parse_args()

    if args.task == "A":
        base, target_rel, tag, title = (SUB_DIR / "pkt_taskA_dti.tsv.zip", "DTI", "ablA",
                                        "Task A (DTI, injected drug--target layer)")
    else:
        base, target_rel, tag, title = (SUB_DIR / "pkt_taskB_treats.tsv.zip", "TREATS", "ablB",
                                        "Task B (TREATS)")

    print(f"Loading base {base.name} ...")
    df = pd.read_csv(base, sep="\t", dtype=str, compression="zip")
    all_rels = set(df["interaction"].unique())
    print(f"  {len(df):,} edges, relations: {sorted(all_rels)}\n")
    if target_rel not in all_rels:
        raise SystemExit(f"target relation {target_rel} not found in {base.name}; rebuild with "
                         f"analysis/06_build_subgraphs.py")

    if args.task == "A":
        variants = {
            f"pkt_{tag}_full":       all_rels,
            f"pkt_{tag}_core_ppi":   {target_rel, "PPI"},
            f"pkt_{tag}_no_ppi":     all_rels - {"PPI"},
            f"pkt_{tag}_no_go":      all_rels - GO_RELS,
            f"pkt_{tag}_no_pathway": all_rels - PATHWAY_RELS,
            f"pkt_{tag}_no_drugctx": all_rels - DRUGCTX_RELS,
            f"pkt_{tag}_no_biochem": all_rels - {"CPI_BIOCHEM"},
        }
    else:
        variants = {
            f"pkt_{tag}_full":           all_rels,
            f"pkt_{tag}_no_pharma":      all_rels - {"DTI"},
            f"pkt_{tag}_no_biochem":     all_rels - {"CPI_BIOCHEM"},
            f"pkt_{tag}_no_disease_ctx": all_rels - DISEASE_CTX_RELS,
            f"pkt_{tag}_no_ppi":         all_rels - {"PPI"},
            f"pkt_{tag}_no_go":          all_rels - GO_RELS,
            f"pkt_{tag}_no_pathway":     all_rels - PATHWAY_RELS,
        }
    variants = {k: v for k, v in variants.items() if v != all_rels or k.endswith("_full")}

    print("Writing ablation variants:")
    rows = [write_variant(df, rels, name, target_rel) for name, rels in variants.items()]

    index = ABL_DIR / f"ablation_index_{args.task}.md"
    with open(index, "w", encoding="utf-8") as fh:
        fh.write(f"# {title} — relational-context ablation subgraphs\n\n")
        fh.write(f"Base: `{base.relative_to(SUB_DIR.parents[1])}` · target relation `{target_rel}`\n\n")
        fh.write("| variant | file | edges | target edges | relations |\n|---|---|---:|---:|---|\n")
        for name, ne, nt, rels in rows:
            fh.write(f"| {name} | `dataset/PKT_subgraphs/ablation/{name}.tsv.zip` | {ne:,} | {nt:,} | "
                     f"{', '.join(rels)} |\n")
    print(f"\nSaved -> {index}")


if __name__ == "__main__":
    main()
