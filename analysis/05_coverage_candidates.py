"""
05_coverage_candidates.py
-------------------------
One streaming pass over PKT edges.json to compute COVERAGE (distinct source / target
nodes) for a curated set of relations relevant to the two repurposing tasks:

  TASK A  drug -> target        (chemical, molecularly interacts with, protein)
  TASK B  drug -> disease/pheno (chemical, is substance that treats, disease/phenotype)

plus candidate CONTEXT relations to enrich the subgraph (PPI, GO, pathway, gene-disease,
variant bridges, gene-protein bridge) and the chemical-chemical relations (to show PKT has
NO pharmacological drug-drug interaction, only ChEBI structural links).

Output: analysis/out/05_coverage_candidates.csv
Uses the uri->type lookup from 01. Run in conda env `gnn`.
"""
import ijson
import zipfile
import pickle
import csv
from pathlib import Path
from collections import defaultdict
from pkt_source import ensure_pkt_files   # downloads dataset/PKT/*.zip from Hugging Face if missing

PKT_DIR = Path(__file__).resolve().parents[1] / "dataset" / "PKT"
OUT_DIR = Path(__file__).resolve().parent / "out"
EDGES_ZIP = PKT_DIR / "edges.zip"
LOOKUP_PKL = OUT_DIR / "uri_to_type.pkl"

# relations to measure: (src_type, predicate_label, tgt_type) -> role label
TRACK = {
    # --- TARGET candidates ---
    ("chemical", "is substance that treats", "disease"):      "TARGET-B  drug->disease (treats)",
    ("chemical", "is substance that treats", "phenotype"):    "TARGET-B' drug->phenotype (treats)",
    ("chemical", "molecularly interacts with", "protein"):    "TARGET-A  drug->protein (DTI)",
    ("chemical", "interacts with", "protein"):                "alt DTI   drug->protein (broad)",
    ("chemical", "interacts with", "gene"):                   "alt DTI   drug->gene (broad)",
    # --- context: drug side ---
    ("chemical", "molecularly interacts with", "go"):         "ctx  drug->GO",
    ("chemical", "participates in", "pathway"):               "ctx  drug->pathway",
    # --- context: protein side (analogs of PathogenKG PPI+GO) ---
    ("protein", "molecularly interacts with", "protein"):     "ctx  PPI protein-protein",
    ("protein", "participates in", "go"):                     "ctx  protein->GO (process)",
    ("protein", "has function", "go"):                        "ctx  protein->GO (function)",
    ("protein", "located_in", "go"):                          "ctx  protein->GO (component)",
    ("protein", "participates in", "pathway"):                "ctx  protein->pathway",
    # --- context: bridges to the disease layer (esp. for TASK B) ---
    ("gene", "participates in", "pathway"):                   "ctx  gene->pathway",
    ("gene", "has gene product", "protein"):                  "ctx  gene->protein (bridge)",
    ("gene", "causes or contributes to condition", "disease"):"ctx  gene->disease (GDA)",
    ("disease", "disease has basis in dysfunction of", "gene"):"ctx disease->gene (GDA)",
    ("variant", "causes or contributes to condition", "disease"): "ctx variant->disease",
    ("variant", "causally influences", "gene"):               "ctx  variant->gene",
    ("disease", "has phenotype", "phenotype"):                "ctx  disease->phenotype",
    # --- chemical-chemical (candidate 'drug-drug'): structural/ontological, NOT DDI ---
    ("chemical", "has_role", "chemical"):                     "chem-chem ChEBI role (NOT DDI)",
    ("chemical", "has functional parent", "chemical"):        "chem-chem functional parent (NOT DDI)",
    ("chemical", "is conjugate acid of", "chemical"):         "chem-chem conjugate acid (NOT DDI)",
    ("chemical", "has part", "chemical"):                     "chem-chem has part (NOT DDI)",
    ("chemical", "is enantiomer of", "chemical"):             "chem-chem enantiomer (NOT DDI)",
}


def main():
    ensure_pkt_files(('edges.zip',))
    print("Loading uri->type lookup ...", flush=True)
    with open(LOOKUP_PKL, "rb") as fh:
        uri_to_type = pickle.load(fh)

    stats = {k: {"n": 0, "src": set(), "tgt": set()} for k in TRACK}
    n = 0
    with zipfile.ZipFile(EDGES_ZIP) as z:
        with z.open("edges.json") as f:
            for e in ijson.items(f, "item"):
                n += 1
                s_uri = e.get("source_uri"); t_uri = e.get("target_uri")
                pred = e.get("predicate_label")
                key = (uri_to_type.get(s_uri), pred, uri_to_type.get(t_uri))
                d = stats.get(key)
                if d is not None:
                    d["n"] += 1
                    d["src"].add(s_uri); d["tgt"].add(t_uri)
                if n % 2_000_000 == 0:
                    print(f"  ...{n:,} edges", flush=True)

    print(f"Total edges scanned: {n:,}")

    rows = []
    for (st, pred, tt), role in TRACK.items():
        d = stats[(st, pred, tt)]
        rows.append((role, st, pred, tt, d["n"], len(d["src"]), len(d["tgt"])))
    rows.sort(key=lambda r: r[4], reverse=True)

    out = OUT_DIR / "05_coverage_candidates.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["role", "source_type", "predicate_label", "target_type",
                    "edges", "distinct_source", "distinct_target"])
        for r in rows:
            w.writerow(r)
    print(f"Saved -> {out}")

    # console preview
    print(f"\n{'role':40s} {'edges':>10s} {'d_src':>8s} {'d_tgt':>8s}")
    for role, st, pred, tt, ne, ds, dt in rows:
        print(f"{role:40s} {ne:>10,} {ds:>8,} {dt:>8,}")


if __name__ == "__main__":
    main()
