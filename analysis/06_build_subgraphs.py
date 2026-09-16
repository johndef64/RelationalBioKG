"""
06_build_subgraphs.py
---------------------
Extract task-specific repurposing subgraphs from the PKT KG and write them in the
EXACT format the PathogenKG framework expects:

    TSV  head <TAB> interaction <TAB> tail <TAB> source <TAB> type
    entity = "<Type>::<entity_id>"   (node type = prefix before "::")

The framework (src/utils.load_data / set_target_label) uses ONLY head/interaction/tail;
node types come from the "::" prefix; the task target is selected at runtime by
`--task <INTERACTION_NAME>`. So we:

  * RENAME relations to be type-constrained (fixes the "molecularly interacts with"
    overloading: it is PPI between proteins, DTI between chemical+protein, drug-GO
    between chemical+go — here they become distinct interaction names PPI / DTI / COMPOUND_GO).
  * Keep ONE direction of every inverse pair (the framework re-adds reverse edges itself).
  * De-duplicate; symmetric same-type relations (PPI) are deduped as unordered pairs.

TWO LAYERS OF EVIDENCE (see TICKET_01_DTI_drug_scope.md)
--------------------------------------------------------
PheKnowLator has no pharmacological source: its chemical->protein edges come from Reactome
(reaction participation), UniProt catalysts and CTD toxicogenomics, so they describe
substrates, cofactors and products. Measured on the first build: the ten hub compounds
(hydron, water, ATP, ADP, phosphate, magnesium) carried 31.4% of that relation, and only
12.5% of its edges involved a compound with any therapeutic use. Predicting it means
predicting known biochemistry, not medicine.

This build therefore keeps the two kinds of evidence apart:

  CPI_BIOCHEM   chemical--protein edges of PheKnowLator (biochemistry) -> CONTEXT
  DTI           drug--target edges injected from DrugBank via UniProt  -> TASK A TARGET

The injected layer is produced by `analysis/10_build_dti_drugbank.py`, which inverts the
DrugBank cross-references published by UniProt (open) and uses the DrugBank release only for
drug names and regulatory status. It is also added to Task B as context: 34.3% of the
compounds with a TREATS edge gain a molecular target this way (4.9% had one before), which is
what closes the drug -> target -> pathway -> disease chain.

  # DrugBank release used for names/status (Kaggle mirror of DrugBank 5.1.10):
  curl -L -o dataset/DRUGBANK/drug-bank-5110.zip \
    https://www.kaggle.com/api/v1/datasets/download/devildev89/drug-bank-5110
  # then, before this script:
  python analysis/10_build_dti_drugbank.py     # -> dataset/PKT_subgraphs/dti_drugbank_edges.tsv

Outputs (dataset/PKT_subgraphs/):
  pkt_taskA_dti.tsv.zip      TASK A: predict DTI  (drug -> protein)       --task DTI
  pkt_taskB_treats.tsv.zip   TASK B: predict TREATS (chemical -> disease) --task TREATS
  pkt_unified.tsv.zip        both targets + all context (multi-task)      --task DTI,TREATS
  06_subgraph_stats.md       per-relation / per-task statistics, with the origin breakdown

Run in conda env `gnn`.  Toggle INCLUDE_VARIANT below to add the genetic (variant) layer to
Task B / unified (heavier: ~145k extra nodes -> use neighbor sampling downstream if enabled).
Use --no-pharma to rebuild the legacy graphs, where the biochemical relation is itself the
Task A target (kept only to reproduce the earlier results).
"""
import argparse
import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import ijson

PKT_DIR = Path(__file__).resolve().parents[1] / "dataset" / "PKT"
OUT_DIR = Path(__file__).resolve().parents[1] / "dataset" / "PKT_subgraphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATS_MD = Path(__file__).resolve().parent / "out" / "06_subgraph_stats.md"
PHARMA_TSV = OUT_DIR / "dti_drugbank_edges.tsv"     # written by analysis/10_build_dti_drugbank.py

# provenance written in the TSV `source` column, and used for the origin breakdown
SRC_PKT = "PKT"
SRC_PHARMA = "DrugBank/UniProt"

INCLUDE_VARIANT = False   # add variant->gene / variant->disease layer to Task B & unified

# node bioentity_type -> entity string prefix (node type seen by the framework)
TYPE_PREFIX = {
    "chemical": "Compound",
    "protein":  "Protein",
    "gene":     "Gene",
    "go":       "GO",
    "pathway":  "Pathway",
    "disease":  "Disease",
    "phenotype":"Phenotype",
    "variant":  "Variant",
}

# (src_type, predicate_label, tgt_type) -> (relation_name, symmetric)
# relation_name is the type-constrained interaction written to the TSV.
RELATION_MAP = {
    # ---- targets / pharmacology ----
    # NB: PheKnowLator's chemical->protein edges are biochemical (Reactome reactions, UniProt
    # catalysts, CTD), not pharmacological: they are context here, and the Task A target is the
    # injected DTI layer. --no-pharma restores the legacy naming (this relation as the target).
    ("chemical", "molecularly interacts with", "protein"):        ("CPI_BIOCHEM", False),
    ("chemical", "is substance that treats",   "disease"):        ("TREATS", False),   # TASK B target
    # ---- molecular CORE (shared by both tasks) ----
    ("protein", "molecularly interacts with", "protein"):         ("PPI", True),
    ("protein", "participates in", "go"):                         ("PROTEIN_GO_PROCESS", False),
    ("protein", "has function", "go"):                            ("PROTEIN_GO_FUNCTION", False),
    ("protein", "located_in", "go"):                              ("PROTEIN_GO_COMPONENT", False),
    ("protein", "participates in", "pathway"):                    ("PROTEIN_PATHWAY", False),
    ("gene", "has gene product", "protein"):                      ("GENE_PRODUCT", False),   # gene<->protein bridge
    # ---- drug-side context (both tasks) ----
    ("chemical", "molecularly interacts with", "go"):             ("COMPOUND_GO", False),
    ("chemical", "participates in", "pathway"):                   ("COMPOUND_PATHWAY", False),
    # ---- disease layer (TASK B) ----
    ("gene", "participates in", "pathway"):                       ("GENE_PATHWAY", False),
    ("gene", "causes or contributes to condition", "disease"):    ("GDA", False),
    ("disease", "disease has basis in dysfunction of", "gene"):   ("GDA_DYSFUNCTION", False),
    ("disease", "has phenotype", "phenotype"):                    ("DISEASE_PHENOTYPE", False),
    # ---- genetic/variant layer (optional, TASK B) ----
    ("variant", "causally influences", "gene"):                   ("VARIANT_GENE", False),
    ("variant", "causes or contributes to condition", "disease"): ("VARIANT_DISEASE", False),
}

CORE = {"PPI", "PROTEIN_GO_PROCESS", "PROTEIN_GO_FUNCTION", "PROTEIN_GO_COMPONENT",
        "PROTEIN_PATHWAY", "GENE_PRODUCT", "COMPOUND_GO", "COMPOUND_PATHWAY", "CPI_BIOCHEM",
        "DRUG_ADME"}

# DTI = injected pharmacological layer (target of Task A, context of Task B)
TASK_A = CORE | {"DTI"}
TASK_B = CORE | {"TREATS", "DTI", "GENE_PATHWAY", "GDA", "GDA_DYSFUNCTION", "DISEASE_PHENOTYPE"}
VARIANT_RELS = {"VARIANT_GENE", "VARIANT_DISEASE"}
if INCLUDE_VARIANT:
    TASK_B |= VARIANT_RELS
UNIFIED = TASK_A | TASK_B

NEEDED_TYPES = set(TYPE_PREFIX) if INCLUDE_VARIANT else (set(TYPE_PREFIX) - {"variant"})


def build_node_lookup():
    """uri -> 'Type::entity_id' for the node types we need."""
    print("Reading nodes.json ...", flush=True)
    uri2node = {}
    with zipfile.ZipFile(PKT_DIR / "nodes.zip") as z, z.open("nodes.json") as f:
        for o in ijson.items(f, "item"):
            bt = o.get("bioentity_type")
            if bt in NEEDED_TYPES:
                eid = o.get("entity_id")
                if eid:
                    uri2node[o["uri"]] = f"{TYPE_PREFIX[bt]}::{eid}"
    print(f"  mapped {len(uri2node):,} nodes", flush=True)
    return uri2node


def extract_edges(uri2node):
    """Stream edges once; return {relation_name: set of (head, tail)} deduped."""
    edges = defaultdict(set)
    n = 0
    with zipfile.ZipFile(PKT_DIR / "edges.zip") as z, z.open("edges.json") as f:
        for e in ijson.items(f, "item"):
            n += 1
            s_uri = e.get("source_uri"); t_uri = e.get("target_uri")
            # classify by bioentity_type via the prefix we stored — need raw types:
            # reconstruct type from mapped string prefix is possible, but we keyed
            # RELATION_MAP on raw bioentity_type, so look them up from the map key using
            # node strings' prefixes.
            hs = uri2node.get(s_uri); ts = uri2node.get(t_uri)
            if hs is None or ts is None:
                continue
            s_type = PREFIX2TYPE[hs.split("::", 1)[0]]
            t_type = PREFIX2TYPE[ts.split("::", 1)[0]]
            spec = RELATION_MAP.get((s_type, e.get("predicate_label"), t_type))
            if spec is None:
                continue
            rel, symmetric = spec
            if symmetric:
                a, b = (hs, ts) if hs <= ts else (ts, hs)
                if a == b:      # drop self loops (framework adds its own)
                    continue
                edges[rel].add((a, b))
            else:
                edges[rel].add((hs, ts))
            if n % 2_000_000 == 0:
                print(f"  ...{n:,} edges scanned", flush=True)
    print(f"Total edges scanned: {n:,}")
    return edges


def load_pharma_edges(edges):
    """Add the injected drug--target layer as relation DTI. Returns how many edges were added.

    The file is produced by analysis/10_build_dti_drugbank.py; an edge already asserted by
    PheKnowLator as CPI_BIOCHEM is kept in both relations, since the two describe different
    evidence (a drug can also be a substrate of its target).
    """
    if not PHARMA_TSV.exists():
        raise SystemExit(f"missing {PHARMA_TSV} — run: python analysis/10_build_dti_drugbank.py "
                         f"(or pass --no-pharma to build the legacy graphs)")
    n = 0
    rels = defaultdict(int)
    with open(PHARMA_TSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rel = row["interaction"]           # DTI (pharmacodynamic) or DRUG_ADME (pharmacokinetic)
            edges[rel].add((row["head"], row["tail"]))
            rels[rel] += 1
            n += 1
    detail = " · ".join(f"{r}: {c:,}" for r, c in sorted(rels.items()))
    print(f"Injected pharmacological layer: {n:,} rows ({detail}); "
          f"{len(edges['DTI'] & edges.get('CPI_BIOCHEM', set())):,} DTI edges also asserted as CPI_BIOCHEM")
    return n


def write_subgraph(name, rel_names, edges, rel_source):
    """Write dataset/PKT_subgraphs/<name>.tsv.zip for the given relation set."""
    tsv_name = f"{name}.tsv"
    zip_path = OUT_DIR / f"{name}.tsv.zip"
    n_edges = 0
    node_types = defaultdict(set)
    per_source = defaultdict(int)
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\n")
    w.writerow(["head", "interaction", "tail", "source", "type"])
    for rel in sorted(rel_names):
        src = rel_source.get(rel, SRC_PKT)
        # edges are collected in sets: iterate them SORTED, or the row order of the TSV would depend
        # on Python's per-process string hashing. The train/val/test split is positional
        # (train_test_split over the target rows), so an unstable row order would silently change the
        # split between two builds of the same data.
        for h, t in sorted(edges.get(rel, ())):
            ht = h.split("::", 1)[0]; tt = t.split("::", 1)[0]
            w.writerow([h, rel, t, src, f"{ht}-{tt}"])
            node_types[ht].add(h); node_types[tt].add(t)
            per_source[src] += 1
            n_edges += 1
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(tsv_name, buf.getvalue())
    return n_edges, {k: len(v) for k, v in node_types.items()}, dict(per_source)


PREFIX2TYPE = {v: k for k, v in TYPE_PREFIX.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pharma", action="store_true",
                    help="legacy build: no injected layer, the biochemical relation is the Task A "
                         "target and keeps the name DTI")
    args = ap.parse_args()

    uri2node = build_node_lookup()
    edges = extract_edges(uri2node)

    rel_source = {}
    if args.no_pharma:
        edges["DTI"] = edges.pop("CPI_BIOCHEM", set())      # legacy naming
        task_a = TASK_A - {"CPI_BIOCHEM", "DRUG_ADME"}
        task_b = TASK_B - {"CPI_BIOCHEM", "DRUG_ADME"}
        print("[legacy] no pharmacological layer: Task A target = PheKnowLator chemical--protein edges")
    else:
        load_pharma_edges(edges)
        rel_source["DTI"] = SRC_PHARMA
        rel_source["DRUG_ADME"] = SRC_PHARMA
        task_a, task_b = TASK_A, TASK_B
    unified = task_a | task_b

    # per-relation counts
    rel_counts = {rel: len(s) for rel, s in edges.items()}

    outputs = {
        "pkt_taskA_dti":    ("TASK A — predict DTI (drug->protein)",        task_a),
        "pkt_taskB_treats": ("TASK B — predict TREATS (chemical->disease)", task_b),
        "pkt_unified":      ("UNIFIED — both targets (multi-task)",         unified),
    }
    results = {}
    for name, (desc, rels) in outputs.items():
        ne, ntypes, per_src = write_subgraph(name, rels, edges, rel_source)
        results[name] = (desc, rels, ne, ntypes, per_src)
        print(f"[{name}] {ne:,} edges  nodes={ntypes}  sources={per_src}")

    # stats markdown
    with open(STATS_MD, "w", encoding="utf-8") as fh:
        fh.write("# PKT subgraphs — build statistics\n\n")
        fh.write(f"INCLUDE_VARIANT = {INCLUDE_VARIANT} · "
                 f"pharmacological layer = {'NO (legacy build)' if args.no_pharma else 'YES'}\n\n")
        if not args.no_pharma:
            fh.write("Task A target = `DTI`, drug--target edges injected from DrugBank through the "
                     "cross-references published by UniProt (`analysis/10_build_dti_drugbank.py`). "
                     "PheKnowLator's own chemical--protein edges are biochemical (Reactome, UniProt "
                     "catalysts, CTD) and are kept as context under the name `CPI_BIOCHEM`; see "
                     "`TICKET_01_DTI_drug_scope.md`.\n\n")
        fh.write("## De-duplicated edges per relation (one direction kept)\n\n")
        fh.write("| relation | edges | origin | in A | in B | in unified |\n|---|---:|---|:--:|:--:|:--:|\n")
        for rel in sorted(rel_counts):
            fh.write(f"| {rel} | {rel_counts[rel]:,} | {rel_source.get(rel, SRC_PKT)} | "
                     f"{'✓' if rel in task_a else ''} | {'✓' if rel in task_b else ''} | "
                     f"{'✓' if rel in unified else ''} |\n")
        fh.write("\n## Per-subgraph totals\n\n")
        for name, (desc, rels, ne, ntypes, per_src) in results.items():
            fh.write(f"### {name}  — {desc}\n\n")
            fh.write(f"- file: `dataset/PKT_subgraphs/{name}.tsv.zip`\n")
            fh.write(f"- relations: {sorted(rels)}\n")
            fh.write(f"- **edges: {ne:,}**\n")
            fh.write(f"- nodes per type: {ntypes}\n")
            fh.write(f"- total nodes: {sum(ntypes.values()):,}\n")
            origin = " · ".join(f"{s}: {n:,} ({100 * n / max(1, ne):.2f}%)"
                                for s, n in sorted(per_src.items(), key=lambda kv: -kv[1]))
            fh.write(f"- edge origin: {origin}\n\n")
    print(f"\nSaved stats -> {STATS_MD}")


if __name__ == "__main__":
    main()
