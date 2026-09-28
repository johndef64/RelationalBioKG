"""
08_build_node_labels.py
-----------------------
Build a human-readable label lookup for the entity strings used in the PKT subgraphs, so
the expert-review sheet shows "acyl-CoA dehydrogenase ... (human)" instead of
"Protein::PR_Q6JQN1". Streams nodes.json once.

Output: dataset/PKT_subgraphs/node_labels.tsv  (columns: entity, label, bioentity_type)
Run in conda env `gnn`.
"""
import ijson
import zipfile
import csv
from pathlib import Path
from pkt_source import ensure_pkt_files   # downloads dataset/PKT/*.zip from Hugging Face if missing

PKT_DIR = Path(__file__).resolve().parents[1] / "dataset" / "PKT"
OUT = Path(__file__).resolve().parents[1] / "dataset" / "PKT_subgraphs" / "node_labels.tsv"

# same prefixes used by 06_build_subgraphs.py
TYPE_PREFIX = {
    "chemical": "Compound", "protein": "Protein", "gene": "Gene", "go": "GO",
    "pathway": "Pathway", "disease": "Disease", "phenotype": "Phenotype", "variant": "Variant",
}


def main():
    ensure_pkt_files(('nodes.zip',))
    n = 0
    with zipfile.ZipFile(PKT_DIR / "nodes.zip") as z, z.open("nodes.json") as f, \
         open(OUT, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out, delimiter="\t", lineterminator="\n")
        w.writerow(["entity", "label", "bioentity_type"])
        for o in ijson.items(f, "item"):
            bt = o.get("bioentity_type")
            pref = TYPE_PREFIX.get(bt)
            eid = o.get("entity_id")
            if pref and eid:
                w.writerow([f"{pref}::{eid}", (o.get("label") or "").replace("\t", " "), bt])
                n += 1
    print(f"wrote {n:,} labels -> {OUT}")


if __name__ == "__main__":
    main()
