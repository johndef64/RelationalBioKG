"""
01_analyze_nodes.py
-------------------
Streaming analysis of the PheKnowLator (PKT) nodes.json (inside dataset/PKT/nodes.zip).

Goals:
  1. Count nodes by bioentity_type / class_code / namespace / source.
  2. Build a compact lookup  uri -> (bioentity_type, class_code)  used by the edges
     analysis (02) to classify each edge's endpoints. Saved as a pickle.

The file is a large (~483 MB) JSON array; we stream it with ijson (yajl2_c backend)
to keep memory low. Run inside the conda env `gnn`.
"""
import ijson
import zipfile
import pickle
from pathlib import Path
from collections import Counter
from pkt_source import ensure_pkt_files   # downloads dataset/PKT/*.zip from Hugging Face if missing

PKT_DIR = Path(__file__).resolve().parents[1] / "dataset" / "PKT"
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)

NODES_ZIP = PKT_DIR / "nodes.zip"
NODES_JSON_NAME = "nodes.json"

LOOKUP_PKL = OUT_DIR / "uri_to_type.pkl"
SUMMARY_MD = OUT_DIR / "01_nodes_summary.md"


def main():
    ensure_pkt_files(('nodes.zip',))
    by_type = Counter()
    by_class = Counter()
    by_namespace = Counter()
    by_source = Counter()
    # combined (bioentity_type, class_code) to understand the type system
    by_type_class = Counter()

    uri_to_type = {}          # uri -> bioentity_type
    n = 0

    with zipfile.ZipFile(NODES_ZIP) as z:
        with z.open(NODES_JSON_NAME) as f:
            for obj in ijson.items(f, "item"):
                n += 1
                bet = obj.get("bioentity_type") or "<none>"
                cc = obj.get("class_code") or "<none>"
                ns = obj.get("namespace") or "<none>"
                src = obj.get("source") or "<none>"
                uri = obj.get("uri")

                by_type[bet] += 1
                by_class[cc] += 1
                by_namespace[ns] += 1
                by_source[src] += 1
                by_type_class[(bet, cc)] += 1

                if uri is not None:
                    uri_to_type[uri] = bet

                if n % 500_000 == 0:
                    print(f"  ...{n:,} nodes processed", flush=True)

    print(f"Total nodes: {n:,}")
    print(f"Distinct URIs in lookup: {len(uri_to_type):,}")

    # --- save lookup ---
    with open(LOOKUP_PKL, "wb") as fh:
        pickle.dump(uri_to_type, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved lookup -> {LOOKUP_PKL}")

    # --- write markdown summary ---
    def fmt_counter(c, title, topn=None):
        lines = [f"\n### {title}\n", "| value | count | % |", "|---|---:|---:|"]
        items = c.most_common(topn)
        for k, v in items:
            lines.append(f"| {k} | {v:,} | {100*v/n:.2f}% |")
        return "\n".join(lines)

    with open(SUMMARY_MD, "w", encoding="utf-8") as fh:
        fh.write(f"# PKT nodes.json — analysis\n\n**Total nodes:** {n:,}\n")
        fh.write(fmt_counter(by_type, "By bioentity_type"))
        fh.write("\n")
        fh.write(fmt_counter(by_class, "By class_code"))
        fh.write("\n")
        fh.write(fmt_counter(by_source, "By source"))
        fh.write("\n")
        fh.write(fmt_counter(by_namespace, "By namespace", topn=40))
        fh.write("\n\n### By (bioentity_type, class_code)\n\n")
        fh.write("| bioentity_type | class_code | count |\n|---|---|---:|\n")
        for (bet, cc), v in by_type_class.most_common():
            fh.write(f"| {bet} | {cc} | {v:,} |\n")

    print(f"Saved summary -> {SUMMARY_MD}")


if __name__ == "__main__":
    main()
