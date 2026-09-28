"""
02_analyze_edges.py
-------------------
Streaming analysis of the PKT edges.json (inside dataset/PKT/edges.zip, ~4.6 GB).

Uses the uri->bioentity_type lookup built by 01_analyze_nodes.py to classify each
edge's endpoints, then aggregates the KG *schema*:

    (source_type) --[predicate_label]--> (target_type)   with counts.

Special focus (Step 0 = feasibility of the TARGET relation for drug repurposing):
find every relation connecting `chemical` to `gene`/`protein` (either direction),
and for each such relation count the DISTINCT chemicals and DISTINCT targets involved
(coverage), so we can pick the right TARGET predicate.

Outputs (analysis/out/):
  02_edges_schema.csv        full (src_type, predicate, tgt_type) counts
  02_predicate_totals.csv    predicate_label totals (any endpoint type)
  02_target_candidates.md    chemical<->gene/protein relations + coverage
  02_edges_summary.md        human-readable overview
"""
import ijson
import zipfile
import pickle
import csv
from pathlib import Path
from collections import Counter, defaultdict
from pkt_source import ensure_pkt_files   # downloads dataset/PKT/*.zip from Hugging Face if missing

PKT_DIR = Path(__file__).resolve().parents[1] / "dataset" / "PKT"
OUT_DIR = Path(__file__).resolve().parent / "out"

EDGES_ZIP = PKT_DIR / "edges.zip"
EDGES_JSON_NAME = "edges.json"
LOOKUP_PKL = OUT_DIR / "uri_to_type.pkl"

TARGET_TYPES = {"gene", "protein"}  # what a drug can "target"


def main():
    ensure_pkt_files(('edges.zip',))
    print("Loading uri->type lookup ...", flush=True)
    with open(LOOKUP_PKL, "rb") as fh:
        uri_to_type = pickle.load(fh)
    print(f"  lookup size: {len(uri_to_type):,}", flush=True)

    schema = Counter()              # (src_type, predicate, tgt_type) -> count
    predicate_totals = Counter()    # predicate -> count
    unresolved = 0                  # edges with an endpoint not in lookup

    # coverage for chemical<->gene/protein relations
    # key = (direction, predicate, other_type) -> {"chem": set, "tgt": set, "n": int}
    cand = defaultdict(lambda: {"chem": set(), "tgt": set(), "n": 0})

    n = 0
    with zipfile.ZipFile(EDGES_ZIP) as z:
        with z.open(EDGES_JSON_NAME) as f:
            for e in ijson.items(f, "item"):
                n += 1
                s_uri = e.get("source_uri")
                t_uri = e.get("target_uri")
                pred = e.get("predicate_label") or "<none>"

                s_type = uri_to_type.get(s_uri, "<unresolved>")
                t_type = uri_to_type.get(t_uri, "<unresolved>")
                if s_type == "<unresolved>" or t_type == "<unresolved>":
                    unresolved += 1

                schema[(s_type, pred, t_type)] += 1
                predicate_totals[pred] += 1

                # TARGET candidates: chemical on one side, gene/protein on the other
                if s_type == "chemical" and t_type in TARGET_TYPES:
                    d = cand[("chem->tgt", pred, t_type)]
                    d["chem"].add(s_uri); d["tgt"].add(t_uri); d["n"] += 1
                elif t_type == "chemical" and s_type in TARGET_TYPES:
                    d = cand[("tgt->chem", pred, s_type)]
                    d["chem"].add(t_uri); d["tgt"].add(s_uri); d["n"] += 1

                if n % 2_000_000 == 0:
                    print(f"  ...{n:,} edges processed", flush=True)

    print(f"Total edges: {n:,}")
    print(f"Edges with an unresolved endpoint: {unresolved:,} ({100*unresolved/n:.2f}%)")

    # --- full schema CSV ---
    with open(OUT_DIR / "02_edges_schema.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_type", "predicate_label", "target_type", "count"])
        for (s, p, t), c in schema.most_common():
            w.writerow([s, p, t, c])

    # --- predicate totals CSV ---
    with open(OUT_DIR / "02_predicate_totals.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["predicate_label", "count"])
        for p, c in predicate_totals.most_common():
            w.writerow([p, c])

    # --- TARGET candidates markdown ---
    with open(OUT_DIR / "02_target_candidates.md", "w", encoding="utf-8") as fh:
        fh.write("# TARGET candidate relations (chemical <-> gene/protein)\n\n")
        fh.write("Coverage = distinct chemicals / distinct targets touched by the relation.\n\n")
        fh.write("| direction | predicate_label | target_type | edges | distinct_chemicals | distinct_targets |\n")
        fh.write("|---|---|---|---:|---:|---:|\n")
        rows = sorted(cand.items(), key=lambda kv: kv[1]["n"], reverse=True)
        for (direction, pred, other), d in rows:
            fh.write(f"| {direction} | {pred} | {other} | {d['n']:,} | "
                     f"{len(d['chem']):,} | {len(d['tgt']):,} |\n")
        if not rows:
            fh.write("| _(none found)_ | | | | | |\n")

    # --- overview markdown ---
    with open(OUT_DIR / "02_edges_summary.md", "w", encoding="utf-8") as fh:
        fh.write(f"# PKT edges.json — analysis\n\n")
        fh.write(f"**Total edges:** {n:,}\n\n")
        fh.write(f"**Edges with unresolved endpoint:** {unresolved:,} ({100*unresolved/n:.2f}%)\n\n")
        fh.write("## Top 40 predicate_label (all endpoint types)\n\n")
        fh.write("| predicate_label | count | % |\n|---|---:|---:|\n")
        for p, c in predicate_totals.most_common(40):
            fh.write(f"| {p} | {c:,} | {100*c/n:.2f}% |\n")
        fh.write("\n## Top 60 schema triples (src_type, predicate, tgt_type)\n\n")
        fh.write("| source_type | predicate_label | target_type | count |\n|---|---|---|---:|\n")
        for (s, p, t), c in schema.most_common(60):
            fh.write(f"| {s} | {p} | {t} | {c:,} |\n")

    print("Saved: 02_edges_schema.csv, 02_predicate_totals.csv, "
          "02_target_candidates.md, 02_edges_summary.md")


if __name__ == "__main__":
    main()
