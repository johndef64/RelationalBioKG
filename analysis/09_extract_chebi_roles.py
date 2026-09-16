"""
09_extract_chebi_roles.py — extract ChEBI `has_role` assertions from the PKT source graph and use
them to tell drugs from metabolites in the Task A target relation (see TICKET_01_DTI_drug_scope.md).

The Task A relation (chemical -> protein) comes from the OWL property `molecularly interacts with`,
which in PheKnowLator covers substrates, cofactors and products as well as drug binding: its ten
most connected compounds are hydron, water, ATP, ADP, phosphate and magnesium. This script gathers
the ontological role of every compound so that the target relation can be restricted to compounds
that ChEBI itself describes as drugs.

Reads (streamed, never fully loaded):
  dataset/PKT/nodes.zip   -> uri -> (label, entity_id, bioentity_type)
  dataset/PKT/edges.zip   -> edges with predicate_label == "has_role"

Writes:
  analysis/out/09_chebi_roles_summary.md   role inventory + coverage of the Task A compounds
  analysis/out/09_compound_roles.tsv       compound <TAB> entity_id <TAB> label <TAB> roles (;-joined)

Usage (conda env gnn, from the repo root):
  python analysis/09_extract_chebi_roles.py
"""
import csv
import io
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import ijson

ROOT = Path(__file__).resolve().parents[1]
NODES_ZIP = ROOT / "dataset" / "PKT" / "nodes.zip"
EDGES_ZIP = ROOT / "dataset" / "PKT" / "edges.zip"
TASK_A = ROOT / "dataset" / "PKT_subgraphs" / "pkt_taskA_dti.tsv.zip"
OUT_DIR = ROOT / "analysis" / "out"


def stream_json_array(zip_path):
    """Yield the objects of the single big JSON array inside `zip_path`."""
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            for obj in ijson.items(fh, "item"):
                yield obj


def task_a_compound_degrees():
    """Compound entity -> number of DTI edges in the Task A graph."""
    deg = Counter()
    with zipfile.ZipFile(TASK_A) as z:
        with z.open(z.namelist()[0]) as fh:
            r = csv.reader(io.TextIOWrapper(fh, encoding="utf-8"), delimiter="\t")
            next(r)
            for row in r:
                if row[1] == "DTI":
                    deg[row[0]] += 1
    return deg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] reading nodes ...")
    uri_info = {}                      # uri -> (entity_id, label, bioentity_type)
    for n in stream_json_array(NODES_ZIP):
        uri_info[n["uri"]] = (n.get("entity_id", ""), n.get("label", ""), n.get("bioentity_type", ""))
    print(f"      {len(uri_info):,} nodes")

    print("[2/3] scanning edges for has_role ...")
    roles_of = defaultdict(set)        # compound entity_id -> {role label}
    role_counts = Counter()
    seen = 0
    for e in stream_json_array(EDGES_ZIP):
        seen += 1
        if seen % 2_000_000 == 0:
            print(f"      {seen:,} edges scanned, {sum(len(v) for v in roles_of.values()):,} roles kept")
        if e.get("predicate_label") != "has_role":
            continue
        src = uri_info.get(e.get("source_uri"))
        tgt = uri_info.get(e.get("target_uri"))
        if not src or not tgt:
            continue
        role = tgt[1] or tgt[0]
        roles_of[src[0]].add(role)
        role_counts[role] += 1
    print(f"      {seen:,} edges scanned; {len(roles_of):,} compounds carry a role")

    print("[3/3] crossing with the Task A target relation ...")
    deg = task_a_compound_degrees()                      # "Compound::CHEBI_15377" -> n edges
    total_edges = sum(deg.values())
    rows = []
    for node, d in deg.items():
        ent = node.split("::", 1)[1] if "::" in node else node
        roles = sorted(roles_of.get(ent, ()))
        label = ""
        rows.append((node, ent, label, d, roles))

    # coverage of each role among Task A compounds and their edges
    per_role_compounds, per_role_edges = Counter(), Counter()
    for _, _, _, d, roles in rows:
        for r in roles:
            per_role_compounds[r] += 1
            per_role_edges[r] += d

    with open(OUT_DIR / "09_compound_roles.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["compound", "entity_id", "dti_edges", "n_roles", "roles"])
        for node, ent, _, d, roles in sorted(rows, key=lambda x: -x[3]):
            w.writerow([node, ent, d, len(roles), "; ".join(roles)])

    md = ["# ChEBI roles of the Task A compounds\n",
          f"Source: `dataset/PKT/edges.zip` (`has_role`), crossed with the DTI relation of "
          f"`pkt_taskA_dti.tsv.zip` ({total_edges:,} edges, {len(deg):,} compounds).\n",
          f"Compounds carrying at least one role in the whole KG: {len(roles_of):,}. "
          f"Distinct roles: {len(role_counts):,}.\n",
          "\n## Roles by coverage of the Task A target relation\n",
          "| role | compounds | DTI edges | % of edges |", "|---|---:|---:|---:|"]
    for role, n_c in per_role_compounds.most_common(60):
        e = per_role_edges[role]
        md.append(f"| {role} | {n_c} | {e:,} | {100 * e / total_edges:.1f} |")

    with_role = sum(1 for _, _, _, _, r in rows if r)
    edges_with_role = sum(d for _, _, _, d, r in rows if r)
    md += ["\n## Coverage\n",
           f"- Task A compounds with at least one ChEBI role: **{with_role:,} / {len(rows):,}** "
           f"({100 * with_role / max(1, len(rows)):.1f}%)",
           f"- Task A edges whose compound has at least one role: **{edges_with_role:,} / {total_edges:,}** "
           f"({100 * edges_with_role / max(1, total_edges):.1f}%)",
           "\n## Most connected compounds and their roles\n",
           "| DTI edges | compound | roles |", "|---:|---|---|"]
    for node, ent, _, d, roles in sorted(rows, key=lambda x: -x[3])[:40]:
        md.append(f"| {d:,} | {ent} | {'; '.join(roles) if roles else '—'} |")
    md.append("\nFull table: `analysis/out/09_compound_roles.tsv`.\n")

    (OUT_DIR / "09_chebi_roles_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"written: {OUT_DIR / '09_chebi_roles_summary.md'}")
    print(f"written: {OUT_DIR / '09_compound_roles.tsv'}")


if __name__ == "__main__":
    main()
