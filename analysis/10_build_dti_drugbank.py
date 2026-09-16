"""
10_build_dti_drugbank.py — build a genuine drug--target relation for Task A (see TICKET_01).

Why: the Task A target relation inherited from PheKnowLator (`molecularly interacts with`,
chemical -> protein) is biochemical, not pharmacological: its hub compounds are hydron, water, ATP
and magnesium, and filtering it by ChEBI role does not recover drugs (analysis/out/09_*). Drug--target
edges must therefore come from outside the KG.

Source of the mapping: **UniProt**, not DrugBank. UniProt is open and publishes, for each protein,
its cross-references to DrugBank; inverting those cross-references yields drug -> protein pairs
without scraping a closed resource. DrugBank's own CSV is used only to turn DrugBank accessions into
drug names and status (approved / investigational / withdrawn), which is public metadata.

Pipeline:
  1. download human reviewed proteins with a DrugBank cross-reference (UniProt REST, one streamed
     request, cached on disk);
  2. invert into DrugBank id -> {UniProt accession};
  3. map DrugBank id -> drug name (DrugBank CSV) -> ChEBI compound of the KG, by normalised label;
  4. map UniProt accession -> Protein Ontology node of the KG (`Protein::PR_<accession>`);
  5. emit the edge list and a coverage report.

Outputs:
  dataset/DRUGBANK/uniprot_human_drugbank.tsv     raw UniProt response (cached; delete to refresh)
  dataset/PKT_subgraphs/dti_drugbank_edges.tsv    head <TAB> interaction <TAB> tail <TAB> source <TAB> type
  analysis/out/10_dti_drugbank_report.md          coverage at every step, and what is lost where

Usage (conda env gnn, from the repo root):
  python analysis/10_build_dti_drugbank.py                  # all drugs with targets
  python analysis/10_build_dti_drugbank.py --approved-only  # only approved drugs
"""
import argparse
import collections
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DRUGBANK_ZIP = ROOT / "dataset" / "DRUGBANK" / "drug-bank-5110.zip"
UNIPROT_TSV = ROOT / "dataset" / "DRUGBANK" / "uniprot_human_drugbank.tsv"
NODE_LABELS = ROOT / "dataset" / "PKT_subgraphs" / "node_labels.tsv"
TASK_A = ROOT / "dataset" / "PKT_subgraphs" / "pkt_taskA_dti.tsv.zip"
OUT_EDGES = ROOT / "dataset" / "PKT_subgraphs" / "dti_drugbank_edges.tsv"
OUT_REPORT = ROOT / "analysis" / "out" / "10_dti_drugbank_report.md"
PKT_DIR = ROOT / "dataset" / "PKT"                      # raw KG (fallback GO source)
GO_CACHE = ROOT / "dataset" / "PKT_subgraphs" / "protein_go_terms.tsv"   # derived, gitignored

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_QUERY = "(organism_id:9606) AND (reviewed:true) AND (database:drugbank)"
UNIPROT_FIELDS = "accession,id,protein_name,xref_drugbank"

csv.field_size_limit(10 ** 9)


def norm_name(s):
    """Normalise a compound name for matching: case, charge state, stereo prefix, punctuation."""
    s = (s or "").strip().lower()
    s = re.sub(r"\(\d*[+-]\)$", "", s).strip()
    s = re.sub(r"^(trans-|cis-|\(\+\)-|\(-\)-|\(r\)-|\(s\)-|l-|d-|dl-)", "", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s.strip()


def fetch_uniprot(force=False):
    """Download (once) the human reviewed proteins carrying DrugBank cross-references.

    Uses the paginated /search endpoint rather than /stream: the streamed response is a single long
    chunked transfer and was truncated ("Response ended prematurely") on this connection.
    """
    if UNIPROT_TSV.exists() and not force:
        print(f"[1/5] using cached {UNIPROT_TSV.relative_to(ROOT)}")
        return UNIPROT_TSV.read_text(encoding="utf-8")

    print("[1/5] downloading from UniProt (paginated) ...")
    session = requests.Session()
    session.headers["User-Agent"] = "RelationalPKT/1.0 (academic research)"
    retry = requests.adapters.Retry(total=5, backoff_factor=1.0,
                                    status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry))

    url = UNIPROT_SEARCH
    params = {"query": UNIPROT_QUERY, "fields": UNIPROT_FIELDS, "format": "tsv", "size": 500}
    parts, header, pages = [], None, 0
    while url:
        r = session.get(url, params=params, timeout=120)
        r.raise_for_status()
        lines = r.text.splitlines()
        if not lines:
            break
        if header is None:
            header = lines[0]
            parts.append(header)
        parts.extend(lines[1:])
        pages += 1
        print(f"      page {pages}: {len(lines) - 1} rows (total {len(parts) - 1})")
        url = r.links.get("next", {}).get("url")   # UniProt paginates through the Link header
        params = None                              # the next URL already carries the cursor

    text = "\n".join(parts) + "\n"
    UNIPROT_TSV.parent.mkdir(parents=True, exist_ok=True)
    UNIPROT_TSV.write_text(text, encoding="utf-8")
    print(f"      saved {len(text):,} bytes -> {UNIPROT_TSV.relative_to(ROOT)}")
    return text


def parse_uniprot(text):
    """DrugBank accession -> {UniProt accession}, plus accession -> protein name."""
    drug2prot = collections.defaultdict(set)
    prot_name = {}
    rows = 0
    for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
        acc = row.get("Entry") or row.get("accession")
        xref = row.get("DrugBank") or row.get("xref_drugbank") or ""
        if not acc:
            continue
        rows += 1
        prot_name[acc] = row.get("Protein names", "")
        for db in re.findall(r"DB\d{5}", xref):
            drug2prot[db].add(acc)
    return drug2prot, prot_name, rows


def drugbank_metadata():
    """DrugBank accession -> (name, groups, n_targets, n_adme).

    The release lists the proteins of each drug in four separate columns (targets, enzymes,
    transporters, carriers) as internal BE identifiers. The identifiers themselves cannot be mapped
    to UniProt without scraping, but their COUNTS say how many proteins of each role a drug has, and
    that is enough to settle the drugs where the roles are unambiguous (see `classify`).
    """
    meta = {}
    with zipfile.ZipFile(DRUGBANK_ZIP) as z:
        with z.open("drugbank_clean.csv") as fh:
            for d in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="ignore")):
                db_id, name = (d.get("drugbank-id") or "").strip(), (d.get("name") or "").strip()
                if db_id.startswith("DB") and name:
                    n_t = len((d.get("targets") or "").split())
                    n_a = sum(len((d.get(k) or "").split())
                              for k in ("enzymes", "transporters", "carriers"))
                    meta[db_id] = (name, (d.get("groups") or "").strip(), n_t, n_a)
    return meta


def kg_nodes():
    """(normalised compound name -> [ChEBI node]), set of Protein Ontology nodes."""
    compounds, proteins = collections.defaultdict(list), set()
    with open(NODE_LABELS, encoding="utf-8") as fh:
        for x in csv.DictReader(fh, delimiter="\t"):
            if x["bioentity_type"] == "chemical" and x["label"]:
                compounds[norm_name(x["label"])].append(x["entity"])
            elif x["bioentity_type"] == "protein":
                proteins.add(x["entity"])
    return compounds, proteins


# --- pharmacodynamics vs pharmacokinetics -----------------------------------------------------
# UniProt's DrugBank cross-references do not say in which role a protein is linked to a drug:
# DrugBank lists targets, but also metabolising enzymes, transporters and plasma carriers. Those
# are pharmacokinetics (ADME), not therapeutic targets, and must not be the prediction target of
# Task A. We separate them with the KG's own Gene Ontology annotations: a protein annotated for
# xenobiotic metabolism, efflux transport or conjugation is ADME. The rule keeps aromatase (a
# cytochrome P450 that IS a drug target) on the target side, because it carries no xenobiotic terms.
ADME_GO = re.compile(r"xenobiotic|drug metabolic|drug catabolic|drug transmembrane|"
                     r"efflux transmembrane transporter|abc-type|monooxygenase activity|"
                     r"glucuronosyltransferase|sulfotransferase activity|"
                     r"organic anion transmembrane|organic cation transmembrane")
# plasma carriers: they bind drugs without metabolising or transporting them, so GO does not flag them
CARRIER_LABEL = re.compile(r"^(albumin|serum albumin|alpha-1-acid glycoprotein|transthyretin|"
                           r"sex hormone-binding globulin|alpha-1-antitrypsin)", re.I)


def protein_go_terms(labels):
    """Protein node -> {GO term label}.

    Read from the built Task A graph when it exists (fast). This script, however, also runs BEFORE
    the graphs are built (they depend on its output), so the fallback streams the raw KG once and
    caches the result in analysis/out/, making the two scripts independent of each other.
    """
    go_of = collections.defaultdict(set)
    if TASK_A.exists():
        with zipfile.ZipFile(TASK_A) as z:
            with z.open(z.namelist()[0]) as fh:
                r = csv.reader(io.TextIOWrapper(fh, encoding="utf-8"), delimiter="\t")
                next(r)
                for row in r:
                    if row[1].startswith("PROTEIN_GO"):
                        go_of[row[0]].add(labels.get(row[2], "").lower())
        return go_of

    if GO_CACHE.exists():
        print(f"      protein--GO annotations from cache {GO_CACHE.relative_to(ROOT)}")
        with open(GO_CACHE, encoding="utf-8") as fh:
            for line in fh:
                node, terms = line.rstrip("\n").split("\t", 1)
                go_of[node] = set(terms.split("; ")) if terms else set()
        return go_of

    print("      task graph not built yet: reading protein--GO annotations from the raw KG "
          "(one pass, a few minutes) ...")
    import ijson                                     # only needed on this path
    uri2node = {}
    with zipfile.ZipFile(PKT_DIR / "nodes.zip") as z, z.open("nodes.json") as f:
        for o in ijson.items(f, "item"):
            bt = o.get("bioentity_type")
            if bt in ("protein", "go") and o.get("entity_id"):
                prefix = "Protein" if bt == "protein" else "GO"
                uri2node[o["uri"]] = (f"{prefix}::{o['entity_id']}", bt, (o.get("label") or "").lower())
    GO_PREDICATES = {"participates in", "has function", "located_in"}
    with zipfile.ZipFile(PKT_DIR / "edges.zip") as z, z.open("edges.json") as f:
        for e in ijson.items(f, "item"):
            if e.get("predicate_label") not in GO_PREDICATES:
                continue
            s = uri2node.get(e.get("source_uri")); t = uri2node.get(e.get("target_uri"))
            if s and t and s[1] == "protein" and t[1] == "go":
                go_of[s[0]].add(t[2])
    GO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(GO_CACHE, "w", encoding="utf-8") as fh:
        for node, terms in go_of.items():
            fh.write(f"{node}\t{'; '.join(sorted(terms))}\n")
    print(f"      cached {len(go_of):,} proteins -> {GO_CACHE.relative_to(ROOT)}")
    return go_of


def adme_proteins(labels):
    """Proteins that DrugBank would list as enzymes / transporters / carriers rather than targets."""
    go_of = protein_go_terms(labels)
    flagged = {p for p, terms in go_of.items() if any(ADME_GO.search(t) for t in terms)}
    flagged |= {p for p in labels if p.startswith("Protein::") and CARRIER_LABEL.search(labels[p])}
    return flagged


def node_labels():
    with open(NODE_LABELS, encoding="utf-8") as fh:
        return {x["entity"]: x["label"] for x in csv.DictReader(fh, delimiter="\t")}


def existing_dti():
    """Compound--protein pairs asserted by PheKnowLator itself (biochemical relation).

    Only used for a statistic in the report; the graph may not be built yet, in which case the
    overlap is simply reported as unavailable.
    """
    pairs = set()
    if not TASK_A.exists():
        return pairs
    with zipfile.ZipFile(TASK_A) as z:
        with z.open(z.namelist()[0]) as fh:
            r = csv.reader(io.TextIOWrapper(fh, encoding="utf-8"), delimiter="\t")
            next(r)
            for row in r:
                if row[1] in ("CPI_BIOCHEM", "DTI_LEGACY"):
                    pairs.add((row[0], row[2]))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approved-only", action="store_true", help="keep only drugs in the 'approved' group")
    ap.add_argument("--refresh", action="store_true", help="re-download the UniProt table")
    ap.add_argument("--relation", default="DTI", help="relation for pharmacodynamic targets")
    ap.add_argument("--adme-relation", default="DRUG_ADME",
                    help="relation for metabolising enzymes, transporters and plasma carriers")
    ap.add_argument("--no-adme-split", action="store_true",
                    help="emit every edge under --relation, without separating ADME proteins")
    args = ap.parse_args()

    drug2prot, prot_name, n_prot = parse_uniprot(fetch_uniprot(force=args.refresh))
    print(f"[2/5] UniProt: {n_prot:,} human reviewed proteins with a DrugBank cross-reference; "
          f"{len(drug2prot):,} distinct DrugBank accessions")

    meta = drugbank_metadata()
    print(f"[3/5] DrugBank CSV: {len(meta):,} named drugs")

    compounds, proteins = kg_nodes()
    print(f"[4/5] KG: {sum(len(v) for v in compounds.values()):,} chemicals, {len(proteins):,} proteins")

    labels = node_labels()
    adme_set = set() if args.no_adme_split else adme_proteins(labels)
    print(f"      ADME-annotated proteins (GO xenobiotic/transport + plasma carriers): {len(adme_set):,}")

    print("[5/5] joining ...")
    edges, seen = [], set()
    unmapped_drug, unmapped_prot = [], set()
    kept_drugs, skipped_status = set(), 0
    rule_counts = collections.Counter()
    for db_id, accs in drug2prot.items():
        name, groups, n_t, n_a = meta.get(db_id, ("", "", 0, 0))
        if not name:
            unmapped_drug.append((db_id, "not in the DrugBank CSV"))
            continue
        if args.approved_only and "approved" not in groups:
            skipped_status += 1
            continue
        hits = compounds.get(norm_name(name))
        if not hits:
            unmapped_drug.append((db_id, f"no ChEBI match for '{name}'"))
            continue
        head = hits[0]                      # a normalised name may map to several ChEBI forms
        for acc in accs:
            tail = f"Protein::PR_{acc}"
            if tail not in proteins:
                unmapped_prot.add(acc)
                continue
            key = (head, tail)
            if key in seen:
                continue
            seen.add(key)
            kept_drugs.add(head)
            # role assignment, in order of evidence strength:
            #  1. the drug declares proteins of only one role -> every edge takes that role;
            #  2. otherwise the protein's own GO annotation decides (ADME vs target).
            if args.no_adme_split:
                rel, rule = args.relation, "no-split"
            elif n_t == 0 and n_a > 0:
                rel, rule = args.adme_relation, "drugbank: no target declared"
            elif n_a == 0 and n_t > 0:
                rel, rule = args.relation, "drugbank: no ADME protein declared"
            elif tail in adme_set:
                rel, rule = args.adme_relation, "GO: xenobiotic/transport/carrier"
            else:
                rel, rule = args.relation, "GO: no ADME annotation"
            rule_counts[rule] += 1
            edges.append((head, rel, tail, "DrugBank/UniProt", "Compound-Protein"))

    OUT_EDGES.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_EDGES, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["head", "interaction", "tail", "source", "type"])
        w.writerows(edges)

    old = existing_dti()
    overlap = sum(1 for h, _, t, _, _ in edges if (h, t) in old)
    target_edges = [e for e in edges if e[1] == args.relation]
    adme_edges = [e for e in edges if e[1] == args.adme_relation]
    per_drug = collections.Counter(h for h, r, _, _, _ in edges if r == args.relation)
    per_prot = collections.Counter(t for _, r, t, _, _ in edges if r == args.relation)
    per_prot_adme = collections.Counter(t for _, r, t, _, _ in edges if r == args.adme_relation)
    label = {}
    with open(NODE_LABELS, encoding="utf-8") as fh:
        for x in csv.DictReader(fh, delimiter="\t"):
            label[x["entity"]] = x["label"]

    md = [f"# Task A drug--target relation from UniProt cross-references\n",
          f"Mode: {'approved drugs only' if args.approved_only else 'all drugs with a cross-reference'}. "
          f"Relation name: `{args.relation}`.\n",
          "\n## Pipeline coverage\n", "| step | value |", "|---|---:|",
          f"| Human reviewed proteins with a DrugBank cross-reference (UniProt) | {n_prot:,} |",
          f"| Distinct DrugBank accessions cross-referenced | {len(drug2prot):,} |",
          f"| Of these, named in the DrugBank CSV | {sum(1 for d in drug2prot if d in meta):,} |",
          f"| Drugs skipped as not approved | {skipped_status:,} |",
          f"| Drugs matched to a ChEBI compound of the KG | {len(kept_drugs):,} |",
          f"| Proteins dropped (accession absent from the KG) | {len(unmapped_prot):,} |",
          f"| **Edges emitted** | **{len(edges):,}** |",
          f"| — of which `{args.relation}` (pharmacodynamic targets) | **{len(target_edges):,}** "
          f"({100 * len(target_edges) / max(1, len(edges)):.1f}%) |",
          f"| — of which `{args.adme_relation}` (enzymes, transporters, carriers) | {len(adme_edges):,} "
          f"({100 * len(adme_edges) / max(1, len(edges)):.1f}%) |",
          f"| Distinct drugs / proteins in `{args.relation}` | {len(per_drug):,} / {len(per_prot):,} |",
          ] + [f"| rule — {rule} | {n:,} |" for rule, n in rule_counts.most_common()] + [
          (f"| Edges already asserted as `CPI_BIOCHEM` by PheKnowLator | {overlap:,} "
           f"({100 * overlap / max(1, len(edges)):.1f}%) |" if TASK_A.exists() else
           "| Edges already asserted as `CPI_BIOCHEM` by PheKnowLator | n/d (task graph not built yet) |"),
          "\n## Most connected drugs\n", "| targets | drug |", "|---:|---|"]
    for node, n in per_drug.most_common(20):
        md.append(f"| {n} | {label.get(node, node)} |")
    md += [f"\n## Most targeted proteins (`{args.relation}`)\n", "| drugs | protein |", "|---:|---|"]
    for node, n in per_prot.most_common(20):
        md.append(f"| {n} | {label.get(node, node)} |")
    md += [f"\n## Proteins moved to `{args.adme_relation}`\n", "| drugs | protein |", "|---:|---|"]
    for node, n in per_prot_adme.most_common(20):
        md.append(f"| {n} | {label.get(node, node)} |")
    md += ["\n## What is lost, and where\n",
           f"- {len(unmapped_drug):,} cross-referenced drugs could not be matched to a ChEBI compound "
           f"(name matching only: no synonyms, CAS or UNII). Examples: "
           + "; ".join(r for _, r in unmapped_drug[:5]) + ".",
           f"- {len(unmapped_prot):,} UniProt accessions are not present as Protein Ontology nodes.",
           "\nEdge list: `dataset/PKT_subgraphs/dti_drugbank_edges.tsv`.\n"]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md), encoding="utf-8")

    print(f"\nedges: {len(edges):,} | drugs: {len(per_drug):,} | proteins: {len(per_prot):,} | "
          f"already in the KG: {overlap:,}")
    print(f"written: {OUT_EDGES.relative_to(ROOT)}")
    print(f"written: {OUT_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
