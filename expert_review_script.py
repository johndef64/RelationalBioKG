import os
import sys
import glob
import json
import time
import subprocess
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments"))
from interpret_predictions import load_subgraph, annotate   # KG-evidence interpretability

"""
expert_review_script.py — PathogenKG-style biological review driver, adapted to PKT.

Runs the compound-centric ranking (drug_eval.py) for a cohort of drugs with the mature model,
attaches the KG-evidence interpretability (interpret_predictions), joins human-readable node
labels, and writes a REVIEW SHEET with empty columns for an expert to fill in the 3 tiers.

Unlike drug_eval_script.py (which hard-codes ~9 case-study compounds), this works WITHOUT a
curated list: leave CANDIDATES empty and it ranks ALL compounds (`--compound all`); set
CANDIDATES to restrict the review to a hand-picked cohort. Either way the output is the same
review sheet, ready for human evaluation.

--- HOW THE OUTPUT IS CAPTURED FOR HUMAN EVALUATION ---
The protocol follows PathogenKG (a pre-declared cohort, the model ranks every candidate of the
target type, an expert rates the top-20 per compound on three tiers) with two additions that
PathogenKG did not have and that this setting needs, because here the reviewer is also the author
of the model.

BLINDING. Two files are written instead of one:
    ..._BLIND.csv   what the reviewer opens: item_id, the two labels, three empty columns.
                    Nothing about the model: no rank, no confidence, no auto tier, no KG evidence.
                    Rows are shuffled, so position carries no signal either.
    ..._KEY.csv     everything that was hidden, joined back by item_id at aggregation time.
Without this, "agreement between expert and auto tier" measures whether the reviewer agrees with
a label they were shown before deciding, which is not a measurement.

DECOYS. The blind sheet also contains control items at a proportion recorded only in the key:
    decoy_random    a candidate of the right type drawn at random from the pool (excluding the
                    compound's known partners) -- the model never proposed it
    decoy_midrank   drawn from ranks 400-1000 of the same compound -- the model ranked it, low
A plausibility rate on the top-k means nothing on its own; it means something against the rate the
same reviewer gives to decoys. That difference is the number to report.

The reviewer fills, for every row of the BLIND sheet:
    expert_tier       High / Moderate / Low   (or 1 / 2 / 3; both are accepted)
                      High     = direct mechanistic or experimental support in the literature
                      Moderate = indirect but defensible biological connection
                      Low      = no clear link identified
    expert_plausible  y / n
    expert_notes      free text: mechanism, reference, ...
Then:
    python expert_review_script.py aggregate <filled_BLIND.csv> <matching_KEY.csv>
"""

# ----------------------- CONFIG -----------------------
TASK          = "A"                                              # "A" = DTI (targets), "B" = TREATS (diseases)
MODEL_FOLDER  = os.path.join("models", "REPLACE_WITH_MODEL_FOLDER")
TOPK          = 50          # how deep drug_eval ranks and stores
REVIEW_TOPK   = 20          # how many per compound actually go to the reviewer (PathogenKG: 20)

# leave empty -> rank ALL compounds; or pin a curated cohort (entity ids), e.g.:
# CANDIDATES = ["Compound::CHEBI_28918", "Compound::CHEBI_45783", ...]
# Declare the cohort BEFORE looking at any output, as in PathogenKG (9 compounds with known and
# diverse mechanisms of action). Reviewing every compound is not a review, it is a spreadsheet.
CANDIDATES = []

# --- review protocol ---
BLIND            = True     # False reproduces the old single-sheet behaviour (not recommended)
DECOY_RANDOM     = 5        # random candidates of the right type, per compound
DECOY_MIDRANK    = 5        # candidates drawn from ranks MIDRANK_FROM..MIDRANK_TO, per compound
MIDRANK_FROM     = 400
MIDRANK_TO       = 1000
DECOY_SEED       = 20260918 # fixed so the sheet is reproducible; the reviewer never sees it

# per-task wiring (matches experiments/config.sh)
_CFG = {
    "A": dict(tsv="dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip", task_rel="DTI",    target_type="Protein"),
    "B": dict(tsv="dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip", task_rel="TREATS", target_type="Disease"),
}
LABELS_TSV = "dataset/PKT_subgraphs/node_labels.tsv"
# ------------------------------------------------------


def run_drug_eval(cfg):
    """Run drug_eval.py for the cohort (all compounds, or one call per curated compound)."""
    common = ["--model_folder", MODEL_FOLDER, "--tsv", cfg["tsv"], "--task", cfg["task_rel"],
              "--target_type", cfg["target_type"], "--topk", str(TOPK)]
    if not CANDIDATES:
        print("[i] No CANDIDATES set -> ranking ALL compounds (--compound all)")
        subprocess.run([sys.executable, "drug_eval.py", *common, "--compound", "all"], check=True)
    else:
        print(f"[i] Ranking {len(CANDIDATES)} curated compounds")
        for i, c in enumerate(CANDIDATES, 1):
            print(f"  [{i}/{len(CANDIDATES)}] {c}")
            r = subprocess.run([sys.executable, "drug_eval.py", *common, "--compound", c])
            if r.returncode != 0:
                print(f"    [WARN] drug_eval exited {r.returncode} for {c}")


def load_all_rankings():
    """Merge every rankings JSON in the model's drug_eval_results/ (newest wins per drug)."""
    files = sorted(glob.glob(os.path.join(MODEL_FOLDER, "drug_eval_results", "*rankings*.json")),
                   key=os.path.getmtime)
    if not files:
        raise SystemExit("No rankings files found — did drug_eval run?")
    merged = {}
    for fp in files:
        with open(fp) as f:
            merged.update(json.load(f))
    print(f"[i] merged {len(merged)} compounds from {len(files)} rankings file(s)")
    return merged


def _decoy_pool(df, target_type):
    """Every node of the target type in the task graph: the pool random decoys are drawn from."""
    nodes = set(df.loc[df["head"].str.startswith(target_type + "::"), "head"]) | \
            set(df.loc[df["tail"].str.startswith(target_type + "::"), "tail"])
    return sorted(nodes)


def _build_items(rankings, ann, df, cfg):
    """Rows to review: the top-REVIEW_TOPK novel predictions per compound, plus decoys.

    Each row carries a `stratum` that stays in the KEY file, never in the blind sheet.
    """
    import random
    rng = random.Random(DECOY_SEED)
    pool = _decoy_pool(df, cfg["target_type"])
    items = []

    for drug, sub in ann.groupby("drug", sort=True):
        sub = sub.sort_values("rank").head(REVIEW_TOPK)
        for _, r in sub.iterrows():
            items.append(dict(drug_id=drug, prediction_id=r["prediction"], stratum="topk",
                              rank=r["rank"], confidence=r.get("confidence"),
                              kg_evidence=r.get("kg_evidence"), n_evidence=r.get("n_evidence"),
                              auto_tier=r.get("auto_tier"),
                              held_out_recovered=r.get("held_out_recovered")))

        preds = rankings.get(drug, [])
        seen = {r["prediction"] for _, r in sub.iterrows()}

        # decoys the model ranked, but low
        mid = [p for p in preds if MIDRANK_FROM <= int(p.get("rank", 0)) <= MIDRANK_TO
               and not p.get("is_known_positive") and p["tail"] not in seen]
        for p in rng.sample(mid, min(DECOY_MIDRANK, len(mid))):
            seen.add(p["tail"])
            items.append(dict(drug_id=drug, prediction_id=p["tail"], stratum="decoy_midrank",
                              rank=p.get("rank"), confidence=p.get("confidence"),
                              kg_evidence="", n_evidence=0, auto_tier="", held_out_recovered=False))

        # decoys the model never proposed
        known = {p["tail"] for p in preds if p.get("is_known_positive")}
        cand = [n for n in pool if n not in known and n not in seen]
        for n in rng.sample(cand, min(DECOY_RANDOM, len(cand))):
            seen.add(n)
            items.append(dict(drug_id=drug, prediction_id=n, stratum="decoy_random",
                              rank="", confidence="", kg_evidence="", n_evidence=0,
                              auto_tier="", held_out_recovered=False))

    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["item_id"] = f"IT{i:05d}"
    return pd.DataFrame(items)


def build_review_sheet(cfg):
    rankings = load_all_rankings()
    if CANDIDATES:
        rankings = {d: p for d, p in rankings.items() if d in set(CANDIDATES)}
    else:
        print("[!] CANDIDATES is empty: every compound goes to review. Declare a cohort first "
              "(PathogenKG used 9 compounds with known, diverse mechanisms).")

    print(f"[i] loading subgraph {cfg['tsv']} for KG-evidence interpretability ...")
    df = load_subgraph(cfg["tsv"])
    ann, heldout = annotate(rankings, df, TASK, TOPK)      # drug/rank/prediction/kg_evidence/auto_tier/...

    labels = {}
    if os.path.exists(LABELS_TSV):
        lab = pd.read_csv(LABELS_TSV, sep="\t", dtype=str)
        labels = dict(zip(lab["entity"], lab["label"]))

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(MODEL_FOLDER, "drug_eval_results")

    if not BLIND:
        ann["drug_label"] = ann["drug"].map(labels).fillna("")
        ann["prediction_label"] = ann["prediction"].map(labels).fillna("")
        sheet = ann.rename(columns={"drug": "drug_id", "prediction": "prediction_id"})
        sheet = sheet[["drug_id", "drug_label", "rank", "prediction_id", "prediction_label",
                       "confidence", "kg_evidence", "n_evidence", "auto_tier", "held_out_recovered"]]
        for c in ("expert_tier", "expert_plausible", "expert_notes"):
            sheet[c] = ""
        out = os.path.join(outdir, f"expert_review_task{TASK}_{ts}.csv")
        sheet.sort_values(["drug_id", "rank"]).to_csv(out, index=False)
        print(f"\n[i] UNBLINDED SHEET -> {out}  ({len(sheet)} rows)")
        return out

    items = _build_items(rankings, ann, df, cfg)
    items["drug_label"] = items["drug_id"].map(labels).fillna("")
    items["prediction_label"] = items["prediction_id"].map(labels).fillna("")

    blind = items[["item_id", "drug_label", "prediction_label"]].copy()
    for c in ("expert_tier", "expert_plausible", "expert_notes"):
        blind[c] = ""
    blind_path = os.path.join(outdir, f"expert_review_task{TASK}_{ts}_BLIND.csv")
    blind.to_csv(blind_path, index=False)

    key = items[["item_id", "drug_id", "prediction_id", "stratum", "rank", "confidence",
                 "kg_evidence", "n_evidence", "auto_tier", "held_out_recovered"]]
    key_path = os.path.join(outdir, f"expert_review_task{TASK}_{ts}_KEY.csv")
    key.to_csv(key_path, index=False)

    counts = items["stratum"].value_counts().to_dict()
    print(f"\n[i] BLIND SHEET (give this to the reviewer) -> {blind_path}")
    print(f"[i] KEY          (do NOT open before rating)  -> {key_path}")
    print(f"[i] {len(items)} rows | strata {counts} | compounds {items['drug_id'].nunique()} | "
          f"held-out recovered in top-k: {heldout}")
    print("[i] Rate expert_tier (High/Moderate/Low), expert_plausible (y/n), expert_notes. Then:")
    print(f"[i]   python expert_review_script.py aggregate {os.path.basename(blind_path)} "
          f"{os.path.basename(key_path)}")
    return blind_path


_TIER = {"high": 1, "moderate": 2, "low": 3, "1": 1, "2": 2, "3": 3}


def _tier_num(v):
    return _TIER.get(str(v).strip().lower(), None)


def aggregate(filled_csv, key_csv=None):
    df = pd.read_csv(filled_csv, dtype=str)

    if key_csv is None:
        if "auto_tier" not in df.columns:
            raise SystemExit("Blind sheet given without its KEY file. Usage: aggregate <BLIND.csv> <KEY.csv>")
        merged = df                                   # legacy unblinded sheet
    else:
        key = pd.read_csv(key_csv, dtype=str)
        merged = df.merge(key, on="item_id", how="left", validate="one_to_one")

    merged["tier"] = merged["expert_tier"].map(_tier_num)
    rated = merged.dropna(subset=["tier"])
    print(f"[aggregate] {filled_csv}")
    print(f"  rated: {len(rated)}/{len(merged)}")
    if rated.empty:
        return

    if "stratum" not in rated.columns:
        rated = rated.assign(stratum="topk")

    print("\n  --- by stratum (this is the result to report) ---")
    order = ["topk", "decoy_midrank", "decoy_random"]
    rows = []
    for st in [o for o in order if o in set(rated["stratum"])]:
        g = rated[rated["stratum"] == st]
        pl = g["expert_plausible"].fillna("").str.lower().str.startswith("y").mean()
        dist = g["tier"].astype(int).value_counts().sort_index().to_dict()
        supported = (g["tier"] <= 2).mean()
        rows.append((st, len(g), pl, supported, dist))
        print(f"    {st:14s} n={len(g):4d} | plausible {pl:6.1%} | High+Moderate {supported:6.1%} | tiers {dist}")

    top = next((r for r in rows if r[0] == "topk"), None)
    for r in rows:
        if top and r[0] != "topk":
            print(f"    top-k minus {r[0]}: {top[2] - r[2]:+.1%} plausible, "
                  f"{top[3] - r[3]:+.1%} High+Moderate")

    if "held_out_recovered" in rated.columns:
        hk = rated[(rated["stratum"] == "topk") &
                   (rated["held_out_recovered"].astype(str).str.lower() == "true")]
        if len(hk):
            print(f"\n  held-out positives recovered inside the reviewed top-k: {len(hk)}")
            print(f"    of these, rated High or Moderate: {(hk['tier'] <= 2).mean():.1%}  "
                  "(non-circular evidence: they were removed from training)")

    if "auto_tier" in rated.columns:
        a = rated.dropna(subset=["auto_tier"])
        a = a[a["auto_tier"].astype(str).str.strip() != ""]
        if len(a):
            agree = (a["tier"].astype(int) == pd.to_numeric(a["auto_tier"]).astype(int)).mean()
            print(f"\n  auto-vs-expert tier agreement: {agree:.1%} over {len(a)} items "
                  + ("(blinded, so this is a measurement)" if key_csv else
                     "(UNBLINDED sheet: the reviewer saw auto_tier, so this is not a measurement)"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "aggregate":
        aggregate(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None); sys.exit(0)

    cfg = _CFG[TASK]
    print(f"[i] Task {TASK} | model {MODEL_FOLDER} | target_type {cfg['target_type']} | topk {TOPK}")
    run_drug_eval(cfg)
    build_review_sheet(cfg)
