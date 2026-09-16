#!/usr/bin/env python3
"""
Table for experiments/convergence_check.sh: did the models still improve when given a much larger
epoch budget, and were the GNNs limited by the learning-rate grid of the HPO?

Reads experiments/logs/v2/conv/conv_<TASK>_<model>_<tag>_<YYYYmmdd>_<HHMMSS>.log and reports, per
task, one row per (model, tag): test AUROC/AUPRC/MRR/Hits@10, M = 0.2 AUROC + 0.4 AUPRC + 0.4 MRR,
warm (cold-start-excluded) MRR, the epoch of the best checkpoint and the training time.

`best ep.` is flagged with ! when it sits at the epoch ceiling: that run was still improving when it
was stopped, so its numbers are a lower bound and the budget must be raised.

Output: <logdir>/convergence_summary.md  and  .csv
Usage:  python experiments/convergence_summary.py [--logdir experiments/logs/v2/conv] [--epochs_cap 1500]
"""
import argparse
import csv
import glob
import math
import os
import re

RUN_RE = re.compile(r"Run\s+(\d+)\s*\|\s*Test Auroc:\s*([\d.]+),\s*Test Auprc:\s*([\d.]+),\s*"
                    r"Test MRR:\s*([\d.]+),\s*TEST HITS:\s*(\{[^}]*\})")
WARM_RE = re.compile(r"Run\s+(\d+)\s*\|\s*WARM Test Auroc:\s*([\d.]+),\s*Test Auprc:\s*([\d.]+),\s*"
                     r"Test MRR:\s*([\d.]+),\s*TEST HITS:\s*(\{[^}]*\})")
TIME_RE = re.compile(r"Run\s+(\d+)\s*\|\s*best_epoch:\s*(\w+)\s*\|\s*train_time_sec:\s*([\d.]+)")
HITS_RE = re.compile(r"(\d+)\s*:\s*([\d.]+)")
NAME_RE = re.compile(r"^conv_([A-Za-z0-9,]+)_([a-z]+)_(.+)_(\d{8}_\d{6})\.log$")


def M(a, p, m):
    return 0.2 * a + 0.4 * p + 0.4 * m


def parse(path):
    """Return the metrics of the last complete run in the log (these are single-run checks)."""
    runs = {}

    def rec(i):
        return runs.setdefault(int(i), {})

    with open(path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            for line in raw.split("\r"):
                if (m := WARM_RE.search(line)):
                    a, p, mrr = float(m.group(2)), float(m.group(3)), float(m.group(4))
                    rec(m.group(1)).update(warm_MRR=mrr)
                elif (m := RUN_RE.search(line)):
                    a, p, mrr = float(m.group(2)), float(m.group(3)), float(m.group(4))
                    hits = {int(k): float(v) for k, v in HITS_RE.findall(m.group(5))}
                    rec(m.group(1)).update(AUROC=a, AUPRC=p, MRR=mrr, M=M(a, p, mrr),
                                           **{"Hits@10": hits.get(10, math.nan)})
                elif (m := TIME_RE.search(line)):
                    r = rec(m.group(1))
                    r["best_epoch"] = float(m.group(2)) if m.group(2).isdigit() else math.nan
                    r["train_time_sec"] = float(m.group(3))
    done = [runs[k] for k in sorted(runs) if "MRR" in runs[k]]
    return done[-1] if done else None


def num(v, fmt="{:.3f}"):
    return "—" if v is None or (isinstance(v, float) and v != v) else fmt.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="experiments/logs/v2/conv")
    ap.add_argument("--epochs_cap", type=float, default=None,
                    help="epoch ceiling used by the runs; best_epoch within 1%% of it is flagged")
    ap.add_argument("--out", default=None, help="output dir (default: --logdir)")
    args = ap.parse_args()
    out_dir = args.out or args.logdir

    data = {}
    for path in sorted(glob.glob(os.path.join(args.logdir, "conv_*.log"))):
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        key = (m.group(1), m.group(2), m.group(3))
        r = parse(path)
        if r:
            data[key] = (r, os.path.basename(path))   # later timestamp wins
    if not data:
        print(f"[conv-summary] no parseable logs in {args.logdir}")
        return

    cap = args.epochs_cap
    md = ["# Convergence check — epoch budget and learning-rate bounds\n",
          "One run per row (test set). M = 0.2·AUROC + 0.4·AUPRC + 0.4·MRR. *warm* = excluding "
          "cold-start test triples. `tuned` = the config chosen by the v2 HPO, unchanged; `lr…` = the "
          "same config with a learning rate outside the HPO grid.\n",
          "**!** next to *best ep.* = the best checkpoint is at the epoch ceiling"
          + (f" ({cap:.0f})" if cap else "") + ": that run was still improving, so its metrics are a "
          "lower bound — re-run with a larger budget before concluding anything.\n"]
    rows = []
    for task in sorted({k[0] for k in data}):
        md.append(f"\n## {task}\n")
        md.append("| model | variant | AUROC | AUPRC | MRR | Hits@10 | **M** | warm MRR | best ep. | time (s) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        keys = [k for k in data if k[0] == task]
        keys.sort(key=lambda k: (-data[k][0].get("M", 0),))
        for k in keys:
            r, log = data[k]
            ep = r.get("best_epoch")
            flag = " !" if (cap and ep == ep and ep is not None and ep >= 0.99 * cap) else ""
            md.append("| {} | {} | {} | {} | {} | {} | **{}** | {} | {}{} | {} |".format(
                k[1], k[2], num(r.get("AUROC")), num(r.get("AUPRC")), num(r.get("MRR")),
                num(r.get("Hits@10")), num(r.get("M")), num(r.get("warm_MRR")),
                num(ep, "{:.0f}"), flag, num(r.get("train_time_sec"), "{:.0f}")))
            rows.append([task, k[1], k[2], r.get("AUROC"), r.get("AUPRC"), r.get("MRR"),
                         r.get("Hits@10"), r.get("M"), r.get("warm_MRR"), ep,
                         r.get("train_time_sec"), log])

    md.append("\n### How to read it\n")
    md.append("- Rows with **!**: raise the budget (`CONV_EPOCHS=3000 bash experiments/convergence_check.sh`) "
              "before comparing models — nothing here is converged.")
    md.append("- A GNN whose `lr0.03` / `lr0.1` row beats its `tuned` row: the HPO learning-rate grid "
              "(top value 0.01) was cut too low, so the v2 GNN configs are not the real optimum — widen "
              "the grid in tuning_hyperparameter.py and re-tune that model.")
    md.append("- All rows converged and the GNNs still below DistMult on MRR: message passing is not "
              "adding ranking information on this graph — that is the result, and this table is the "
              "evidence that it is not a budget artefact.")

    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "convergence_summary.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    csv_path = os.path.join(out_dir, "convergence_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "model", "variant", "AUROC", "AUPRC", "MRR", "Hits@10", "M",
                    "warm_MRR", "best_epoch", "train_time_sec", "log"])
        w.writerows(rows)
    print("\n".join(md))
    print(f"\n[conv-summary] written: {md_path}  and  {csv_path}")


if __name__ == "__main__":
    main()
