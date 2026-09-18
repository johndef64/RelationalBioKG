#!/usr/bin/env python3
"""
models_index.py — what is actually inside models/, in one readable table.

A trained-model folder is named <task>_<dataset>_<timestamp>, which says nothing about WHICH model
it holds or whether the run finished: that lives in <model>_params.json and <model>_metrics.json.
With dozens of folders accumulated across dataset versions, telling a definitive E1 model from a
two-epoch smoke test means opening files one by one. This does it for you.

READ-ONLY: it opens only the two JSON files per folder (never a checkpoint, never torch), so it is
safe to run while training.

Verdicts:
  E1 definitivo    finished, >= MIN_E1_RUNS seeds, config carries the current suffix -> usable
  parziale         finished but fewer seeds (a quick check, E0, a convergence run)
  in corso         checkpoints on disk, no metrics.json yet: still training
  fallito          folder created, no checkpoint: the run died before the first epoch
  superato         finished but trained under an older config suffix (older data) -> deletable
  smoke test       1-2 epochs: a pipeline check, never a result

Usage:
  python experiments/models_index.py                      # everything
  python experiments/models_index.py --current-only       # only what is usable now
  python experiments/models_index.py --suffix -v2b        # what counts as "current" (default -v2b)
  python experiments/models_index.py --stale              # only what can be deleted, with the size
"""
import argparse
import glob
import json
import os

MIN_E1_RUNS = 12


def folder_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def human(n):
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit != "G" else f"{n:.1f}G"
        n /= 1024


def inspect(folder, suffix):
    params = sorted(glob.glob(os.path.join(folder, "*_params.json")))
    if not params:
        return None
    with open(params[0]) as f:
        p = json.load(f)

    model = p.get("model", "?")
    config = p.get("config", "?")
    epochs = p.get("epochs", 0)
    runs_req = p.get("runs", 0)
    task = p.get("task", "?")

    n_pt = len([f for f in os.listdir(folder)
                if f.endswith(".pt") and "pretrained" not in f])

    mean = sd = best_run = None
    n_done = 0
    metrics = sorted(glob.glob(os.path.join(folder, "*_metrics.json")))
    if metrics:
        with open(metrics[0]) as f:
            m = json.load(f)
        ind = m.get("individual_runs", [])
        n_done = len(ind)
        if ind:
            best_run = max(range(len(ind)), key=lambda i: ind[i].get("MRR", 0.0))
            mrrs = [r.get("MRR") for r in ind if r.get("MRR") is not None]
            if mrrs:
                mean = sum(mrrs) / len(mrrs)
                sd = (sum((x - mean) ** 2 for x in mrrs) / max(1, len(mrrs) - 1)) ** 0.5

    current = str(config).endswith(suffix)
    if epochs <= 2:
        verdict = "smoke test"
    elif not metrics and n_pt == 0:
        verdict = "fallito"
    elif not metrics:
        verdict = "in corso"
    elif not current:
        verdict = "superato"
    elif n_done >= MIN_E1_RUNS:
        verdict = "E1 definitivo"
    else:
        verdict = "parziale"

    return dict(folder=folder, task=task, model=model, config=config, epochs=epochs,
                runs_req=runs_req, n_done=n_done, n_pt=n_pt, mean=mean, sd=sd,
                best_run=best_run, size=folder_size(folder), verdict=verdict)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--suffix", default="-v2b", help="config suffix that counts as current data")
    ap.add_argument("--current-only", action="store_true", help="only usable folders")
    ap.add_argument("--stale", action="store_true", help="only what can be deleted")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.models_dir, "*/"))):
        r = inspect(d.rstrip("/\\"), args.suffix)
        if r:
            rows.append(r)
    if not rows:
        print(f"nessuna cartella modello in {args.models_dir}/")
        return

    keep = {"E1 definitivo", "parziale", "in corso"}
    if args.current_only:
        rows = [r for r in rows if r["verdict"] in ("E1 definitivo", "parziale")]
    if args.stale:
        rows = [r for r in rows if r["verdict"] in ("superato", "smoke test", "fallito")]

    rows.sort(key=lambda r: (r["verdict"] != "E1 definitivo", r["task"], r["model"], r["folder"]))

    print(f"{'CARTELLA':>15}  {'TASK':6} {'MODELLO':9} {'CONFIG':22} {'EP':>5} {'RUN':>6} "
          f"{'MRR medio':>16} {'BEST':>5} {'DIM':>6}  VERDETTO")
    print("-" * 125)
    for r in rows:
        stamp = os.path.basename(r["folder"])[-15:]
        mrr = f"{r['mean']:.3f} ± {r['sd']:.3f}" if r["mean"] is not None and r["sd"] is not None \
            else (f"{r['mean']:.3f}" if r["mean"] is not None else "-")
        best = f"run{r['best_run']}" if r["best_run"] is not None else "-"
        runs = f"{r['n_done']}/{r['runs_req']}" if r["n_done"] else f"{r['n_pt']}pt"
        print(f"{stamp:>15}  {r['task']:6} {r['model']:9} {str(r['config'])[:22]:22} "
              f"{r['epochs']:>5} {runs:>6} {mrr:>16} {best:>5} {human(r['size']):>6}  {r['verdict']}")

    print()
    by_verdict = {}
    for r in rows:
        by_verdict.setdefault(r["verdict"], []).append(r)
    for v, rs in sorted(by_verdict.items()):
        print(f"  {v:15s} {len(rs):3d} cartelle, {human(sum(x['size'] for x in rs))}")

    stale = [r for r in rows if r["verdict"] in ("superato", "smoke test", "fallito")]
    if stale and not args.current_only:
        print(f"\n  recuperabili cancellando le cartelle 'superato/smoke test/fallito': "
              f"{human(sum(r['size'] for r in stale))}")
        print("  elencale con:  python experiments/models_index.py --stale")

    usable = [r for r in rows if r["verdict"] == "E1 definitivo"]
    if usable:
        print("\n  da usare per E4 (una cartella per modello, servono solo 3 file ciascuna):")
        for r in usable:
            base = os.path.basename(r["folder"])
            print(f"    {base}/{r['model']}_params.json  {r['model']}_metrics.json  "
                  f"{r['model']}_run{r['best_run']}.pt")


if __name__ == "__main__":
    main()
