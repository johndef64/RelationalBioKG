"""
fetch_hpo_report.py — standalone W&B report downloader. Does NOT run any HPO; it only
queries EXISTING W&B projects and dumps, for each project:
  * a FULL report CSV  (one row per trial: name, state, ALL logged metrics + ALL hyperparams)
  * the best config JSON (ranked by --sort-metric)
Works while a sweep is still running (it reports whatever trials exist so far).

Point it at projects either by task shortcut or explicitly:
  python experiments/fetch_hpo_report.py --task DTI
  python experiments/fetch_hpo_report.py --task TREATS --sort-metric final_mixed_metric
  python experiments/fetch_hpo_report.py --projects RelationalPKT-DTI-compgcn RelationalPKT-DTI-rgcn

Requires:  wandb login  (or WANDB_API_KEY set).
Output:    experiments/hpo_report/<project>_report.csv  and  <project>_best.json
"""
import argparse
import csv
import json
import os
from pathlib import Path

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "giovannimaria-defilippis-university-of-naples-federico-ii")
OUT_DIR = Path(__file__).resolve().parent / "hpo_report"
PARAM_KEYS = ["conv_layer_num", "dropout", "layer_0", "layer_1", "layer_2", "mlp_out_layer",
              "learning_rate", "opn", "grad_norm", "num_bases", "regularization",
              "weight_decay", "scheduler_gamma", "model_name", "train_negative_rate", "use_layer_norm",
              "protocol", "oversample_rate", "undersample_rate", "disjoint_supervision"]


def _scalar(v):
    """Flatten non-scalar summary values (e.g. hits dict) to a string for CSV."""
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v


def dump_project(api, entity, project, sort_metric):
    try:
        runs = list(api.runs(f"{entity}/{project}"))
    except Exception as e:
        print(f"[!] cannot read {project}: {e}")
        return
    if not runs:
        print(f"[{project}] no runs found."); return

    rows, metric_keys, cfg_keys = [], set(), set()
    for r in runs:
        summ = {k: v for k, v in dict(r.summary).items() if not k.startswith("_")}
        cfg = dict(r.config)
        metric_keys |= set(summ); cfg_keys |= set(cfg)
        rows.append({"run": r.name, "state": r.state,
                     "_summ": summ, "_cfg": cfg,
                     "_sort": summ.get(sort_metric)})

    # sort by chosen metric (runs without it go last)
    rows.sort(key=lambda x: (x["_sort"] is not None, x["_sort"] if x["_sort"] is not None else 0),
              reverse=True)

    OUT_DIR.mkdir(exist_ok=True)
    metric_cols = sorted(metric_keys)
    cfg_cols = PARAM_KEYS + sorted(cfg_keys - set(PARAM_KEYS))
    csv_path = OUT_DIR / f"{project}_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "run", "state", sort_metric] + metric_cols + cfg_cols)
        for i, row in enumerate(rows, 1):
            w.writerow([i, row["run"], row["state"], _scalar(row["_sort"])]
                       + [_scalar(row["_summ"].get(m, "")) for m in metric_cols]
                       + [row["_cfg"].get(k, "") for k in cfg_cols])

    # best config json
    ranked = [r for r in rows if r["_sort"] is not None]
    best_path = None
    if ranked:
        b = ranked[0]
        params = {k: b["_cfg"][k] for k in PARAM_KEYS if k in b["_cfg"]}
        best_path = OUT_DIR / f"{project}_best.json"
        best_path.write_text(json.dumps(
            {"sort_metric": sort_metric, "value": b["_sort"], "run": b["run"], "params": params}, indent=2))

    # A trial that hit CUDA_OOM is caught by tuning_hyperparameter.py, logged with status='skipped'
    # and left in state 'finished' — so counting W&B states would overstate the real budget. What the
    # sweep actually explored is the number of trials that produced the ranking metric.
    errors = {}
    for r in rows:
        err = r["_summ"].get("error")
        if err or r["_summ"].get("status") == "skipped":
            errors[err or "skipped"] = errors.get(err or "skipped", 0) + 1
    best_str = f"{ranked[0]['_sort']:.4f} ({ranked[0]['run']})" if ranked else "n/a"
    print(f"[{project}] {len(rows)} trials launched, {len(ranked)} usable | best {sort_metric}={best_str}")
    if errors:
        detail = ", ".join(f"{n} {k}" for k, n in sorted(errors.items()))
        print(f"           !! {sum(errors.values())} trial(s) produced no result: {detail}")
        print(f"              the effective search budget was {len(ranked)}, not {len(rows)} — "
              f"report that number, and check WHICH configs were lost (OOM hits the widest ones).")
    print(f"           -> {csv_path.name}" + (f"  +  {best_path.name}" if best_path else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", help="DTI or TREATS -> RelationalPKT-<task>-{compgcn,rgcn}")
    g.add_argument("--projects", nargs="+", help="explicit W&B project names")
    ap.add_argument("--models", nargs="+", default=["compgcn", "rgcn", "distmult"])
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--sort-metric", default="val_mixed_metric",
                    help="use best_val_mixed_metric for protocol-v2 sweeps")
    ap.add_argument("--suffix", default="", help="project suffix, e.g. '-v2' -> RelationalPKT-<TASK>-v2-<model>")
    args = ap.parse_args()

    projects = args.projects or [f"RelationalPKT-{args.task}{args.suffix}-{m}" for m in args.models]
    import wandb
    api = wandb.Api()
    print(f"[i] entity={args.entity} | projects={projects} | sort by {args.sort_metric}")
    for p in projects:
        dump_project(api, args.entity, p, args.sort_metric)
    print(f"\n[i] reports in {OUT_DIR}/")


if __name__ == "__main__":
    main()
