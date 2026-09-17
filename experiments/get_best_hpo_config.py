"""
get_best_hpo_config.py — pull the best hyperparameter config(s) from a W&B HPO sweep and
save them ready to use. QUERIES W&B ONLY — it does NOT run any HPO, so it works while a
sweep is still running (it just ranks whatever trials are logged so far).

The E2 sweep logs every trial to projects  RelationalPKT-<TASK>-<model>
(e.g. RelationalPKT-DTI-compgcn, RelationalPKT-DTI-rgcn). This tool, per model:
  * ranks trials by the composite metric,
  * writes the winning config to experiments/hpo_best/<TASK>_<model>_best.json,
  * writes a leaderboard of the top trials to experiments/hpo_best/<TASK>_<model>_leaderboard.csv,
  * with --write, injects the best configs into src/models_params.json as PKT-<TASK>-best.

Requires:  wandb login  (or WANDB_API_KEY set).
Usage:
  python experiments/get_best_hpo_config.py --task DTI            # best-so-far, saves reports
  python experiments/get_best_hpo_config.py --task DTI --write    # + inject into models_params.json
  python experiments/get_best_hpo_config.py --task TREATS --metric final_mixed_metric
"""
import argparse
import csv
import json
import os
from pathlib import Path

PARAM_KEYS = ["conv_layer_num", "dropout", "layer_0", "layer_1", "layer_2", "mlp_out_layer",
              "learning_rate", "opn", "grad_norm", "num_bases", "regularization",
              "weight_decay", "scheduler_gamma", "model_name", "train_negative_rate", "use_layer_norm"]
REPORT_METRICS = ["best_val_mixed_metric", "best_epoch", "val_mixed_metric", "final_mixed_metric", "val_auroc", "val_auprc", "val_mrr",
                  "test_auroc", "test_auprc", "test_mrr"]

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "giovannimaria-defilippis-university-of-naples-federico-ii")
OUT_DIR = Path(__file__).resolve().parent / "hpo_best"
PARAMS_JSON = Path(__file__).resolve().parents[1] / "src" / "models_params.json"


def ranked_runs(api, entity, project, metric, losses=None):
    """Return runs sorted by `metric` desc: list of (value, config, summary, name, state).

    `losses`, when a dict is passed in, is filled with the trials that produced NO result:
    a CUDA_OOM is caught by tuning_hyperparameter.py and logged with status='skipped', which
    leaves the run in W&B state 'finished'. Counting states would therefore overstate how much
    of the search space was really explored, and OOM does not strike at random -- it removes the
    widest configurations first, so the loss is a systematic hole, not a smaller sample.
    """
    out = []
    try:
        runs = api.runs(f"{entity}/{project}")
    except Exception as e:
        print(f"  [!] cannot read project {project}: {e}")
        return out
    for r in runs:
        v = r.summary.get(metric)
        if v is None:
            if losses is not None:
                err = r.summary.get("error") or ("skipped" if r.summary.get("status") == "skipped"
                                                 else f"no '{metric}'")
                losses[err] = losses.get(err, 0) + 1
            continue
        try:
            out.append((float(v), dict(r.config), dict(r.summary), r.name, r.state))
        except Exception:
            continue
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="DTI or TREATS (matches the W&B project prefix)")
    ap.add_argument("--models", nargs="+", default=["compgcn", "rgcn", "distmult"])
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--metric", default="auto",
                    help="metric to rank by. 'auto' (default) = best_val_mixed_metric when the sweep logged it "
                         "(protocol v2), else val_mixed_metric (v1). Or e.g. final_mixed_metric")
    ap.add_argument("--suffix", default="",
                    help="W&B project suffix, e.g. '-v2' -> RelationalPKT-<TASK>-v2-<model>; also names the "
                         "injected config PKT-<TASK>-best<suffix>")
    ap.add_argument("--top", type=int, default=15, help="leaderboard length")
    ap.add_argument("--write", action="store_true",
                    help="inject the best configs into src/models_params.json as PKT-<TASK>-best")
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    OUT_DIR.mkdir(exist_ok=True)

    task_config = {}
    for model in args.models:
        project = f"RelationalPKT-{args.task}{args.suffix}-{model}"
        # 'auto': v2 sweeps log best_val_mixed_metric (best checkpoint), v1 only val_mixed_metric
        losses = {}
        if args.metric == "auto":
            metric = "best_val_mixed_metric"
            runs = ranked_runs(api, args.entity, project, metric, losses)
            if not runs:
                metric, losses = "val_mixed_metric", {}
                runs = ranked_runs(api, args.entity, project, metric, losses)
        else:
            metric = args.metric
            runs = ranked_runs(api, args.entity, project, metric, losses)
        print(f"\n[{project}] ranking by '{metric}' ...")
        if not runs:
            print(f"  no runs with metric '{metric}' yet — skipping")
            continue
        print(f"  {len(runs)} trials with metric; best = {runs[0][0]:.4f} ({runs[0][3]}, {runs[0][4]})")
        real_losses = {k: n for k, n in losses.items() if not k.startswith("no '")}
        if real_losses:
            print(f"  !! {sum(real_losses.values())} trial(s) produced no result "
                  f"({', '.join(f'{n} {k}' for k, n in sorted(real_losses.items()))}): the effective "
                  f"budget for {model} was {len(runs)}. Quote that number, not the requested one.")

        # best config -> JSON (+ collect for models_params.json)
        best_val, best_cfg, _, best_name, _ = runs[0]
        params = {k: best_cfg[k] for k in PARAM_KEYS if k in best_cfg}
        params["model_name"] = model
        task_config[model] = params
        (OUT_DIR / f"{args.task}{args.suffix}_{model}_best.json").write_text(json.dumps(
            {"metric": metric, "value": best_val, "run": best_name, "params": params}, indent=2))

        # leaderboard -> CSV (top N)
        lb_path = OUT_DIR / f"{args.task}{args.suffix}_{model}_leaderboard.csv"
        with open(lb_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "run", "state"] + REPORT_METRICS + PARAM_KEYS)
            for i, (val, cfg, summ, name, state) in enumerate(runs[: args.top], 1):
                w.writerow([i, name, state]
                           + [summ.get(m, "") for m in REPORT_METRICS]
                           + [cfg.get(k, "") for k in PARAM_KEYS])
        print(f"  saved: {args.task}{args.suffix}_{model}_best.json  +  {lb_path.name}")

    if args.write and task_config:
        with open(PARAMS_JSON) as f:
            allp = json.load(f)
        key = f"PKT-{args.task}-best{args.suffix}"
        allp[key] = task_config
        with open(PARAMS_JSON, "w") as f:
            json.dump(allp, f, indent=4)
        print(f"\n[i] injected into {PARAMS_JSON} as '{key}'  ->  HP_CONFIG={key} bash experiments/e1_main_training.sh")
    elif task_config:
        print(f"\n[i] add the best block(s) to src/models_params.json, or re-run with --write.")


if __name__ == "__main__":
    main()
