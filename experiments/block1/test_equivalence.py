"""
test_equivalence.py — B0, the prerequisite of phase B: does train_b1.py train exactly what v1 trains?

For one task it trains the E1 R-GCN on one seed (run 0 = seed 42), with the E1 flags read from the
E1 run folder, three times, all with --deterministic and PYTHONHASHSEED=0:

  v1     train_and_eval.py (v1.0.0, unchanged), the reference;
  b1     the train_b1.py path for a v1 model: experiments/block1/train_b1.py --model rgcn, with a
         no-op --override (the config value written back), so the copy of the hyperparameters and
         the override machinery are exercised;
  b1reg  train_b1.py --model rgcn_v1: the same R-GCN built through the block-1 registry
         (b1_models.py), the path every new model of phase B will take.

PASS (binding) = b1 and b1reg are bit-identical to v1: the same validation trajectory (every [val]
line), the same test record (every metric, the best epoch) and the same checkpoint, tensor by tensor.

E1 itself is NOT a bit-level reference: it was trained before --deterministic existed (its params
have no such flag), and atomic GPU sums make it non-reproducible in the last bits. The distance of
the v1 run from E1 seed 42 and from the 12-seed E1 distribution is reported, for information only.

Usage (from the repository root):
    PYTHONHASHSEED=0 python experiments/block1/test_equivalence.py --task DTI --check
    PYTHONHASHSEED=0 python experiments/block1/test_equivalence.py --task DTI --out_dir experiments/block1/results/B0_v1
    ... --epochs 10   # quick plumbing check of the code (not binding: E1 trains up to 1500 epochs)
Job: experiments/block1/slurm/b0_equivalence.sbatch
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# E1 R-GCN folders (the same as in x0a_distance_curve.py).
E1_RGCN = {
    "DTI": "models/dti_pkt_taskA_dti.tsv_20260918_130755",
    "TREATS": "models/treats_pkt_taskB_treats.tsv_20260918_151621",
}
# E1 flags passed on unchanged to all three runs.
CARRY = ["tsv", "model", "task", "config", "epochs", "patience", "early_stopping", "min_delta",
         "validation_size", "test_size", "quiet", "evaluate_every", "negative_sampling", "negative_rate",
         "oversample_rate", "undersample_rate", "alpha", "gamma", "alpha_adv", "split_seed",
         "select_metric", "train_negative_rate", "train_negative_rate_default", "disjoint_supervision",
         "learning_rate", "dedup_eval", "warm_eval", "eval_filtered"]
# E1 flags train_b1.py does not expose: they must have the value train_b1.py fixes.
FIXED = {"pretrain_epochs": 0, "freeze_base": False, "dry_run": False, "rank_after_train": False,
         "detect_anomaly": False}
NOOP_OVERRIDE_KEY = "mlp_out_layer"
VAL_LINE = re.compile(r"^\[val\] \S+ \S+ (run .*)$")     # the timestamp (date, time) is dropped


class Checks:
    def __init__(self):
        self.rows, self.ok = [], True

    def add(self, name, passed, detail=""):
        self.rows.append((name, bool(passed), detail))
        self.ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def to_flags(params, epochs):
    flags = []
    for key in CARRY:
        value = params.get(key)
        if key == "epochs" and epochs is not None:
            value = epochs
        if value is None or value is False:
            continue
        flags += [f"--{key}"] if value is True else [f"--{key}", str(value)]
    return flags + ["--runs", "1", "--deterministic"]


def check_inputs(task, chk):
    folder = E1_RGCN[task]
    path = os.path.join(folder, "rgcn_params.json")
    chk.add("E1 params", os.path.exists(path), path)
    if not chk.ok:
        return None, None
    with open(path, encoding="utf-8") as f:
        params = json.load(f)
    chk.add("E1 is R-GCN on this task", params.get("model") == "rgcn" and params.get("task") == task,
            f"model {params.get('model')}, task {params.get('task')}")
    bad = {k: params.get(k) for k, v in FIXED.items() if params.get(k, v) != v}
    chk.add("E1 used no flag that train_b1.py fixes", not bad, str(bad) if bad else "")
    chk.add("E1 was trained without --deterministic (so it is an information-only reference)",
            not params.get("deterministic", False), "")
    chk.add("task graph", os.path.exists(params["tsv"]), params["tsv"])
    with open(os.path.join(REPO, "src", "models_params.json"), encoding="utf-8") as f:
        cfg = json.load(f).get(params["config"], {})
    chk.add("config with an rgcn block", "rgcn" in cfg and NOOP_OVERRIDE_KEY in cfg.get("rgcn", {}), params["config"])
    metrics_path = os.path.join(folder, "rgcn_metrics.json")
    chk.add("E1 metrics", os.path.exists(metrics_path), metrics_path)
    if not chk.ok:
        return None, None
    with open(metrics_path, encoding="utf-8") as f:
        e1 = json.load(f)
    chk.add("E1 metrics hold 12 seeds starting at 42",
            len(e1["individual_runs"]) == 12 and e1["individual_runs"][0]["seed"] == 42, "")
    for script in ("train_and_eval.py", "experiments/block1/train_b1.py", "experiments/block1/b1_models.py"):
        chk.add(f"script {script}", os.path.exists(script), "")
    try:
        import torch  # noqa: F401
        chk.add("torch importable", True, f"cuda available: {torch.cuda.is_available()}")
    except ImportError as exc:
        chk.add("torch importable", False, str(exc))
    return params, (e1, cfg["rgcn"][NOOP_OVERRIDE_KEY] if "rgcn" in cfg else None)


def launch(name, cmd, runs_dir, model):
    metrics = os.path.join(runs_dir, name, f"{model}_metrics.json")
    log = os.path.join(runs_dir, f"{name}.log")
    if os.path.exists(metrics):
        print(f"[b0] {name}: finished run found, reused ({metrics})", flush=True)
        return metrics, log, 0
    print(f"[b0] {time.strftime('%F %T')} {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONHASHSEED="0")).returncode
    print(f"[b0] {name}: exit code {rc} after {time.time() - t0:.0f} s, log {log}", flush=True)
    return metrics, log, rc


def val_lines(log):
    with open(log, encoding="utf-8", errors="replace") as fh:
        return [m.group(1) for m in (VAL_LINE.match(line.rstrip("\n")) for line in fh) if m]


def compare(ref, other, name, chk):
    import torch
    with open(ref["metrics"], encoding="utf-8") as f:
        a = json.load(f)
    with open(other["metrics"], encoding="utf-8") as f:
        b = json.load(f)
    ra = {k: v for k, v in a["individual_runs"][0].items() if k != "train_time_sec"}
    rb = {k: v for k, v in b["individual_runs"][0].items() if k != "train_time_sec"}
    diff = sorted(k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k))
    chk.add(f"{name}: test record identical to v1", not diff,
            ", ".join(f"{k}: {ra.get(k)} vs {rb.get(k)}" for k in diff[:6]))
    pa = {k: v for k, v in a["protocol"].items() if k != "model"}
    pb = {k: v for k, v in b["protocol"].items() if k != "model"}
    chk.add(f"{name}: protocol identical to v1", pa == pb,
            "" if pa == pb else str({k: (pa.get(k), pb.get(k)) for k in set(pa) | set(pb) if pa.get(k) != pb.get(k)}))
    va, vb = val_lines(ref["log"]), val_lines(other["log"])
    first = next((i for i, (x, y) in enumerate(zip(va, vb)) if x != y), None)
    same = first is None and len(va) == len(vb) and len(va) > 0
    chk.add(f"{name}: validation trajectory identical to v1", same,
            f"{len(va)} validation lines" if same else
            (f"first difference at line {first}: '{va[first]}' vs '{vb[first]}'" if first is not None
             else f"{len(va)} vs {len(vb)} lines"))
    sa = torch.load(ref["ckpt"], map_location="cpu")
    sb = torch.load(other["ckpt"], map_location="cpu")
    keys_ok = list(sa) == list(sb)
    unequal = [k for k in sa if k in sb and not torch.equal(sa[k], sb[k])]
    chk.add(f"{name}: checkpoint bit-identical to v1", keys_ok and not unequal,
            f"{len(sa)} tensors" if keys_ok and not unequal else
            f"keys equal: {keys_ok}; differing tensors: {unequal[:5]}")
    return {"record_diff": diff, "val_lines": [len(va), len(vb)], "first_val_diff": first,
            "differing_tensors": unequal, "keys_equal": keys_ok}


def main():
    ap = argparse.ArgumentParser(description="B0: train_b1.py against train_and_eval.py, bit for bit.")
    ap.add_argument("--task", required=True, choices=sorted(E1_RGCN))
    ap.add_argument("--out_dir", default=None, help="Version folder, e.g. experiments/block1/results/B0_v1")
    ap.add_argument("--epochs", type=int, default=None, help="Override the E1 epochs (plumbing check, not binding).")
    ap.add_argument("--skip_registry", action="store_true", help="Do not run the b1reg arm.")
    ap.add_argument("--check", action="store_true", help="Verify the requirements only; write nothing.")
    args = ap.parse_args()

    if os.path.abspath(os.getcwd()) != REPO:
        sys.exit(f"[b0] run from the repository root ({REPO})")
    chk = Checks()
    chk.add("PYTHONHASHSEED=0", os.environ.get("PYTHONHASHSEED") == "0", os.environ.get("PYTHONHASHSEED", "unset"))
    params, extra = check_inputs(args.task, chk)
    if args.check or not chk.ok:
        print(f"[b0] CHECK {args.task}: {'ALL OK' if chk.ok else 'FAILED'}")
        sys.exit(0 if chk.ok else 1)
    if not args.out_dir:
        sys.exit("[b0] --out_dir is required to run")
    e1, noop_value = extra
    binding = args.epochs is None

    runs_dir = os.path.join(args.out_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    tag = args.task.lower() + ("" if binding else f"_ep{args.epochs}")
    flags = to_flags(params, args.epochs)
    arms = {
        "v1": ([sys.executable, "-u", "train_and_eval.py"] + flags +
               ["--models_dir", runs_dir, "--run_name", f"{tag}_v1"], "rgcn"),
        "b1": ([sys.executable, "-u", "experiments/block1/train_b1.py"] + flags +
               ["--override", f"{NOOP_OVERRIDE_KEY}={json.dumps(noop_value)}",
                "--out_dir", runs_dir, "--run_name", f"{tag}_b1"], "rgcn"),
    }
    if not args.skip_registry:
        reg_flags = list(flags)
        reg_flags[reg_flags.index("--model") + 1] = "rgcn_v1"
        arms["b1reg"] = ([sys.executable, "-u", "experiments/block1/train_b1.py"] + reg_flags +
                         ["--out_dir", runs_dir, "--run_name", f"{tag}_b1reg"], "rgcn_v1")
    out = {}
    for arm, (cmd, model) in arms.items():
        metrics, log, rc = launch(f"{tag}_{arm}", cmd, runs_dir, model)
        chk.add(f"{arm}: run finished", rc == 0 and os.path.exists(metrics), f"exit code {rc}")
        out[arm] = {"metrics": metrics, "log": log, "ckpt": os.path.join(runs_dir, f"{tag}_{arm}", f"{model}_run0.pt"),
                    "cmd": cmd}
    results = {}
    if chk.ok:
        for arm in [a for a in arms if a != "v1"]:
            results[arm] = compare(out["v1"], out[arm], arm, chk)

    info = {}
    if os.path.exists(out["v1"]["metrics"]):
        with open(out["v1"]["metrics"], encoding="utf-8") as f:
            v1rec = json.load(f)["individual_runs"][0]
        e1_42 = e1["individual_runs"][0]
        for key in ("MRR", "Auroc", "Auprc", "Hits@10", "best_epoch"):
            seeds = [r[key] for r in e1["individual_runs"]]
            info[key] = {"v1_deterministic": v1rec.get(key), "E1_seed42": e1_42.get(key),
                         "E1_mean": statistics.mean(seeds), "E1_sd": statistics.stdev(seeds)}

    os.makedirs(os.path.join(args.out_dir, "summary"), exist_ok=True)
    stem = os.path.join(args.out_dir, "summary", f"b0_{tag}")
    verdict = ("PASS" if chk.ok else "FAIL") + ("" if binding else " (plumbing check only, not binding)")
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump({"task": args.task, "binding": binding, "epochs": args.epochs or params["epochs"],
                   "verdict": verdict, "checks": chk.rows, "comparisons": results, "vs_E1": info,
                   "e1_folder": E1_RGCN[args.task], "commands": {a: o["cmd"] for a, o in out.items()}}, f, indent=2)
    lines = [f"# B0 equivalence, {args.task}{'' if binding else f' ({args.epochs} epochs)'}: {verdict}", "",
             f"E1 folder: `{E1_RGCN[args.task]}`. Arms: v1 = train_and_eval.py, b1 = train_b1.py --model rgcn "
             f"(no-op override {NOOP_OVERRIDE_KEY}={noop_value}), b1reg = train_b1.py --model rgcn_v1 (registry). "
             "All with --deterministic, seed 42, PYTHONHASHSEED=0.", "", "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if p else 'FAIL'} | {d} |" for n, p, d in chk.rows]
    if info:
        lines += ["", "For information only (E1 was not deterministic):", "",
                  "| metric | v1 now, deterministic | E1 seed 42 | E1 mean ± sd (12 seeds) |", "|---|---|---|---|"]
        fmt = lambda k, x: f"{x:.0f}" if k == "best_epoch" else f"{x:.4f}"   # noqa: E731
        lines += [f"| {k} | {fmt(k, v['v1_deterministic'])} | {fmt(k, v['E1_seed42'])} | "
                  f"{fmt(k, v['E1_mean'])} ± {fmt(k, v['E1_sd'])} |" for k, v in info.items()]
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[b0] {args.task}: {verdict} -> {stem}.md")
    sys.exit(0 if chk.ok else 1)


if __name__ == "__main__":
    main()
