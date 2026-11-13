"""
train_b1.py — training entry point of block 1 (experiments after release v1.0.0).

It runs the v1.0.0 training loop itself, not a copy: train_and_eval.py is imported as a module and
its main() is called. Dataset and split, negative sampling, focal loss with adversarial weighting,
type-constrained filtered evaluation, selection on M, early stopping, --warm_eval / --dedup_eval
and --deterministic are therefore the released code. No v1 file is modified. What block 1 adds is
injected from outside, in two places only:

  1. Hyperparameter overrides (--override key=value). The --config block of src/models_params.json
     is copied into the run folder (b1_models_params.json), the overrides are applied to the copy,
     and the module-level path train_and_eval.models_params_path is pointed at it, so get_model()
     and --train_negative_rate auto read the copy exactly as they read the v1 file.
  2. New models (--model <name> registered in experiments/block1/b1_models.py), built by a wrapper
     of train_and_eval.get_model that calls the original for the v1 names.

With a v1 model and no override, a run is the run `train_and_eval.py` does with the same flags.
experiments/block1/test_equivalence.py checks this bit for bit under --deterministic.

Usage (from the repository root, PYTHONHASHSEED=0 as in E1), e.g. the E1 R-GCN of Task B at 400
dimensions:
    PYTHONHASHSEED=0 python experiments/block1/train_b1.py --tsv dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip \
        --task TREATS --model rgcn --config PKT-TREATS-best-v2b --override mlp_out_layer=400 --override layer_0=400 \
        <the E1 flags> --out_dir experiments/block1/results/B1_v1/runs --run_name treats_rgcn_d400

Output: <out_dir>/<run_name>/ with the files train_and_eval.py writes (<model>_params.json,
<model>_run<i>.pt, <model>_metrics.json) plus b1_models_params.json. Checkpoints are git-ignored.
Downstream v1 tools that rebuild a model from src/models_params.json (drug_eval.py,
dump_test_ranks.py) do not know the overrides or the new models: block 1 needs its own rank dump.
"""
import argparse
import copy
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_ROOT = os.path.join("experiments", "block1", "results")
V1_MODELS = ["rgcn", "rgat", "compgcn", "distmult"]


def fail(msg, code=2):
    print(f"[train_b1] ERROR: {msg}", flush=True)
    sys.exit(code)


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def build_parser(model_choices):
    # The v1 flags with the v1 defaults (train_and_eval.py, __main__), minus the two paths that the
    # protocol never uses: multi-relational pretraining and --rank_after_train.
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument('--tsv', type=str, required=True)
    p.add_argument('-m', '--model', type=str, required=True, choices=model_choices)
    p.add_argument('-r', '--runs', type=int, default=1)
    p.add_argument('-e', '--epochs', type=int, default=400)
    p.add_argument('-p', '--patience', type=int, default=20)
    p.add_argument('--early_stopping', action='store_true')
    p.add_argument('--min_delta', type=float, default=0.0)
    p.add_argument('--validation_size', type=float, default=0.1)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--evaluate_every', type=int, default=5)
    p.add_argument('--negative_sampling', type=str, default='filtered')
    p.add_argument('--negative_rate', type=float, default=1)
    p.add_argument('--oversample_rate', type=int, default=5)
    p.add_argument('--undersample_rate', type=float, default=0.5)
    p.add_argument('--alpha', type=float, default=0.25)
    p.add_argument('--gamma', type=float, default=3.0)
    p.add_argument('--alpha_adv', type=float, default=2.0)
    p.add_argument('--split_seed', type=int, default=None)
    p.add_argument('--select_metric', type=str, default='loss', choices=['loss', 'mixed'])
    p.add_argument('--train_negative_rate', type=str, default=None)
    p.add_argument('--train_negative_rate_default', type=int, default=5)
    p.add_argument('--disjoint_supervision', type=float, default=0.0)
    p.add_argument('--detect_anomaly', action='store_true')
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--learning_rate', type=float, default=None)
    p.add_argument('--dedup_eval', action='store_true')
    p.add_argument('--warm_eval', action='store_true')
    p.add_argument('--eval_filtered', action='store_true', default=True)
    p.add_argument('--task', type=str, required=True)
    p.add_argument('--config', type=str, required=True)
    # ---- block 1 ----
    p.add_argument('--override', action='append', default=[], metavar='KEY=VALUE',
                   help='Hyperparameter of the --config block of --model to replace (repeatable). '
                        'VALUE is read as JSON (400, 0.1, true, "x"). The key must already exist in '
                        'the block, so that a typo cannot silently add an unused parameter.')
    p.add_argument('--out_dir', type=str, required=True,
                   help=f'Parent folder of the run, inside {RESULTS_ROOT}/ (e.g. {RESULTS_ROOT}/B1_v1/runs).')
    p.add_argument('--run_name', type=str, required=True, help='Name of the run folder inside --out_dir.')
    return p


def params_copy(src_path, config, model, overrides, registry):
    """The --config block of src/models_params.json, with the block-1 changes applied."""
    with open(src_path, encoding="utf-8") as f:
        all_params = json.load(f)
    if config not in all_params:
        fail(f"config '{config}' not in {src_path}")
    cfg = copy.deepcopy(all_params[config])
    if model in registry:
        spec = registry[model]
        if spec.base not in cfg:
            fail(f"config '{config}' has no '{spec.base}' block, base of the block-1 model '{model}'")
        block = dict(cfg[spec.base])
        block.update(spec.defaults)
        block["model_name"] = model
        cfg[model] = block
    elif overrides and model not in cfg:
        fail(f"config '{config}' has no '{model}' block to override")
    for key, value in overrides.items():
        if key not in cfg[model]:
            fail(f"--override {key}: not a hyperparameter of '{model}' in '{config}' "
                 f"(keys: {sorted(cfg[model])})")
        cfg[model][key] = value
    return {config: cfg}


def git_state():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, cwd=REPO).stdout.strip().splitlines()
        return {"commit": head, "modified_tracked_files": len(dirty)}
    except OSError:
        return {"commit": None, "modified_tracked_files": None}


def main():
    if os.path.abspath(os.getcwd()) != REPO:
        fail(f"run from the repository root ({REPO}): train_and_eval.py reads ./src/models_params.json")
    if os.environ.get("PYTHONHASHSEED") != "0":
        fail("PYTHONHASHSEED=0 is required (as in E1): without it the entity indexing changes between "
             "processes and the run is not comparable with E1")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, REPO)
    import b1_models
    registry = b1_models.REGISTRY
    clash = set(registry) & set(V1_MODELS)
    if clash:
        fail(f"block-1 models must not reuse v1 names: {sorted(clash)}")

    args = build_parser(V1_MODELS + sorted(registry)).parse_args()
    overrides = {}
    for item in args.override:
        if "=" not in item:
            fail(f"--override {item}: expected KEY=VALUE")
        key, value = item.split("=", 1)
        overrides[key.strip()] = parse_value(value.strip())

    out_dir = os.path.normpath(args.out_dir)
    try:
        inside = os.path.commonpath([os.path.abspath(out_dir), os.path.abspath(RESULTS_ROOT)]) == os.path.abspath(RESULTS_ROOT)
    except ValueError:            # different drives on Windows
        inside = False
    if not inside:
        fail(f"--out_dir must be inside {RESULTS_ROOT}/ (rule 2 of experiments/block1/README.md)")
    run_dir = os.path.join(out_dir, args.run_name)
    model_save_path = os.path.join(run_dir, f"{args.model.lower()}.pt")
    if os.path.exists(model_save_path.replace(".pt", "_metrics.json")):
        fail(f"{run_dir} already holds a finished run: choose another --run_name or a new version")
    os.makedirs(run_dir, exist_ok=True)

    # The v1 module: importing it runs nothing (its CLI lives under `if __name__ == '__main__'`).
    import train_and_eval as tae
    params_path = os.path.join(run_dir, "b1_models_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params_copy(tae.models_params_path, args.config, args.model, overrides, registry), f, indent=4)
    tae.models_params_path = params_path

    v1_get_model = tae.get_model

    def get_model(model_name, task, in_channels_dict, num_nodes_per_type, num_entities, num_relations,
                  config_name="pathogen31-128"):
        if model_name not in registry:
            return v1_get_model(model_name, task, in_channels_dict, num_nodes_per_type, num_entities,
                                num_relations, config_name=config_name)
        with open(tae.models_params_path, encoding="utf-8") as fh:
            p = json.load(fh)[config_name][model_name]
        print(f"[get_model] block-1 model '{model_name}' from config '{config_name}': {p}")
        model = registry[model_name].build(p, in_channels_dict, num_nodes_per_type, num_entities,
                                           num_relations, tae.device)
        return model, p['learning_rate'], p['regularization'], p['grad_norm'], p['weight_decay'], p['scheduler_gamma']

    tae.get_model = get_model

    # What train_and_eval.py's __main__ sets before calling main(), in the same order.
    args.pretrain_epochs, args.freeze_base, args.dry_run, args.rank_after_train = 0, False, False, False
    tae.args = args
    tae.CONFIG_NAME = args.config
    print(f"[i] Running on {tae.device}")
    if args.deterministic:
        from src.deterministic_ops import configure
        configure()
        print("[i] Deterministic mode: fixed-order GPU sums for R-GCN and the decoder (src/deterministic_ops.py)")
    if tae.torch.cuda.is_available():
        print(f"[i] GPU: {tae.torch.cuda.get_device_name(0)} | torch {tae.torch.__version__} | CUDA {tae.torch.version.cuda}")

    record = dict(vars(args))
    record["b1"] = {"script": "experiments/block1/train_b1.py", "overrides": overrides,
                    "block1_model": args.model in registry, "params_file": params_path.replace(os.sep, "/"),
                    "git": git_state(), "pythonhashseed": os.environ.get("PYTHONHASHSEED")}
    with open(model_save_path.replace(".pt", "_params.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=4)
    print(f"[train_b1] run folder {run_dir} | model {args.model} | overrides {overrides or 'none'}", flush=True)

    tae.main(args.model, tae._resolve_dataset_path(args.tsv), args.task, args.runs, args.epochs, args.patience,
             args.validation_size, args.test_size, args.quiet, args.evaluate_every, args.negative_rate,
             model_save_path, args.oversample_rate, args.undersample_rate, args.pretrain_epochs,
             args.freeze_base, args.alpha, args.gamma, args.alpha_adv, args.early_stopping, args.min_delta,
             args.eval_filtered, split_seed=args.split_seed, select_metric=args.select_metric,
             train_negative_rate=tae.resolve_train_negative_rate(args),
             disjoint_supervision=args.disjoint_supervision, detect_anomaly=args.detect_anomaly,
             warm_eval=args.warm_eval, dedup_eval=args.dedup_eval, learning_rate=args.learning_rate)
    print(f"[train_b1] done: {model_save_path.replace('.pt', '_metrics.json')}", flush=True)


if __name__ == "__main__":
    main()
