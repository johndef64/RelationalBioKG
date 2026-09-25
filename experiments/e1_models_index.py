"""Index of the E1 checkpoints in models/: which seed each file is, its test metrics, which one is best.

    python experiments/e1_models_index.py            # all folders in models/
    python experiments/e1_models_index.py --paths    # only the path of the best checkpoint per folder

Nothing is guessed: <model>_run<i>.pt is seed 42+i, and its test metrics are individual_runs[i] of
<model>_metrics.json in the same folder. "Best" = highest test MRR (the checkpoint used by the
single-seed stratified analysis); for the multi-seed confirmation use all twelve.
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--models_dir", default="models")
ap.add_argument("--paths", action="store_true")
args = ap.parse_args()

for d in sorted(glob.glob(os.path.join(args.models_dir, "*", ""))):
    mf = glob.glob(os.path.join(d, "*_metrics.json"))
    if not mf:
        continue
    model = os.path.basename(mf[0])[: -len("_metrics.json")]
    runs = json.load(open(mf[0]))["individual_runs"]
    best = max(range(len(runs)), key=lambda i: runs[i]["MRR"])
    pt = lambda i: os.path.join(d, f"{model}_run{i}.pt")
    if args.paths:
        print(pt(best) if os.path.exists(pt(best)) else f"{pt(best)}  (MISSING)")
        continue
    have = sum(os.path.exists(pt(i)) for i in range(len(runs)))
    print(f"\n{os.path.basename(os.path.normpath(d))}  [{model}]  checkpoints {have}/{len(runs)}")
    print("   run  seed  best_ep   MRR    AUROC  AUPRC   file")
    for i, r in enumerate(runs):
        mark = "  <- best" if i == best else ""
        f = "ok" if os.path.exists(pt(i)) else "--"
        print(f"   {i:>3}  {r['seed']:>4}  {r.get('best_epoch', '?'):>7}  {r['MRR']:.3f}  {r['Auroc']:.3f}  "
              f"{r['Auprc']:.3f}   {f}{mark}")
