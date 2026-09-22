#!/usr/bin/env python
"""
plot_training_curves.py — validation curves from the training logs, without instrumenting the code.

train_and_eval.py already prints one line per evaluation:

    [val] run 0 epoch 245 | loss 0.0412 | AUROC 0.8873 | AUPRC 0.9102 | MRR 0.4471 | M 0.6421

so every run in experiments/logs/ already carries its full learning curve. This reads those lines and
draws them: no TensorBoard, no extra dependency, and it works retroactively on logs already produced.

It answers two questions:
  1. OPERATIONAL. Is the epoch budget enough, or is the selected checkpoint still drifting towards the
     ceiling? For models whose median best epoch sits at the ceiling, the summary also reports how
     much the curve was still gaining over the last 10% of the budget: only a real gain is flagged.
  2. FOR THE PAPER. Do the encoders converge differently? On Task B the claim is that message passing
     costs ranking quality; a curve where MRR plateaus early for the GNNs while the embedding-only
     model keeps climbing is direct evidence for it, next to the final numbers.

Usage:
    python experiments/plot_training_curves.py                         # experiments/logs/v2
    python experiments/plot_training_curves.py --logdir experiments/logs/e0_ladder_v2b
    python experiments/plot_training_curves.py --metric MRR            # headline figure metric
    python experiments/plot_training_curves.py --out figures/curves

Output: <out>/curves_<TASK>.{pdf,png}      headline, one metric, models overlaid
        <out>/curves_<TASK>_all.{pdf,png}  loss, AUROC, AUPRC, MRR and M as small multiples

Reading the figures: the line is the mean over seeds and the band is +/- 1 sd. Early stopping ends
runs at different epochs, so the line is drawn faded past the point where the first run stops: there
the mean is taken over the survivors only and is not comparable with the solid part.
"""
import argparse
import math
import os
import re
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

VAL_RE = re.compile(
    r"\[val\]\s+run\s+(\d+)\s+epoch\s+(\d+)\s*\|\s*loss\s+([\d.eE+-]+)\s*\|\s*AUROC\s+([\d.eE+-]+)\s*"
    r"\|\s*AUPRC\s+([\d.eE+-]+)\s*\|\s*MRR\s+([\d.eE+-]+)\s*\|\s*M\s+([\d.eE+-]+)")
BEST_RE = re.compile(r"Run\s+(\d+)\s*\|\s*best_epoch:\s*(\d+)")
# e1_DTI_rgcn_20260918_120000.log | e0_DTI_rgcn_v2_20260916_154626.log | e3_ctx_full_2026...log
NAME_RE = re.compile(r"^(e\d[a-z]?)_(.+)_(\d{8}_\d{6})\.log$")

METRICS = ["loss", "AUROC", "AUPRC", "MRR", "M"]
MODEL_ORDER = ["distmult", "rgcn", "compgcn"]
MODEL_LABEL = {"distmult": "DistMult (no GNN)", "rgcn": "R-GCN", "compgcn": "CompGCN"}

# Slots 1-3 of the validated reference palette, assigned in fixed order and never cycled.
# The dash pattern is a secondary encoding: identity survives greyscale printing and colour-vision
# deficiency, which a paper figure has to do.
STYLE = {
    "distmult": ("#2a78d6", (0, ())),            # blue, solid
    "rgcn":     ("#eb6834", (0, (6, 2))),        # orange, dashed
    "compgcn":  ("#1baf7a", (0, (1.5, 1.8))),    # aqua, dotted
}
FALLBACK = [("#eda100", (0, (5, 1, 1, 1))), ("#e87ba4", (0, (3, 1, 1, 1))), ("#4a3aa7", (0, (8, 3)))]
INK, INK_SOFT, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#e4e4e0"

# --paper draws the figure at its printed size (half the LNCS text width, two panels side by side),
# so sizes are scaled here rather than by LaTeX. A 6 pt floor keeps every label legible in print.
PRINT = {"on": False}


def F(points):
    return points if not PRINT["on"] else max(6.0, round(points * 0.72, 1))


def W(width):
    return width if not PRINT["on"] else round(width * 0.6, 2)


def parse_log(path):
    """-> {run: {'epochs': [...], 'loss': [...], ...}}, {run: best_epoch}"""
    runs, best = defaultdict(lambda: defaultdict(list)), {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            for line in raw.split("\r"):          # tqdm writes carriage returns
                if (m := VAL_RE.search(line)):
                    r = int(m.group(1))
                    runs[r]["epochs"].append(int(m.group(2)))
                    for name, g in zip(METRICS, range(3, 8)):
                        runs[r][name].append(float(m.group(g)))
                elif (m := BEST_RE.search(line)):
                    best[int(m.group(1))] = int(m.group(2))
    return runs, best


def discover(logdir):
    """-> {(task, model): (runs, best_epochs)} keeping the most recent log per pair."""
    newest = {}
    for fn in sorted(os.listdir(logdir)):
        m = NAME_RE.match(fn)
        if not m:
            continue
        family, middle, stamp = m.groups()
        parts = middle.split("_")
        if family == "e1" and len(parts) >= 2:
            key = (parts[0], parts[-1])                       # (TASK, model)
        elif family == "e0" and len(parts) >= 3:
            key = (parts[0], f"{parts[1]} · {'_'.join(parts[2:])}")   # (TASK, model · variant)
        else:
            key = (family.upper(), middle)
        if key not in newest or stamp > newest[key][0]:
            newest[key] = (stamp, os.path.join(logdir, fn))

    out = {}
    for key, (_, path) in newest.items():
        runs, best = parse_log(path)
        runs = {r: d for r, d in runs.items() if d["epochs"]}
        if runs:
            out[key] = (runs, best, os.path.basename(path))
    return out


def aggregate(runs, metric):
    """-> epochs, mean, sd, n_at_epoch, n_total. Aligned on the shared evaluation grid."""
    grid = sorted({e for d in runs.values() for e in d["epochs"]})
    by_epoch = defaultdict(list)
    for d in runs.values():
        for e, v in zip(d["epochs"], d[metric]):
            if not math.isnan(v):
                by_epoch[e].append(v)
    xs = [e for e in grid if by_epoch[e]]
    mean = [st.mean(by_epoch[e]) for e in xs]
    sd = [st.stdev(by_epoch[e]) if len(by_epoch[e]) > 1 else 0.0 for e in xs]
    n = [len(by_epoch[e]) for e in xs]
    return xs, mean, sd, n, len(runs)


def style_for(model, i):
    return STYLE.get(model, FALLBACK[i % len(FALLBACK)])


def draw(ax, series, metric, show_band=True, labels=True):
    """series: list of (model, runs). Returns the legend handles.

    labels=False suppresses the axis titles: inside the small multiples the panel title already
    names the metric and every panel shares the same x, so repeating them is pure chrome.
    """
    handles = []
    for i, (model, runs) in enumerate(series):
        colour, dash = style_for(model, i)
        xs, mean, sd, n, n_tot = aggregate(runs, metric)
        if not xs:
            continue
        # solid while every seed is still training; faded once early stopping thins the population,
        # because past that point the mean is over survivors only
        full = [k for k, c in enumerate(n) if c == n_tot]
        cut = full[-1] + 1 if full else len(xs)
        ax.plot(xs[:cut], mean[:cut], color=colour, linestyle=dash, linewidth=W(2), solid_capstyle="round")
        if cut < len(xs):
            ax.plot(xs[cut - 1:], mean[cut - 1:], color=colour, linestyle=dash, linewidth=W(2),
                    alpha=0.35, solid_capstyle="round")
        if show_band and n_tot > 1:
            ax.fill_between(xs[:cut], [m - s for m, s in zip(mean[:cut], sd[:cut])],
                            [m + s for m, s in zip(mean[:cut], sd[:cut])],
                            color=colour, alpha=0.13, linewidth=W(0))
        handles.append(Line2D([], [], color=colour, linestyle=dash, linewidth=W(2),
                              label=(MODEL_LABEL.get(model, model).replace(" (no GNN)", "") if PRINT["on"]
                                     else f"{MODEL_LABEL.get(model, model)}  (n={n_tot})")))
    if labels:
        ax.set_xlabel("epoch", color=INK_SOFT, fontsize=F(9))
        ax.set_ylabel(metric, color=INK_SOFT, fontsize=F(9))
    ax.grid(True, color=GRID, linewidth=W(0.8))
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=F(8), length=3)
    return handles


def label_ends(ax, series):
    """Direct labels in ink with a coloured marker: identity is never carried by colour alone."""
    for i, (model, runs) in enumerate(series):
        colour, _ = style_for(model, i)
        xs, mean, _, _, _ = aggregate(runs, "M")
        if not xs:
            continue
        ax.plot([xs[-1]], [mean[-1]], marker="o", markersize=W(5), color=colour,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax.annotate(f" {MODEL_LABEL.get(model, model)}", (xs[-1], mean[-1]),
                    color=INK, fontsize=F(8), va="center", ha="left",
                    xytext=(6, 0), textcoords="offset points")


def save(fig, out, stem, formats):
    os.makedirs(out, exist_ok=True)
    written = []
    for ext in formats:
        p = os.path.join(out, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200, facecolor="white")
        written.append(p)
    plt.close(fig)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", default="experiments/logs/v2")
    ap.add_argument("--out", default=None, help="output directory (default: alongside the logs)")
    ap.add_argument("--metric", default="M", choices=METRICS, help="metric of the headline figure")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--paper", action="store_true",
                    help="version for LaTeX: no figure title and no footnote (the caption has them)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="the budget the runs were given; used to flag a checkpoint at the ceiling")
    args = ap.parse_args()
    PRINT["on"] = args.paper

    if not os.path.isdir(args.logdir):
        print(f"[!] {args.logdir} does not exist"); return
    found = discover(args.logdir)
    if not found:
        print(f"[!] no parsable '[val]' lines under {args.logdir}/"
              f"\n    (logs must be named e1_<TASK>_<model>_<timestamp>.log and come from a run"
              f" that printed validation lines)")
        return
    out = args.out or args.logdir

    by_task = defaultdict(list)
    for (task, model), (runs, best, fn) in found.items():
        by_task[task].append((model, runs, best, fn))

    print(f"[i] {args.logdir} -> {len(found)} (task, model) pairs\n")
    for task in sorted(by_task):
        entries = sorted(by_task[task],
                         key=lambda e: MODEL_ORDER.index(e[0]) if e[0] in MODEL_ORDER else 99)
        series = [(m, r) for m, r, _, _ in entries]

        print(f"== Task {task} ==")
        print(f"  {'model':<22}{'seeds':>6}{'last ep.':>10}{'best ep. (median)':>20}"
              f"{'final ' + args.metric:>12}{'gain last 10%':>15}")
        for model, runs, best, fn in entries:
            xs, mean, _, _, n_tot = aggregate(runs, args.metric)
            last = max(max(d["epochs"]) for d in runs.values())
            bests = [best[r] for r in sorted(best)] or [float("nan")]
            med = st.median(bests) if bests and not math.isnan(bests[0]) else float("nan")
            ceiling = args.epochs or last
            # A best epoch at the ceiling is not enough to call a run unconverged: a curve that is
            # flat but still inching up by thousandths also sets its best at the last epoch. What
            # matters is how much it was still gaining, so measure, for every run that reached the
            # ceiling, its gain over the last 10% of the budget, and take the median.
            gains = []
            for d in runs.values():
                if max(d["epochs"]) < 0.98 * ceiling:
                    continue
                vals = dict(zip(d["epochs"], d[args.metric]))
                start = min((e for e in vals if e >= 0.9 * ceiling), default=None)
                if start is not None:
                    gains.append(vals[max(vals)] - vals[start])
            gain = st.median(gains) if gains else float("nan")
            flag = ""
            if not math.isnan(med) and med >= 0.98 * ceiling:
                flag = ("  <- still climbing at the ceiling: raise EPOCHS" if gain > 0.005
                        else "  (best at the ceiling, but flat: converged in practice)")
            gtxt = f"{gain:+.4f}" if not math.isnan(gain) else "n/a"
            print(f"  {MODEL_LABEL.get(model, model):<22}{n_tot:>6}{last:>10}{med:>20.0f}"
                  f"{mean[-1] if mean else float('nan'):>12.3f}{gtxt:>15}{flag}")
        print()

        # --- headline figure -------------------------------------------------
        # printed at half the LNCS text width (12.2 cm / 2) when --paper
        fig, ax = plt.subplots(figsize=(2.35, 1.75) if args.paper else (6.4, 3.8))
        handles = draw(ax, series, args.metric)
        label_ends(ax, series) if args.metric == "M" else None
        if args.paper:
            # LaTeX version: the caption carries the title and the reading notes; the panel keeps
            # only the task it shows, since two tasks are set side by side in the document
            ax.set_title({"DTI": "Task A (DTI)", "TREATS": "Task B (TREATS)"}.get(task, task),
                         color=INK, fontsize=F(10), loc="left")
        else:
            ax.set_title(f"Task {task}: validation {args.metric} during training",
                         color=INK, fontsize=F(11), loc="left", pad=10)
        # side by side in print, one legend is enough: on Task A, whose lower right corner is empty
        if not args.paper or task == "DTI":
            ax.legend(handles=handles, frameon=False, fontsize=F(8), labelcolor=INK, loc="lower right")
        if not args.paper:
            fig.text(0.0, -0.06, "mean over seeds, band = ±1 sd; faded where early stopping has "
                                 "thinned the runs and the mean is over survivors only",
                     fontsize=F(7), color=INK_MUTED, ha="left")
        for p in save(fig, out, f"curves_{task}", args.formats):
            print(f"  -> {p}")

        # --- small multiples -------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4))
        for k, metric in enumerate(METRICS):
            a = axes[k // 3][k % 3]
            handles = draw(a, series, metric, labels=False)
            a.set_title(metric, color=INK, fontsize=F(10), loc="left")
            if k >= 2:            # bottom row of each column
                a.set_xlabel("epoch", color=INK_SOFT, fontsize=F(9))
        axes[1][2].axis("off")
        axes[1][2].legend(handles=handles, frameon=False, fontsize=F(9), labelcolor=INK, loc="center")
        fig.suptitle(f"Task {task}: validation metrics during training",
                     color=INK, fontsize=F(12), x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for p in save(fig, out, f"curves_{task}_all", args.formats):
            print(f"  -> {p}")
        print()


if __name__ == "__main__":
    main()
