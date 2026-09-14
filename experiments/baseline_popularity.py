#!/usr/bin/env python3
"""
Popularity (node-degree) baseline for the target relation — no learning at all.

Score of a candidate = how many TRAINING target edges it has (tail degree for tail prediction,
head degree for head prediction). It uses exactly the same data split (get_dataset with the
same seed/sizes), the same type-constrained candidate pools, the same filtered setting and the
same rank definition (#candidates with score >= true score) as the model evaluation in
train_and_eval.py, so its numbers are directly comparable with the GNN rows.

Because degrees are integers, many candidates tie. The model rank definition (>=) counts every
tie against the true node (pessimistic); we also report the expected rank under random
tie-breaking (#greater + (#ties + 1) / 2), which is the fair number for a degree ranking.

AUROC/AUPRC use the same 1:1 filtered test negatives as the v2 protocol (seed = split_seed + 2000)
with pair score log1p(deg_head) + log1p(deg_tail).

Output lines follow the train_and_eval.py log format ("Run 0 | Test Auroc: ...") so that
experiments/protocol_compare_summary.py parses them like any other run.

Usage:
  python experiments/baseline_popularity.py --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task DTI
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torcheval.metrics.functional import binary_auprc, binary_auroc  # noqa: E402
from train_and_eval import get_dataset, cold_start_mask  # noqa: E402
from src.utils import negative_sampling_filtered  # noqa: E402

HITS_K = (1, 3, 10)


def rank_stats(test, pool, deg, positives, mode):
    """Return (pessimistic ranks, random-tie expected ranks) for head or tail prediction."""
    pool = np.asarray(sorted(pool))
    pool_scores = np.array([deg.get(n, 0) for n in pool], dtype=np.int64)
    in_pool = set(pool.tolist())
    pess, rand = [], []
    for h, r, t in test:
        true_node = t if mode == "tail" else h
        if true_node not in in_pool:
            continue
        s = deg.get(true_node, 0)
        greater = int((pool_scores > s).sum())
        ties = int((pool_scores == s).sum()) - 1          # other candidates with the same score
        key = (h, r) if mode == "tail" else (r, t)
        for known in positives.get(key, ()):              # filtered setting
            if known == true_node or known not in in_pool:
                continue
            ks = deg.get(known, 0)
            if ks > s:
                greater -= 1
            elif ks == s:
                ties -= 1
        pess.append(max(1, greater + ties + 1))
        rand.append(greater + (ties + 1) / 2 + 0.5)
    return pess, rand


def summarise(ranks):
    r = np.asarray(ranks, dtype=float)
    out = {"MRR": float((1.0 / r).mean()) if len(r) else 0.0}
    for k in HITS_K:
        out[k] = float((r <= k).mean()) if len(r) else 0.0
    return out


def evaluate(test, all_true, train_pos):
    deg_t = Counter(t for _, _, t in train_pos)
    deg_h = Counter(h for h, _, _ in train_pos)
    pos_tail, pos_head = defaultdict(set), defaultdict(set)
    for h, r, t in all_true:
        pos_tail[(h, r)].add(t)
        pos_head[(r, t)].add(h)
    tail_pool = {t for _, _, t in all_true}
    head_pool = {h for h, _, _ in all_true}
    pt, rt = rank_stats(test, tail_pool, deg_t, pos_tail, "tail")
    ph, rh = rank_stats(test, head_pool, deg_h, pos_head, "head")
    return summarise(pt + ph), summarise(rt + rh), deg_h, deg_t


def auc_metrics(test_tensor, all_entities, all_true_arr, seed, deg_h, deg_t):
    samples, labels = negative_sampling_filtered(test_tensor, all_entities, 1, all_true_arr, seed=seed)
    s = samples.numpy()
    score = torch.tensor([np.log1p(deg_h.get(h, 0)) + np.log1p(deg_t.get(t, 0)) for h, _, t in s],
                         dtype=torch.float)
    return binary_auroc(score, labels).item(), binary_auprc(score, labels).item()


def line(prefix, auroc, auprc, m):
    hits = {k: m[k] for k in HITS_K}
    return (f"Run 0 | {prefix}Test Auroc: {auroc:.3f}, Test Auprc: {auprc:.3f}, "
            f"Test MRR: {m['MRR']:.3f}, TEST HITS: {hits}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--validation_size", type=float, default=0.1)
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    (_, _, num_entities, num_relations, train_triplets, train_index, _, val_triplets,
     _, test_triplets, train_val_test, _, _, _) = get_dataset(
        args.tsv, args.task, args.validation_size, args.test_size, True, args.split_seed,
        oversample_rate=1, undersample_rate=1.0)

    train_pos = [tuple(x) for x in torch.as_tensor(train_triplets).tolist()]
    test = [tuple(x) for x in torch.as_tensor(test_triplets).tolist()]
    all_true_t = train_val_test.cpu()
    all_true = [tuple(x) for x in all_true_t.tolist()]
    all_entities = np.arange(num_entities)
    all_true_arr = all_true_t.numpy()

    print(f"[popularity] task={args.task} split_seed={args.split_seed} | train {len(train_pos)} "
          f"| test {len(test)} | entities {num_entities}")
    pess, rand, deg_h, deg_t = evaluate(test, all_true, train_pos)
    auroc, auprc = auc_metrics(torch.as_tensor(test_triplets), all_entities, all_true_arr,
                               args.split_seed + 2000, deg_h, deg_t)
    print(line("", auroc, auprc, pess))
    print(f"Run 0 | RANDOM-TIES MRR: {rand['MRR']:.3f}, HITS: {{1: {rand[1]}, 3: {rand[3]}, 10: {rand[10]}}}")

    cold = cold_start_mask(test_triplets, train_index, num_relations)
    warm = [x for x, c in zip(test, cold.tolist()) if not c]
    print(f"Run 0 | cold-start test triples: {int(cold.sum())}/{len(cold)} ({float(cold.float().mean()):.2%})")
    if warm:
        wp, wr, _, _ = evaluate(warm, all_true, train_pos)
        w_auroc, w_auprc = auc_metrics(torch.tensor(warm), all_entities, all_true_arr,
                                       args.split_seed + 3000, deg_h, deg_t)
        print(line("WARM ", w_auroc, w_auprc, wp))
        print(f"Run 0 | WARM RANDOM-TIES MRR: {wr['MRR']:.3f}")
    print("Run 0 | best_epoch: None | train_time_sec: 0.0")
    print("[i] Completed run 0/1")


if __name__ == "__main__":
    main()
