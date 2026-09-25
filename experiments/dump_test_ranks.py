"""
dump_test_ranks.py — per-triple ranks of the test set for a trained model.

E1 reports aggregate metrics only; every stratified question ("does the model work on
poorly connected nodes?", "is the true target beaten by its own paralogues?", "are the
recoverable indications the redundant ones?") needs the rank of each individual test
triple, which the training logs do not keep. This script recomputes them from a saved
checkpoint, with EXACTLY the protocol of E1: type-constrained, filtered, both directions.

It reuses drug_eval.py to rebuild the split, the message-passing graph and the model from
the run's own *_params.json, and src/evaluation_metrics_filtered._compute_ranks to rank,
so the mean of the dumped ranks reproduces the MRR of the run (printed as a check).

Output: one CSV row per test triple, with
    head, tail, head_label, tail_label   entity ids and readable labels
    rank_tail                            rank of the true tail among candidate tails (drug -> target)
    rank_head                            rank of the true head among candidate heads (target -> drug)
    rr_mean                              (1/rank_tail + 1/rank_head)/2, the triple's contribution to MRR
    head_target_deg_train, tail_target_deg_train   target-relation degree in the training split
    head_ctx_deg, tail_ctx_deg           degree in the message-passing graph, target edges excluded
    cold_start                           True if either endpoint has no edge at all in that graph

Usage:
  python experiments/dump_test_ranks.py --model_folder models/dti_pkt_taskA_dti.tsv_20260918_130755 \
      --tsv dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip --task DTI
  python experiments/dump_test_ranks.py --model_folder models/treats_pkt_taskB_treats.tsv_20260920_212827 \
      --tsv dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip --task TREATS

Default output: <model_folder>/test_ranks_<TASK>.csv
"""
import argparse
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import drug_eval
from src.evaluation_metrics_filtered import build_positive_maps, _compute_ranks
from src.utils import set_seed

LABELS_TSV = os.path.join("dataset", "PKT_subgraphs", "node_labels.tsv")

# Entity ids come from the iteration order of Python sets/dicts of strings, so they depend on the
# per-process string hash seed (protocol issue P12). With a different seed the saved embeddings are
# read against a different id assignment and the rankings become noise, silently: the script still
# runs and reports an MRR near zero. experiments/config.sh exports this for every experiment script.
if os.environ.get("PYTHONHASHSEED") != "0":
    raise SystemExit(
        "[!] PYTHONHASHSEED is not 0, so entity ids would not match the ones used in training.\n"
        "    Re-run as:  PYTHONHASHSEED=0 python experiments/dump_test_ranks.py ...\n"
        "    (on Windows PowerShell:  $env:PYTHONHASHSEED=0;  then the same command)")


def load_labels():
    if not os.path.exists(LABELS_TSV):
        return {}
    df = pd.read_csv(LABELS_TSV, sep="\t", dtype=str).fillna("")
    return dict(zip(df["entity"], df["label"]))


def degrees(train_index, num_relations, num_entities, target_rel_id):
    """Context degree (no target edges, no self-loops) per entity id, from the training graph."""
    self_rel = num_relations - 1
    ei = train_index.cpu()
    keep = (ei[:, 1] != self_rel)
    ctx = ei[keep & (ei[:, 1] != target_rel_id) & (ei[:, 1] != target_rel_id + num_relations)]
    any_edge = ei[keep]
    ctx_deg = torch.bincount(torch.cat([ctx[:, 0], ctx[:, 2]]), minlength=num_entities)
    all_deg = torch.bincount(torch.cat([any_edge[:, 0], any_edge[:, 2]]), minlength=num_entities)
    return ctx_deg.numpy(), all_deg.numpy()


def top_competitors(model, emb, test, tail_pool, pos_tail, device, k, batch_rows=None):
    """For each test triple, the k highest-scoring candidate tails that beat the true one.

    Same filtering as the ranking (known positives of the same head are removed), so these are
    exactly the entities responsible for the rank: what the model prefers to the right answer.
    Returned as a list of entity-id lists, aligned with `test`.
    """
    out = []
    cands = tail_pool.to(device)
    n = cands.numel()
    cand_list = cands.tolist()
    cand_to_idx = {c: i for i, c in enumerate(cand_list)}
    if batch_rows is None:   # same budget of scored pairs per GPU call as the ranking (PKT_EVAL_BATCH_ROWS)
        batch_rows = max(1, int(os.environ.get("PKT_EVAL_BATCH_ROWS", "250000")) // n)
    rows = test.cpu().tolist()
    with torch.no_grad():
        for start in range(0, len(rows), batch_rows):
            chunk = rows[start:start + batch_rows]
            B = len(chunk)
            kt = torch.tensor(chunk, dtype=torch.long, device=device)
            triples = torch.stack([kt[:, 0].repeat_interleave(n),
                                   kt[:, 1].repeat_interleave(n),
                                   cands.repeat(B)], dim=1)
            scores = model.distmult(emb, triples).view(B, n)
            # mask the other known positives in one indexed write (one write per element was the
            # bottleneck on Task B, where a drug has ~40 known indications)
            mr, mc = [], []
            for b, (h_i, r_i, t_i) in enumerate(chunk):
                for known in pos_tail.get((h_i, r_i), ()):
                    if known != t_i and known in cand_to_idx:
                        mr.append(b)
                        mc.append(cand_to_idx[known])
            if mr:
                scores[torch.tensor(mr, device=device), torch.tensor(mc, device=device)] = -float("inf")
            true_scores = scores[torch.arange(B, device=device),
                                 torch.tensor([cand_to_idx[t] for _, _, t in chunk], device=device)]
            better = scores > true_scores.unsqueeze(1)
            top = torch.topk(scores.masked_fill(~better, -float("inf")), min(k, n), dim=1)
            for b in range(B):
                ids = [cand_list[i] for i, v in zip(top.indices[b].tolist(), top.values[b].tolist())
                       if v != -float("inf")]
                out.append(ids)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_folder", required=True)
    p.add_argument("--tsv", required=True)
    p.add_argument("--task", required=True, help="target relation: DTI or TREATS")
    p.add_argument("--out", default=None, help="output CSV (default: <model_folder>/test_ranks_<TASK>.csv)")
    p.add_argument("--competitors", type=int, default=10,
                   help="also store the k top-scoring candidates that beat the true tail (0 = off)")
    p.add_argument("--run", default="best",
                   help="which seed: 'best' (highest test MRR, the default), a run index, or 'all'. "
                        "With a run index or 'all' the output goes to "
                        "<model_folder>/ranks/test_ranks_<TASK>_run<i>.csv; the data is rebuilt once")
    p.add_argument("--encode_on_cpu", action="store_true",
                   help="compute the node embeddings (one forward pass) on the CPU, then rank on the GPU: "
                        "for a display GPU whose watchdog kills long kernels (CompGCN on Task B locally)")
    p.add_argument("--skip_existing", action="store_true",
                   help="with --run all: skip the seeds whose rank file already exists (resume)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    device = drug_eval.device
    set_seed(drug_eval.BASE_SEED)

    model_path, model_name, train_params = drug_eval.resolve_model_folder(args.model_folder)
    selected_run = int(train_params.get("_selected_run", 0))
    if args.run == "best":
        jobs = [(selected_run, model_path, args.out)]
    else:
        runs = sorted(int(f[len(model_name) + 4:-3]) for f in os.listdir(args.model_folder)
                      if f.startswith(f"{model_name}_run") and f.endswith(".pt")
                      and f[len(model_name) + 4:-3].isdigit()) if args.run == "all" else [int(args.run)]
        os.makedirs(os.path.join(args.model_folder, "ranks"), exist_ok=True)
        jobs = [(i, os.path.join(args.model_folder, f"{model_name}_run{i}.pt"),
                 os.path.join(args.model_folder, "ranks", f"test_ranks_{args.task}_run{i}.csv"))
                for i in runs]
        if args.skip_existing:
            done = [j[0] for j in jobs if os.path.exists(j[2])]
            jobs = [j for j in jobs if not os.path.exists(j[2])]
            if done:
                print(f"[i] skipping runs already dumped: {done}")
            if not jobs:
                print("[i] nothing to do")
                return
    split_seed = train_params.get("split_seed")
    data_seed = int(split_seed) if split_seed is not None else drug_eval.BASE_SEED + selected_run
    undersample_rate = float(train_params.get("undersample_rate", 0.5))
    validation_size = float(train_params.get("validation_size", 0.1))
    test_size = float(train_params.get("test_size", 0.2))
    config_name = train_params.get("config", drug_eval.CONFIG_NAME)
    print(f"[i] {model_name} runs {[j[0] for j in jobs]} | config {config_name} | split_seed {data_seed} | "
          f"undersample {undersample_rate} | val/test {validation_size}/{test_size}")

    t0 = time.time()
    (in_channels_dict, num_nodes_per_type, num_entities, num_relations,
     train_triplets, train_index, feats, val_triplets, test_triplets,
     edge_index, ent2id, relation2id, all_nodes_per_type) = drug_eval.get_dataset_for_drug_eval(
        args.tsv, args.task, validation_size, test_size, args.quiet, data_seed, undersample_rate)
    print(f"[i] data rebuilt in {time.time() - t0:.1f}s | "
          f"train/val/test = {len(train_triplets)}/{len(val_triplets)}/{len(test_triplets)}")

    rel_id = None
    for rn, rid in relation2id.items():
        if rn.lower().replace(" ", "_") == args.task.lower().replace(" ", "_"):
            rel_id = rid
    if rel_id is None:
        raise KeyError(f"relation {args.task} not in {list(relation2id)}")


    # --- shared by every run: the split, the candidate pools and the degrees do not depend on the seed ---
    train_index = train_index.to(device)
    feats = {k: (v.to(device) if v is not None else None) for k, v in feats.items()}
    all_target = torch.cat([train_triplets, val_triplets, test_triplets], dim=0).to(device)
    test = test_triplets.to(device)
    pos_tail, pos_head = build_positive_maps(all_target)
    tail_pool = torch.unique(all_target[:, 2])
    head_pool = torch.unique(all_target[:, 0])
    print(f"[i] candidate pools: {tail_pool.numel()} tails, {head_pool.numel()} heads")

    id2ent = {v: k for k, v in ent2id.items()}
    labels = load_labels()
    ctx_deg, all_deg = degrees(train_index, num_relations, num_entities, rel_id)
    tgt_deg = Counter()
    for h, _, t in train_triplets.tolist():
        tgt_deg[h] += 1
        tgt_deg[t] += 1

    expected = {}
    mfile = os.path.join(args.model_folder, f"{model_name}_metrics.json")
    if os.path.exists(mfile):
        import json
        with open(mfile) as f:
            expected = {i: r["MRR"] for i, r in enumerate(json.load(f).get("individual_runs", []))}

    for run_i, pt_path, out in jobs:
        print(f"\n[i] ---- run {run_i}: {os.path.basename(pt_path)}")
        model, *_ = drug_eval.get_model(model_name, args.task, in_channels_dict, num_nodes_per_type,
                                        num_entities, num_relations, config=config_name)
        model.load_state_dict(torch.load(pt_path, map_location=device))
        if args.encode_on_cpu:
            cpu = torch.device("cpu")
            model = model.to(cpu).eval()
            dev_attr = getattr(model, "device", None)   # the encoders also keep a device attribute
            if dev_attr is not None:
                model.device = cpu
            with torch.no_grad():
                emb = model({k: (v.to(cpu) if v is not None else None) for k, v in feats.items()},
                            train_index.to(cpu))
            emb = emb.to(device)
            model = model.to(device)
            if dev_attr is not None:
                model.device = dev_attr
            for name, val in list(vars(model).items()):   # tensors cached by forward (CompGCN final_rel_emb)
                if torch.is_tensor(val):
                    setattr(model, name, val.to(device))
        else:
            model = model.to(device).eval()
            with torch.no_grad():
                emb = model(feats, train_index)

        t0 = time.time()
        tail_ranks, skip_t = _compute_ranks(model, emb, test, tail_pool, pos_tail,
                                            None, device, "tail", False)
        head_ranks, skip_h = _compute_ranks(model, emb, test, head_pool, pos_head,
                                            None, device, "head", False)
        print(f"[i] ranked {len(test)} test triples in {time.time() - t0:.1f}s "
              f"(skipped: {skip_t} tail, {skip_h} head)")
        if skip_t or skip_h:
            raise SystemExit("[!] some test endpoints fell outside the candidate pool: rows would be "
                             "misaligned with the test triples. This should not happen when the pools "
                             "come from train+val+test, so investigate before trusting the output.")

        comps = None
        if args.competitors > 0:
            t0 = time.time()
            comps = top_competitors(model, emb, test, tail_pool, pos_tail, device, args.competitors)
            print(f"[i] top-{args.competitors} competitors per triple in {time.time() - t0:.1f}s")

        rows = []
        for i, ((h, _, t), rt, rh) in enumerate(zip(test.cpu().tolist(), tail_ranks, head_ranks)):
            he, te = id2ent.get(h, str(h)), id2ent.get(t, str(t))
            rows.append(dict(
                head=he, tail=te,
                head_label=labels.get(he, ""), tail_label=labels.get(te, ""),
                rank_tail=rt, rank_head=rh, rr_mean=(1.0 / rt + 1.0 / rh) / 2,
                head_target_deg_train=tgt_deg[h], tail_target_deg_train=tgt_deg[t],
                head_ctx_deg=int(ctx_deg[h]), tail_ctx_deg=int(ctx_deg[t]),
                cold_start=bool(all_deg[h] == 0 or all_deg[t] == 0),
                competitors=";".join(id2ent.get(c, str(c)) for c in comps[i]) if comps else "",
            ))
        df = pd.DataFrame(rows)

        out = out or os.path.join(args.model_folder, f"test_ranks_{args.task}.csv")
        df.to_csv(out, index=False)

        mrr = df["rr_mean"].mean()
        exp = expected.get(run_i)
        check = "" if exp is None else (f" | run's test MRR {exp:.4f} -> "
                                        + ("OK" if abs(mrr - exp) < 1e-3 else "MISMATCH"))
        print(f"[i] MRR from the dumped ranks: {mrr:.4f}{check}")
        for k in (1, 3, 10):
            hits = ((df["rank_tail"] <= k).mean() + (df["rank_head"] <= k).mean()) / 2
            print(f"    Hits@{k}: {hits:.4f}")
        print(f"    cold-start triples: {df['cold_start'].sum()} / {len(df)} "
              f"({df['cold_start'].mean():.1%})")
        print(f"[i] written: {out}  ({len(df)} rows)")
        del model, emb
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
