"""
x0b_embedding_smoothing.py -- block 1, experiment X0b: is the "aggregation makes relatives alike"
mechanism visible in the embeddings? Plan: future_works/04_TODO_blocco1.md, point 0b.

The thesis infers the mechanism from the competitors (the candidates that beat the true target are its
relatives 11.8 times more often than chance for R-GCN). This experiment measures it directly, on every
E1 checkpoint (3 models x 12 seeds per task), with one forward pass each and no training:

  relative contrast   for each test target t: mean cosine similarity between t and its relatives in the
                      candidate pool, minus the mean similarity between t and random pool members.
                      Computed in the space the decoder actually uses, z = e ⊙ r (DistMult: a drug
                      scores t and t' alike for EVERY drug iff z_t ≈ z_t'), and in the raw embedding.
                      Relatives: Task A PPI partners, shared pathway, shared GO molecular function;
                      Task B shared phenotypes, and Mondo parent/child if the X0a cache exists.
  Dirichlet energy    sum over edges of ||x_u/sqrt(d_u) - x_v/sqrt(d_v)||^2 / sum ||x_u||^2, on the PPI
                      graph and on the whole (untyped) training graph: lower = smoother over the graph.
  effective rank      exp(entropy of the normalised singular values) of z over the candidate pool: a
                      collapse of the candidates onto few directions would show here.

Prediction from the thesis: R-GCN and CompGCN have a higher relative contrast and a lower energy than
DistMult. DistMult never sees the graph, so its values are the reference for "no smoothing".

Sanity check per checkpoint (entity ids depend on PYTHONHASHSEED=0): AUROC of the test positives against
type-constrained random negatives must be well above 0.5, otherwise the embeddings are mapped onto the
wrong entities and the run stops.

Read-only with respect to release v1.0.0 (imports drug_eval; modifies nothing).

Usage (repository root, environment gnn, PYTHONHASHSEED=0):
  python experiments/block1/x0b_embedding_smoothing.py --task DTI --check
  python experiments/block1/x0b_embedding_smoothing.py --task DTI --out_dir experiments/block1/results/X0b_v1
"""
import argparse
import datetime
import importlib
import json
import os
import socket
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from x0a_distance_curve import (E1, ROOT, REFERENCE, N_SEEDS, MONDO_CACHE, Hierarchy, Checks, check_inputs,  # noqa: E402
                                load_ranks, read_params, sha256, git_state)

N_RANDOM = 50
N_SANITY = 4000
MIN_SANITY_AUROC = 0.6
RELATIVES = {"DTI": dict(direct=["PPI"], shared=["PROTEIN_PATHWAY", "PROTEIN_GO_FUNCTION"]),
             "TREATS": dict(direct=[], shared=["DISEASE_PHENOTYPE"])}
MODEL_NAME = {"R-GCN": "rgcn", "CompGCN": "compgcn", "DistMult": "distmult"}


# ================================================================================================
def relatives_sets(task, cfg, pool_names):
    """{relation group: {pool node: set of related pool nodes}}."""
    g = pd.read_csv(cfg["tsv"], sep="\t", dtype=str, usecols=[0, 1, 2], compression="zip")
    g.columns = ["head", "interaction", "tail"]
    inpool = set(pool_names)
    out = {}
    for rel in RELATIVES[task]["direct"]:
        s = g[(g.interaction == rel) & g["head"].isin(inpool) & g["tail"].isin(inpool)]
        m = defaultdict(set)
        for a, b in zip(s["head"], s["tail"]):
            if a != b:
                m[a].add(b)
                m[b].add(a)
        out[rel] = m
    for rel in RELATIVES[task]["shared"]:
        s = g[(g.interaction == rel) & g["head"].isin(inpool)]
        members = defaultdict(set)
        for a, attr in zip(s["head"], s["tail"]):
            members[attr].add(a)
        m = defaultdict(set)
        for ms in members.values():
            if 1 < len(ms) <= 500:  # an attribute shared by > 500 pool nodes says nothing about kinship
                for a in ms:
                    m[a] |= ms
        for a in m:
            m[a].discard(a)
        out[f"shared {rel}"] = m
    if task == "TREATS" and os.path.isfile(MONDO_CACHE):
        # same hierarchy as X0a: cycles removed (declared there), then the transitive reduction.
        # v1 skipped this group silently because the raw graph has cycles (fixed in v2).
        H = Hierarchy(pd.read_csv(MONDO_CACHE, sep="\t", dtype=str))
        m = defaultdict(set)
        for c, a in H.g.edges():
            c, a = f"Disease::{c}", f"Disease::{a}"
            if c in inpool and a in inpool:
                m[c].add(a)
                m[a].add(c)
        out["Mondo parent/child"] = m
    return out


def rebuild(task, cfg, ref_df, chk):
    import drug_eval
    params = read_params(next(iter(cfg["models"].values())))
    out = drug_eval.get_dataset_for_drug_eval(
        cfg["tsv"], task, float(params["validation_size"]), float(params["test_size"]), True,
        int(params["split_seed"]), float(params["undersample_rate"]))
    (in_ch, nnpt, n_ent, n_rel, train_t, train_index, feats, val_t, test_t, _, ent2id, rel2id, _) = out
    id2ent = {v: k for k, v in ent2id.items()}
    got = [(id2ent[h], id2ent[t]) for h, _, t in test_t.tolist()]
    chk.add("rebuilt test set == rank dumps", sorted(got) == sorted(zip(ref_df["head"], ref_df["tail"])),
            f"{len(got):,} pairs")
    rel_id = [rid for rn, rid in rel2id.items() if rn.lower() == task.lower()]
    chk.add(f"relation {task} in the graph", len(rel_id) == 1)
    return dict(in_ch=in_ch, nnpt=nnpt, n_ent=n_ent, n_rel=n_rel, train=train_t, train_index=train_index,
                feats=feats, val=val_t, test=test_t, ent2id=ent2id, id2ent=id2ent,
                rel_id=rel_id[0] if rel_id else None, config=params.get("config"))


def embed(D, folder, model_name, run_i, task, device, on_cpu):
    """Entity embeddings and the target-relation vector of one checkpoint (as dump_test_ranks.py does)."""
    import torch
    import drug_eval
    model, *_ = drug_eval.get_model(model_name, task, D["in_ch"], D["nnpt"], D["n_ent"], D["n_rel"],
                                    config=D["config"])
    model.load_state_dict(torch.load(os.path.join(folder, f"{model_name}_run{run_i}.pt"), map_location="cpu"))
    dev = torch.device("cpu") if on_cpu else device
    model = model.to(dev).eval()
    if getattr(model, "device", None) is not None:
        model.device = dev
    with torch.no_grad():
        emb = model({k: (v.to(dev) if v is not None else None) for k, v in D["feats"].items()},
                    D["train_index"].to(dev))
        rel = getattr(model, "final_rel_emb", None)
        if rel is None:
            rel = model.relation_embedding
        r = rel[D["rel_id"]]
        # sanity: test positives against type-constrained random negatives
        test = D["test"].to(dev)
        rng = np.random.default_rng(run_i)
        tails = torch.unique(torch.cat([D["train"][:, 2], D["val"][:, 2], D["test"][:, 2]])).to(dev)
        idx = torch.as_tensor(rng.choice(len(test), min(N_SANITY, len(test)), replace=False), device=dev)
        pos = test[idx]
        neg = pos.clone()
        neg[:, 2] = tails[torch.as_tensor(rng.integers(0, len(tails), len(neg)), device=dev)]
        sp, sn = model.distmult(emb, pos), model.distmult(emb, neg)
    s = torch.cat([sp, sn]).detach().cpu().numpy()
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
    order = s.argsort()
    rk = np.empty(len(s))
    rk[order] = np.arange(1, len(s) + 1)
    auroc = (rk[y == 1].sum() - len(sp) * (len(sp) + 1) / 2) / (len(sp) * len(sn))
    # detach: DistMult returns its embedding table and every model its relation table, both Parameters
    return (emb.detach().cpu().numpy().astype(np.float32), r.detach().cpu().numpy().astype(np.float32),
            float(auroc))


def unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


def contrast(Z, pool_ids, targets, rels, rng):
    """Mean over targets with >= 1 relative of [mean cos(t, relatives) - mean cos(t, random)]."""
    U = unit(Z)
    pos = {e: i for i, e in enumerate(pool_ids)}
    diffs = []
    for t in targets:
        r = [pos[x] for x in rels.get(t, ()) if x in pos]
        if not r:
            continue
        i = pos[t]
        rnd = rng.choice(len(pool_ids), N_RANDOM, replace=False)
        rnd = rnd[rnd != i]
        diffs.append(float(np.mean(U[r] @ U[i]) - np.mean(U[rnd] @ U[i])))
    return float(np.mean(diffs)) if diffs else float("nan"), len(diffs)


def dirichlet(X, edges):
    """Normalised Dirichlet energy of X on an undirected edge list (u, v), u != v."""
    deg = np.bincount(np.r_[edges[:, 0], edges[:, 1]], minlength=len(X)).astype(np.float64)
    s = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1)), 0.0)[:, None]
    num = 0.0
    for a in range(0, len(edges), 500_000):
        e = edges[a:a + 500_000]
        num += float(((X[e[:, 0]] * s[e[:, 0]] - X[e[:, 1]] * s[e[:, 1]]) ** 2).sum())
    used = np.unique(edges)
    return num / float((X[used] ** 2).sum())


def eff_rank(Z):
    sv = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    p = sv / sv.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# ================================================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(E1))
    p.add_argument("--out_dir", default=None)
    p.add_argument("--check", action="store_true")
    p.add_argument("--encode_on_cpu", default="auto", choices=["auto", "yes", "no"],
                   help="forward pass on the CPU (auto: when the GPU has < 12 GB)")
    args = p.parse_args()
    os.chdir(ROOT)
    if not args.check and not args.out_dir:
        p.error("--out_dir is required unless --check")
    task, cfg = args.task, E1[args.task]
    t0 = time.time()
    print(f"[x0b] {datetime.datetime.now():%F %T} task {task} on {socket.gethostname()}"
          f"{' (check only)' if args.check else ''}", flush=True)

    chk = Checks()
    chk.add("PYTHONHASHSEED=0 (entity ids of the checkpoints)", os.environ.get("PYTHONHASHSEED") == "0",
            f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}")
    files = check_inputs(task, cfg, chk)
    for label, folder in cfg["models"].items():
        m = MODEL_NAME[label]
        miss = [i for i in range(N_SEEDS) if not os.path.isfile(os.path.join(folder, f"{m}_run{i}.pt"))]
        chk.add(f"{label}: {N_SEEDS} checkpoints {m}_run0..{N_SEEDS - 1}.pt", not miss, f"missing runs {miss}" if miss else "")
    if not chk.ok:
        raise SystemExit("[x0b] requirements not met (FAIL above): nothing computed")
    ranks = load_ranks(files, chk)
    if ranks is None:
        raise SystemExit("[x0b] rank dumps not aligned")
    ref = ranks[REFERENCE][0]
    D = rebuild(task, cfg, ref, chk)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if device.type == "cuda" else 0
    on_cpu = args.encode_on_cpu == "yes" or (args.encode_on_cpu == "auto" and gb < 12)
    chk.add("device", True, f"{device} ({gb:.0f} GB), forward on {'CPU' if on_cpu else 'GPU'}")
    if not chk.ok:
        raise SystemExit("[x0b] requirements not met")
    if args.check:
        print(f"[x0b] CHECK PASSED: all requirements met for {task} ({time.time() - t0:.0f}s)")
        return

    # ---- model-independent structures ----
    allt = torch.cat([D["train"], D["val"], D["test"]]).numpy()
    pool = np.unique(allt[:, 2])
    pool_names = [D["id2ent"][i] for i in pool]
    targets = sorted(set(ref["tail"]))
    rels = relatives_sets(task, cfg, pool_names)
    rel_any = defaultdict(set)
    for m in rels.values():
        for k, v in m.items():
            rel_any[k] |= v
    groups = dict(rels, **{"any relative": rel_any})
    ti = D["train_index"].numpy() if hasattr(D["train_index"], "numpy") else np.asarray(D["train_index"])
    full = ti[ti[:, 0] != ti[:, 2]][:, [0, 2]]
    full = np.unique(np.sort(full, axis=1), axis=0)
    g = pd.read_csv(cfg["tsv"], sep="\t", dtype=str, usecols=[0, 1, 2], compression="zip")
    g.columns = ["head", "interaction", "tail"]
    ppi = g[g.interaction == "PPI"]
    ppi = np.array([(D["ent2id"][a], D["ent2id"][b]) for a, b in zip(ppi["head"], ppi["tail"]) if a != b])
    print(f"[i] pool {len(pool):,} {cfg['what']}s, {len(targets):,} test {cfg['what']}s; relatives: "
          + ", ".join(f"{k} {sum(1 for t in targets if v.get(t))}" for k, v in groups.items())
          + f"; graph {len(full):,} edges, PPI {len(ppi):,}", flush=True)

    # ---- per checkpoint ----
    rows = []
    for label, folder in cfg["models"].items():
        for i in range(N_SEEDS):
            t1 = time.time()
            E, r, auroc = embed(D, folder, MODEL_NAME[label], i, task, device, on_cpu)
            if auroc < MIN_SANITY_AUROC:
                raise SystemExit(f"[x0b] {label} run {i}: sanity AUROC {auroc:.3f} < {MIN_SANITY_AUROC}: "
                                 "embeddings mapped onto the wrong entities? Nothing written.")
            Zp, Ep = E[pool] * r, E[pool]
            row = dict(task=task, model=label, seed=i, sanity_auroc=auroc)
            for gname, gm in groups.items():
                rng = np.random.default_rng(1000 + i)
                row[f"contrast z | {gname}"], row[f"n | {gname}"] = contrast(Zp, pool_names, targets, gm, rng)
                rng = np.random.default_rng(1000 + i)
                row[f"contrast raw | {gname}"], _ = contrast(Ep, pool_names, targets, gm, rng)
            row["Dirichlet energy | PPI"] = dirichlet(E, ppi)
            row["Dirichlet energy | training graph"] = dirichlet(E, full)
            row["effective rank of z | pool"] = eff_rank(Zp)
            row["effective rank of z / dimension | pool"] = row["effective rank of z | pool"] / Zp.shape[1]
            rows.append(row)
            print(f"[i] {label} run {i}: AUROC {auroc:.3f}, contrast z (any) {row['contrast z | any relative']:+.3f}, "
                  f"E_PPI {row['Dirichlet energy | PPI']:.3f} ({time.time() - t1:.0f}s)", flush=True)

    # ---- tables ----
    from stratified_multiseed import fmt, pval
    df = pd.DataFrame(rows)
    models = list(cfg["models"])
    metrics = [c for c in df.columns if c.startswith(("contrast", "Dirichlet", "effective"))]
    L = [f"# X0b — smoothing of relatives in the embeddings — {task}\n",
         f"Every E1 checkpoint ({N_SEEDS} seeds per model), one forward pass each. Cells: mean ± sd over seeds; "
         f"Δ and p (Welch, two-sided) against {REFERENCE}. z = e ⊙ r_{task}, the vectors the DistMult decoder "
         f"compares. Contrast = mean cos(target, its relatives) − mean cos(target, {N_RANDOM} random "
         f"{cfg['what']}s of the pool), over the {len(targets):,} test {cfg['what']}s with ≥ 1 relative. "
         f"Shared attributes count only when they have ≤ 500 pool members. Generated by "
         f"`experiments/block1/x0b_embedding_smoothing.py`.\n",
         "| metric | n targets | " + " | ".join(models) + " | " +
         " | ".join(f"Δ {m} − {REFERENCE} | p" for m in models if m != REFERENCE) + " |",
         "|" + "---|" * (2 + len(models) + 2 * (len(models) - 1))]
    for mt in metrics:
        ncol = mt.split(" | ")[-1]
        n = int(df[f"n | {ncol}"].iloc[0]) if mt.startswith("contrast") and f"n | {ncol}" in df else ""
        vals = {m: df.loc[df.model == m, mt].values for m in models}
        cells = [mt, f"{n:,}" if n != "" else ""] + [fmt(vals[m]) for m in models]
        for m in models:
            if m != REFERENCE:
                cells += [f"{vals[m].mean() - vals[REFERENCE].mean():+.3f}", pval(vals[m], vals[REFERENCE])]
        L.append("| " + " | ".join(cells) + " |")
    L.append(f"\nSanity AUROC (test positives vs random negatives, {N_SANITY} pairs): " +
             ", ".join(f"{m} {fmt(df.loc[df.model == m, 'sanity_auroc'].values)}" for m in models))

    out = os.path.join(args.out_dir, "summary")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"x0b_{task}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    df.to_csv(os.path.join(out, f"x0b_{task}_per_seed.csv"), index=False)
    commit, dirty = git_state()
    prov = dict(experiment="X0b", task=task, finished=f"{datetime.datetime.now():%F %T}",
                seconds=round(time.time() - t0), host=socket.gethostname(), device=str(device),
                forward_on_cpu=on_cpu, slurm_job=os.environ.get("SLURM_JOB_ID"), git_commit=commit,
                git_dirty_tracked=dirty,
                packages={m: getattr(importlib.import_module(m), "__version__", "") for m in
                          ["numpy", "pandas", "scipy", "torch"]},
                params=dict(n_random=N_RANDOM, n_sanity=N_SANITY, min_sanity_auroc=MIN_SANITY_AUROC,
                            shared_attr_max_members=500, mondo_cache_used=os.path.isfile(MONDO_CACHE)),
                inputs=dict(tsv=cfg["tsv"], tsv_sha256=sha256(cfg["tsv"]), models=cfg["models"]),
                checks=[dict(item=i, ok=o, detail=d) for i, o, d in chk.rows])
    with open(os.path.join(out, f"x0b_{task}_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2, default=str)
    print(f"[x0b] written {out}/x0b_{task}.md (+ _per_seed.csv, _provenance.json) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
