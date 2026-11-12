"""
x0c_heuristics.py -- block 1, experiment X0c: classical link-prediction heuristics, no learning.
Plan: future_works/04_TODO_blocco1.md, point 0c.

How far does graph topology alone go, under exactly the E1 protocol? Every heuristic scores a
(drug, candidate) pair from the TRAINING message-passing graph only (context edges + training target
edges; validation and test target edges are not in it), and is ranked with the E1 rules:
type-constrained candidate pools (the heads / tails of the target relation), every known positive
filtered out, rank = #candidates with score >= the true score (ties count against, as for the
models), MRR = mean over both directions. Because count-based scores tie a lot, the expected rank
under random tie-breaking is reported as well (as in experiments/baseline_popularity.py).

Heuristics (graph-theoretic, untyped unless stated):
  popularity   training target degree of the candidate. Reproduces experiments/baseline_popularity.py
               (MRR 0.041 / 0.047): the check that this ranking code is the E1 one.
  CN, AA, RA   common neighbours, Adamic-Adar, resource allocation (2-hop paths).
  Katz-3       truncated Katz index, sum over path lengths 1..3 of beta^l (A^l), beta chosen on validation.
  PPR          personalised PageRank from the anchor, restart alpha chosen on validation.
  relatives    typed: "relatives of the known targets/indications", the heuristic a pharmacologist would
               use. Task A: score(d, t) = sum over the drug's training targets k of rel(k, t), with
               rel = PPI edge + resource-allocation weight of shared pathways and shared GO molecular
               functions. Task B: rel = resource-allocation weight of shared phenotypes (the task graph
               has no disease-disease edges; the Mondo hierarchy is NOT used, so that every heuristic
               sees exactly what the models saw).
  CF-drug      typed collaborative filtering on the training target edges only, the non-learned
               counterpart of an embedding-only factorisation: score(d, t) = sum over the other drugs d'
               of cos(d, d') T[d', t], with cos the cosine of the drugs' training target vectors
               (meta-path drug -> target' <- drug' -> target).
  CF-target    the same on the target side: score(d, t) = sum over t' of T[d, t'] cos(t', t).
Note: on Task B no compound-disease path of length 2 exists in the graph (the neighbour types of the two
are disjoint), so CN, AA and RA give 0 to every candidate there, and Katz-3 reduces to its length-3 term
(its ranking does not depend on beta).
Beta and alpha are selected on the validation target edges (a fixed random subsample of at most
--val_max triples), with the same graph, as the models' hyperparameters were.

Output: per-heuristic per-triple ranks aligned with the E1 rank dumps (so the strata of X0a apply),
and a summary next to the E1 models (12-seed means from the rank dumps).

Read-only with respect to release v1.0.0 (imports drug_eval; modifies nothing).

Usage (repository root, environment gnn):
  python experiments/block1/x0c_heuristics.py --task DTI --check
  python experiments/block1/x0c_heuristics.py --task DTI --out_dir experiments/block1/results/X0c_v1
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
from x0a_distance_curve import (E1, ROOT, REFERENCE, N_SEEDS, Checks, check_inputs, load_ranks,  # noqa: E402
                                read_params, sha256, git_state)

# Grids widened after a first (aborted, unreported) pass picked the edge value of both narrower grids
# (beta 0.1 of 0.001-0.1, alpha 0.5 of 0.15-0.5): the selected value must be interior, as for the models.
# v2: alpha 0.99 added (v1 picked 0.95 on Task B); an edge value is flagged only when it beats its
# neighbour by more than PLATEAU (otherwise the curve is flat there and the edge is not a truncation).
KATZ_BETAS = (0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
PPR_ALPHAS = (0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99)
PLATEAU = 0.001
PPR_ITERS = 30
HITS_K = (1, 3, 10)
CTX_REL = {"DTI": dict(direct=["PPI"], shared=["PROTEIN_PATHWAY", "PROTEIN_GO_FUNCTION"]),
           "TREATS": dict(direct=[], shared=["DISEASE_PHENOTYPE"])}


# ================================================================================================
# data
# ================================================================================================
def rebuild(task, cfg, ref_df, chk):
    """Split, training graph and name maps, by the same function the E1 rank dumps used."""
    import drug_eval
    params = read_params(next(iter(cfg["models"].values())))
    out = drug_eval.get_dataset_for_drug_eval(
        cfg["tsv"], task, float(params["validation_size"]), float(params["test_size"]), True,
        int(params["split_seed"]), float(params["undersample_rate"]))
    num_entities, train_t, train_index, val_t, test_t, ent2id = out[2], out[4], out[5], out[7], out[8], out[10]
    id2ent = {v: k for k, v in ent2id.items()}
    test = test_t.numpy()
    names = list(zip(ref_df["head"], ref_df["tail"]))
    got = [(id2ent[h], id2ent[t]) for h, _, t in test.tolist()]
    if not chk.add("rebuilt test triples == rank dumps (same order)", got == names,
                   f"{len(got):,} vs {len(names):,}"):
        if sorted(got) == sorted(names):
            order = {p: i for i, p in enumerate(got)}
            test = test[[order[p] for p in names]]
            chk.add("same test set, reordered to the rank dumps", True)
    return dict(num_entities=num_entities, train=train_t.numpy(), val=val_t.numpy(), test=test,
                train_index=train_index.numpy() if hasattr(train_index, "numpy") else np.asarray(train_index),
                ent2id=ent2id, id2ent=id2ent)


def adjacency(train_index, n):
    """Untyped, undirected, binary adjacency of the training message-passing graph (no self-loops)."""
    import scipy.sparse as sp
    h, t = train_index[:, 0], train_index[:, 2]
    keep = h != t
    h, t = h[keep], t[keep]
    A = sp.coo_matrix((np.ones(2 * len(h), dtype=np.float32), (np.r_[h, t], np.r_[t, h])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    A.eliminate_zeros()
    return A


def typed_relatives(task, cfg, D, pool):
    """Dense pool x pool 'relatives' weight (Task A: PPI + RA over pathway / GO function; Task B: RA over phenotypes)."""
    g = pd.read_csv(cfg["tsv"], sep="\t", dtype=str, usecols=[0, 1, 2], compression="zip")
    g.columns = ["head", "interaction", "tail"]
    idx = {e: i for i, e in enumerate(D["id2ent"][x] for x in pool)}
    R = np.zeros((len(pool), len(pool)), dtype=np.float32)
    for rel in CTX_REL[task]["direct"]:
        s = g[g.interaction == rel]
        for a, b in zip(s["head"], s["tail"]):
            i, j = idx.get(a), idx.get(b)
            if i is not None and j is not None and i != j:
                R[i, j] += 1.0
                R[j, i] += 1.0
    for rel in CTX_REL[task]["shared"]:
        s = g[g.interaction == rel]
        members = defaultdict(list)
        deg = s.groupby("tail").size()
        for a, attr in zip(s["head"], s["tail"]):
            i = idx.get(a)
            if i is not None:
                members[attr].append(i)
        for attr, ms in members.items():
            if len(ms) > 1:
                w = 1.0 / float(deg[attr])
                ms = np.asarray(ms)
                R[np.ix_(ms, ms)] += w
    np.fill_diagonal(R, 0.0)
    return R


# ================================================================================================
# scoring and ranking
# ================================================================================================
class Scorer:
    """Scores of anchors against a candidate pool, by propagation on the adjacency (torch, GPU if any)."""

    def __init__(self, A, device):
        import torch
        self.torch, self.dev = torch, device
        A = A.tocsr()
        self.n = A.shape[0]
        deg = np.asarray(A.sum(1)).ravel().astype(np.float32)
        self.deg = deg
        self.A = torch.sparse_csr_tensor(torch.from_numpy(A.indptr.astype(np.int64)),
                                         torch.from_numpy(A.indices.astype(np.int64)),
                                         torch.from_numpy(A.data.astype(np.float32)),
                                         size=A.shape).to(device)
        with np.errstate(divide="ignore"):
            self.inv_deg = torch.from_numpy(np.where(deg > 0, 1.0 / deg, 0.0).astype(np.float32)).to(device)
            self.inv_log = torch.from_numpy(np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
                                            .astype(np.float32)).to(device)

    def onehot(self, anchors):
        X = self.torch.zeros((self.n, len(anchors)), dtype=self.torch.float32, device=self.dev)
        X[self.torch.as_tensor(anchors, device=self.dev), self.torch.arange(len(anchors), device=self.dev)] = 1.0
        return X

    def mm(self, X):
        return self.torch.sparse.mm(self.A, X)

    def scores(self, name, anchors, cands, param=None):
        """[len(anchors), len(cands)] numpy array."""
        X = self.onehot(anchors)
        c = self.torch.as_tensor(cands, device=self.dev)
        if name in ("CN", "AA", "RA"):
            Y = self.mm(X)
            if name == "AA":
                Y = Y * self.inv_log[:, None]
            elif name == "RA":
                Y = Y * self.inv_deg[:, None]
            S = self.mm(Y)[c]
        elif name == "Katz-3":
            Y1 = self.mm(X)
            Y2 = self.mm(Y1)
            Y3 = self.mm(Y2)
            S = (param * Y1 + param ** 2 * Y2 + param ** 3 * Y3)[c]
        elif name == "PPR":
            r = X.clone()
            for _ in range(PPR_ITERS):  # r = alpha x + (1 - alpha) A D^-1 r   (random walk, undirected)
                r = param * X + (1 - param) * self.mm(r * self.inv_deg[:, None])
            S = r[c]
        else:
            raise ValueError(name)
        return S.T.cpu().numpy()


def rank_block(S, true_idx, filt_rows, filt_cols):
    """Pessimistic (E1) and random-tie ranks of the true candidate in each row of S."""
    S = S.astype(np.float64, copy=True)
    tv = S[np.arange(len(S)), true_idx]
    if len(filt_rows):
        S[filt_rows, filt_cols] = -np.inf
    S[np.arange(len(S)), true_idx] = -np.inf  # the true one is counted separately
    greater = (S > tv[:, None]).sum(1)
    ties = (S == tv[:, None]).sum(1)
    return greater + ties + 1, greater + ties / 2.0 + 1.0


def evaluate(score_fn, triples, pools, positives, batch=256):
    """Ranks of every triple, tail then head direction; triples whose true node is not in the pool are skipped."""
    out = {}
    for mode in ("tail", "head"):
        pool = pools[mode]
        cidx = {c: i for i, c in enumerate(pool.tolist())}
        anchors = triples[:, 0] if mode == "tail" else triples[:, 2]
        truth = triples[:, 2] if mode == "tail" else triples[:, 0]
        pess = np.full(len(triples), np.nan)
        rand = np.full(len(triples), np.nan)
        uniq = np.unique(anchors)
        for s in range(0, len(uniq), batch):
            ub = uniq[s:s + batch]
            S = score_fn(mode, ub, pool)
            row_of = {a: i for i, a in enumerate(ub.tolist())}
            sel = np.where(np.isin(anchors, ub))[0]
            sel = np.array([i for i in sel if truth[i] in cidx])
            if not len(sel):
                continue
            rows = np.array([row_of[a] for a in anchors[sel]])
            Sb = S[rows]
            ti = np.array([cidx[x] for x in truth[sel]])
            fr, fc = [], []
            for b, i in enumerate(sel):
                for k in positives[mode].get(int(anchors[i]), ()):
                    j = cidx.get(k)
                    if j is not None and k != truth[i]:
                        fr.append(b)
                        fc.append(j)
            p, r = rank_block(Sb, ti, np.array(fr, dtype=int), np.array(fc, dtype=int))
            pess[sel], rand[sel] = p, r
        out[mode] = (pess, rand)
    return out


def mrr(ranks):
    both = np.r_[ranks["tail"], ranks["head"]]
    both = both[~np.isnan(both)]
    return float((1.0 / both).mean()) if len(both) else float("nan")


# ================================================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(E1))
    p.add_argument("--out_dir", default=None, help="version folder, e.g. experiments/block1/results/X0c_v1")
    p.add_argument("--check", action="store_true", help="verify the requirements only; write nothing")
    p.add_argument("--val_max", type=int, default=3000, help="validation triples used to pick beta / alpha")
    p.add_argument("--batch", type=int, default=256, help="anchors propagated together")
    p.add_argument("--cpu", action="store_true", help="do not use the GPU")
    args = p.parse_args()
    os.chdir(ROOT)
    if not args.check and not args.out_dir:
        p.error("--out_dir is required unless --check")
    task, cfg = args.task, E1[args.task]
    t0 = time.time()
    print(f"[x0c] {datetime.datetime.now():%F %T} task {task} on {socket.gethostname()}"
          f"{' (check only)' if args.check else ''}", flush=True)

    chk = Checks()
    files = check_inputs(task, cfg, chk)
    if not chk.ok:
        raise SystemExit("[x0c] requirements not met (FAIL above): nothing computed")
    ranks = load_ranks(files, chk)
    if ranks is None:
        raise SystemExit("[x0c] rank dumps not aligned: nothing computed")
    ref = ranks[REFERENCE][0]
    D = rebuild(task, cfg, ref, chk)
    if not chk.ok:
        raise SystemExit("[x0c] the rebuilt split does not match the rank dumps: nothing computed")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    chk.add("device", True, str(device))
    if args.check:
        print(f"[x0c] CHECK PASSED: all requirements met for {task} ({time.time() - t0:.0f}s)")
        return

    # ---- graph, pools, filters ----
    A = adjacency(D["train_index"], D["num_entities"])
    print(f"[i] training graph: {A.shape[0]:,} nodes, {A.nnz // 2:,} undirected edges", flush=True)
    allt = np.r_[D["train"], D["val"], D["test"]]
    pools = {"tail": np.unique(allt[:, 2]), "head": np.unique(allt[:, 0])}
    positives = {"tail": defaultdict(set), "head": defaultdict(set)}
    for h, _, t in allt.tolist():
        positives["tail"][h].add(t)
        positives["head"][t].add(h)
    sc = Scorer(A, device)

    # popularity (= experiments/baseline_popularity.py) and typed relatives, both pool-level
    tdeg_t = np.bincount(D["train"][:, 2], minlength=D["num_entities"]).astype(np.float32)
    tdeg_h = np.bincount(D["train"][:, 0], minlength=D["num_entities"]).astype(np.float32)
    R = typed_relatives(task, cfg, D, pools["tail"])
    tpos = {c: i for i, c in enumerate(pools["tail"].tolist())}
    hpos = {c: i for i, c in enumerate(pools["head"].tolist())}
    T = np.zeros((len(pools["head"]), len(pools["tail"])), dtype=np.float32)  # training target edges, head pool x tail pool
    for h, _, t in D["train"].tolist():
        T[hpos[h], tpos[t]] = 1.0
    # collaborative filtering on the training target edges (cosine similarities, self excluded)
    nr = np.linalg.norm(T, axis=1, keepdims=True)
    Tn = T / np.where(nr > 0, nr, 1.0)
    S_drug = Tn @ Tn.T
    np.fill_diagonal(S_drug, 0.0)
    nc = np.linalg.norm(T, axis=0, keepdims=True)
    Tc = T / np.where(nc > 0, nc, 1.0)
    S_tgt = Tc.T @ Tc
    np.fill_diagonal(S_tgt, 0.0)
    CF = {"CF-drug": S_drug @ T, "CF-target": T @ S_tgt}   # head pool x tail pool

    def fn(name, param=None):
        def f(mode, anchors, pool):
            if name == "popularity":
                d = tdeg_t if mode == "tail" else tdeg_h
                return np.tile(d[pool], (len(anchors), 1))
            if name == "relatives":
                # score(d, t) = sum over the drug's training targets k of rel(k, t) = (T R)[d, t]
                if mode == "tail":  # anchors = drugs, pool = tail pool
                    return T[[hpos[a] for a in anchors]] @ R
                col = R[:, [tpos[a] for a in anchors]]      # anchors = targets, pool = head pool
                return (T[[hpos[c] for c in pool]] @ col).T
            if name in CF:  # pool-level matrices: rows = head pool, columns = tail pool
                M = CF[name]
                if mode == "tail":
                    return M[[hpos[a] for a in anchors]][:, [tpos[c] for c in pool]]
                return M[[hpos[c] for c in pool]][:, [tpos[a] for a in anchors]].T
            return sc.scores(name, anchors, pool, param)
        return f

    # ---- select beta / alpha on validation ----
    rng = np.random.default_rng(0)
    val = D["val"] if len(D["val"]) <= args.val_max else D["val"][rng.choice(len(D["val"]), args.val_max, replace=False)]
    selected, sel_lines = {}, []
    for name, grid in (("Katz-3", KATZ_BETAS), ("PPR", PPR_ALPHAS)):
        vals = []
        for v in grid:
            m = mrr({k: x[0] for k, x in evaluate(fn(name, v), val, pools, positives, args.batch).items()})
            vals.append(m)
            sel_lines.append(f"| {name} | {v} | {m:.4f} |")
            print(f"[i] validation {name} {v}: MRR {m:.4f} ({time.time() - t0:.0f}s)", flush=True)
        b = int(np.argmax(vals))
        selected[name] = grid[b]
        if b in (0, len(grid) - 1):
            nb = vals[1] if b == 0 else vals[-2]
            if vals[b] - nb > PLATEAU:
                msg = (f"{name}: selected value {grid[b]} is at the EDGE of the grid and beats its neighbour by "
                       f"{vals[b] - nb:.4f}: the optimum may lie outside")
            else:
                msg = (f"{name}: selected value {grid[b]} is at the edge of a plateau (within {PLATEAU} of its "
                       f"neighbour): not a truncation")
            print(f"[!] {msg}", flush=True)
            sel_lines.append(f"| {name} | note | {msg} |")

    # ---- test ----
    heur = [("popularity", None), ("CN", None), ("AA", None), ("RA", None),
            ("Katz-3", selected["Katz-3"]), ("PPR", selected["PPR"]), ("relatives", None),
            ("CF-drug", None), ("CF-target", None)]
    test = D["test"]
    rows, per_triple = [], {}
    cold = ref["cold_start"].values
    for name, v in heur:
        r = evaluate(fn(name, v), test, pools, positives, args.batch)
        pess = {k: x[0] for k, x in r.items()}
        rand = {k: x[1] for k, x in r.items()}
        warm = {k: x[~cold] for k, x in pess.items()}
        both = np.r_[pess["tail"], pess["head"]]
        rows.append(dict(heuristic=name + (f" ({'β' if name == 'Katz-3' else 'α'}={v})" if v is not None else ""),
                         MRR=mrr(pess), MRR_random_ties=mrr(rand), MRR_warm=mrr(warm),
                         **{f"Hits@{k}": float(np.nanmean(both <= k)) for k in HITS_K}))
        per_triple[name] = pd.DataFrame({"head": ref["head"], "tail": ref["tail"],
                                         "rank_tail": pess["tail"], "rank_head": pess["head"],
                                         "rr_mean": (1 / pess["tail"] + 1 / pess["head"]) / 2,
                                         "rr_mean_random_ties": (1 / rand["tail"] + 1 / rand["head"]) / 2})
        print(f"[i] test {rows[-1]['heuristic']}: MRR {rows[-1]['MRR']:.4f} "
              f"(random ties {rows[-1]['MRR_random_ties']:.4f}) ({time.time() - t0:.0f}s)", flush=True)

    # ---- E1 models, for reference, and strata ----
    e1 = {m: np.array([df["rr_mean"].mean() for df in dfs]) for m, dfs in ranks.items()}
    reg = pd.Series("drug has known " + cfg["what"] + "s", index=ref.index)
    reg[ref["head_target_deg_train"] == 0] = "context but no known " + cfg["what"]
    reg[ref["cold_start"]] = "cold start"

    L = [f"# X0c — classical link-prediction heuristics — {task}\n",
         f"{len(ref):,} test triples of the fixed E1 split, E1 protocol (type-constrained filtered ranking, "
         f"rank = #candidates with score ≥ the true one, MRR over both directions). Heuristics see only the "
         f"training message-passing graph ({A.shape[0]:,} nodes, {A.nnz // 2:,} undirected edges, untyped). "
         f"β and α chosen on {len(val):,} validation triples. Generated by `experiments/block1/x0c_heuristics.py`.\n",
         "## Test\n", "| method | MRR | MRR (random ties) | warm MRR | Hits@1 | Hits@3 | Hits@10 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['heuristic']} | {r['MRR']:.3f} | {r['MRR_random_ties']:.3f} | {r['MRR_warm']:.3f} | "
                 f"{r['Hits@1']:.3f} | {r['Hits@3']:.3f} | {r['Hits@10']:.3f} |")
    for m, v in e1.items():
        L.append(f"| *E1 {m} ({N_SEEDS} seeds)* | {v.mean():.3f} ± {v.std(ddof=1):.3f} | | | | | |")
    L += ["\n## By supervision regime (MRR)\n",
          "| regime | n | " + " | ".join(n for n, _ in heur) + " | " + " | ".join(f"E1 {m}" for m in e1) + " |",
          "|" + "---|" * (2 + len(heur) + len(e1))]
    for g in [f"drug has known {cfg['what']}s", f"context but no known {cfg['what']}", "cold start"]:
        mask = (reg == g).values
        if not mask.any():
            continue
        cells = [f"{per_triple[n].loc[mask, 'rr_mean'].mean():.3f}" for n, _ in heur]
        cells += [f"{np.mean([df.loc[mask, 'rr_mean'].mean() for df in ranks[m]]):.3f}" for m in e1]
        L.append(f"| {g} | {int(mask.sum()):,} | " + " | ".join(cells) + " |")
    L += ["\n## Validation (selection of β and α)\n", "| heuristic | value | validation MRR |", "|---|---|---|"] + sel_lines

    out = os.path.join(args.out_dir, "summary")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"x0c_{task}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(out, f"x0c_{task}.csv"), index=False)
    rk = os.path.join(args.out_dir, "ranks")
    os.makedirs(rk, exist_ok=True)
    for n, df in per_triple.items():
        df.to_csv(os.path.join(rk, f"x0c_{task}_{n}.csv"), index=False)
    commit, dirty = git_state()
    prov = dict(experiment="X0c", task=task, finished=f"{datetime.datetime.now():%F %T}",
                seconds=round(time.time() - t0), host=socket.gethostname(), device=str(device),
                slurm_job=os.environ.get("SLURM_JOB_ID"), git_commit=commit, git_dirty_tracked=dirty,
                packages={m: getattr(importlib.import_module(m), "__version__", "") for m in
                          ["numpy", "pandas", "scipy", "torch"]},
                params=dict(katz_betas=KATZ_BETAS, ppr_alphas=PPR_ALPHAS, ppr_iters=PPR_ITERS, plateau=PLATEAU,
                            val_max=args.val_max, val_used=len(val), selected=selected),
                inputs=dict(tsv=cfg["tsv"], tsv_sha256=sha256(cfg["tsv"]), models=cfg["models"]),
                checks=[dict(item=i, ok=o, detail=d) for i, o, d in chk.rows])
    with open(os.path.join(out, f"x0c_{task}_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2, default=str)
    print(f"[x0c] written {out}/x0c_{task}.md (+ .csv, _provenance.json) and {rk}/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
