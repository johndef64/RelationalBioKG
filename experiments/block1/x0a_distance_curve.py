"""
x0a_distance_curve.py -- block 1, experiment X0a: MRR as a function of how far a held-out pair lies
from the supervision the drug already has. Contains step 0 of X7 (redundancy of the Mondo hierarchy
in Task B). Plan: future_works/04_TODO_blocco1.md, point 0a.

No training, no GPU. Inputs:
  * the per-seed rank dumps of the twelve E1 checkpoints of each model,
        <model folder>/ranks/test_ranks_<TASK>_run<i>.csv   (written by experiments/dump_test_ranks.py)
    plus the <model>_params.json of each folder (the checkpoints themselves are NOT needed);
  * the task graph (dataset/PKT_subgraphs/pkt_task*.tsv.zip);
  * Task B only: dataset/PKT/edges.zip, for the Mondo hierarchy.

The E1 split is rebuilt with drug_eval.get_dataset_for_drug_eval, the same function the rank dump
used, and is checked against the dumps (same test pairs, same training degree of every endpoint)
before anything is computed. For every test pair (drug d, target t) the script measures how far t lies
from the targets d already has in the training split:

  Task A (DTI)     hops in the protein-protein interaction graph of the task graph (relation PPI),
                   multi-source BFS from the drug's training targets, up to --max_hops.
  Task B (TREATS)  steps in the Mondo hierarchy: rdf:type edges between two Mondo classes in the raw
                   PheKnowLator graph (child -> ancestor; they include non-direct ancestors, so the
                   transitive reduction is taken first). Distance between two diseases = fewest steps
                   up from each to a common ancestor: 1 = parent or child of a known indication,
                   2 = sibling or grandparent, and so on. A lineal connection (one disease is an
                   ancestor of the other) always counts; a collateral one counts only through a common
                   ancestor with at most --max_lca_desc descendants, so that top-level grouping
                   classes ("disease by anatomical system") do not connect everything to everything.
                   The curve without this cap is reported as well.

Then, for every stratum, the MRR of each model over all 12 seeds (mean +- sd, Welch's two-sided test
against DistMult over seeds, as in E1 and experiments/stratified_multiseed.py). Main metric: rr_mean,
the per-triple contribution to the E1 MRR (both directions); the drug -> target direction alone is
reported too.

Read-only with respect to release v1.0.0: it imports drug_eval and experiments/stratified_multiseed
and modifies nothing.

Usage (repository root, environment gnn):
  python experiments/block1/x0a_distance_curve.py --task DTI    --check
  python experiments/block1/x0a_distance_curve.py --task DTI    --out_dir experiments/block1/results/X0a_v1
  python experiments/block1/x0a_distance_curve.py --task TREATS --out_dir experiments/block1/results/X0a_v1
--check verifies every requirement (files, packages, alignment of the 36 rank dumps, exact
reconstruction of the split, Mondo source) and writes nothing.
"""
import argparse
import datetime
import hashlib
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict, deque

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --------------------------------------------------------------------------------------------------
# The E1 runs this experiment reads (the same folders as experiments/logs/v2/stratified_multiseed_*.md)
# --------------------------------------------------------------------------------------------------
E1 = {
    "DTI": dict(
        tsv="dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip",
        models={"R-GCN": "models/dti_pkt_taskA_dti.tsv_20260918_130755",
                "CompGCN": "models/dti_pkt_taskA_dti.tsv_20260918_135629",
                "DistMult": "models/dti_pkt_taskA_dti.tsv_20260918_153101"},
        e1_mrr={"R-GCN": 0.408, "CompGCN": 0.352, "DistMult": 0.407},
        what="target", ctx_rel="PPI"),
    "TREATS": dict(
        tsv="dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip",
        models={"R-GCN": "models/treats_pkt_taskB_treats.tsv_20260918_151621",
                "CompGCN": "models/treats_pkt_taskB_treats.tsv_compgcn_20260920_123201",
                "DistMult": "models/treats_pkt_taskB_treats.tsv_distmult_20260920_212827"},
        e1_mrr={"R-GCN": 0.348, "CompGCN": 0.229, "DistMult": 0.692},
        what="indication", ctx_rel=None),
}
REFERENCE = "DistMult"
N_SEEDS = 12
RANK_COLS = ["head", "tail", "rank_tail", "rank_head", "rr_mean",
             "head_target_deg_train", "tail_target_deg_train", "cold_start"]

EDGES_ZIP = "dataset/PKT/edges.zip"
MONDO_CACHE = "experiments/block1/cache/mondo_rdf_type_edges.tsv"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

REG_COLD = "cold start: an endpoint has no edge at all"
REG_NONE = "context but no known {what}"


# ================================================================================================
# Requirements
# ================================================================================================
class Checks:
    def __init__(self):
        self.rows = []

    def add(self, item, ok, detail=""):
        self.rows.append((item, bool(ok), detail))
        print(f"[{'OK' if ok else 'FAIL'}] {item}{(' -- ' + detail) if detail else ''}", flush=True)
        return ok

    @property
    def ok(self):
        return all(r[1] for r in self.rows)


def read_params(folder):
    files = [f for f in os.listdir(folder) if f.endswith("_params.json")] if os.path.isdir(folder) else []
    if not files:
        return None
    with open(os.path.join(folder, files[0])) as f:
        return json.load(f)


def check_inputs(task, cfg, chk):
    """Files and packages. Returns {model: [rank file paths]} (only meaningful if chk.ok)."""
    for mod in ["numpy", "pandas", "scipy", "torch"] + (["ijson", "networkx"] if task == "TREATS" else []):
        try:
            m = importlib.import_module(mod)
            chk.add(f"python package {mod}", True, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            chk.add(f"python package {mod}", False, f"{type(e).__name__}: {e}")
    chk.add(f"task graph {cfg['tsv']}", os.path.isfile(cfg["tsv"]))

    files, split_keys = {}, {}
    for label, folder in cfg["models"].items():
        params = read_params(folder)
        if not chk.add(f"{label}: params json in {folder}", params is not None):
            continue
        if params.get("task") != task:
            chk.add(f"{label}: params task", False, f"{params.get('task')} != {task}")
        split_keys[label] = (params.get("split_seed"), params.get("undersample_rate"),
                             params.get("validation_size"), params.get("test_size"),
                             os.path.basename(str(params.get("tsv"))))
        fl = [os.path.join(folder, "ranks", f"test_ranks_{task}_run{i}.csv") for i in range(N_SEEDS)]
        missing = [os.path.basename(f) for f in fl if not os.path.isfile(f)]
        chk.add(f"{label}: {N_SEEDS} rank dumps in {folder}/ranks", not missing,
                f"missing {missing}" if missing else "")
        files[label] = fl
    if split_keys:
        same = len(set(split_keys.values())) == 1
        chk.add("same split parameters in every model folder", same,
                "; ".join(f"{k}: {v}" for k, v in split_keys.items()))
        sk = next(iter(split_keys.values()))
        chk.add("split is the fixed E1 split", sk[0] is not None and float(sk[1]) == 1.0,
                f"split_seed={sk[0]}, undersample_rate={sk[1]}")
        chk.add("params tsv is this task graph", sk[4] == os.path.basename(cfg["tsv"]), sk[4])

    if task == "TREATS":
        from pkt_source import FILES  # analysis/pkt_source.py: the size and SHA-256 of the release
        size, digest = FILES["edges.zip"]
        if os.path.isfile(MONDO_CACHE):
            chk.add(f"Mondo hierarchy cache {MONDO_CACHE}", True, "edges.zip will not be read")
        elif chk.add(f"PKT source {EDGES_ZIP}", os.path.isfile(EDGES_ZIP),
                     "" if os.path.isfile(EDGES_ZIP) else "run `python analysis/pkt_source.py` on a node "
                     "with internet access"):
            got = os.path.getsize(EDGES_ZIP)
            chk.add("edges.zip size is the release", got == size, f"{got:,} bytes (expected {size:,})")
            chk.add("edges.zip SHA-256 is the release", sha256(EDGES_ZIP) == digest)
    return files


def load_ranks(files, chk):
    """{model: [DataFrame per seed]}, all aligned on the same (head, tail) rows."""
    ranks, ref = {}, None
    for label, fl in files.items():
        ranks[label] = []
        for f in fl:
            df = pd.read_csv(f, usecols=RANK_COLS)
            df["cold_start"] = df["cold_start"].astype(str).str.lower().eq("true")
            key = list(zip(df["head"], df["tail"]))
            if ref is None:
                ref = key
            elif key != ref:
                chk.add(f"rank dump aligned: {f}", False, "different rows or order: different split?")
                return None
            ranks[label].append(df)
    chk.add(f"all {sum(len(v) for v in ranks.values())} rank dumps aligned", True,
            f"{len(ref):,} test triples each")
    return ranks


def rebuild_split(task, cfg, ref_df, chk):
    """Exact E1 training pairs, by the same function the rank dump used; checked against the dump."""
    import drug_eval
    params = read_params(next(iter(cfg["models"].values())))
    t0 = time.time()
    out = drug_eval.get_dataset_for_drug_eval(
        cfg["tsv"], task, float(params["validation_size"]), float(params["test_size"]), True,
        int(params["split_seed"]), float(params["undersample_rate"]))
    train_t, val_t, test_t, ent2id = out[4], out[7], out[8], out[10]
    id2ent = {v: k for k, v in ent2id.items()}
    train = [(id2ent[h], id2ent[t]) for h, _, t in train_t.tolist()]
    test = [(id2ent[h], id2ent[t]) for h, _, t in test_t.tolist()]
    print(f"[i] split rebuilt in {time.time() - t0:.0f}s: train/val/test = "
          f"{len(train):,}/{len(val_t):,}/{len(test):,}", flush=True)

    dumped = list(zip(ref_df["head"], ref_df["tail"]))
    chk.add("rebuilt test set == test set of the rank dumps",
            Counter(test) == Counter(dumped), f"{len(test):,} vs {len(dumped):,} pairs")
    deg = Counter()
    for h, t in train:
        deg[h] += 1
        deg[t] += 1
    bad_h = int((ref_df["head"].map(deg).fillna(0).astype(int) != ref_df["head_target_deg_train"]).sum())
    bad_t = int((ref_df["tail"].map(deg).fillna(0).astype(int) != ref_df["tail_target_deg_train"]).sum())
    chk.add("rebuilt training degrees == degrees stored in the rank dumps", bad_h == 0 and bad_t == 0,
            f"{bad_h} head and {bad_t} tail mismatches")
    known = defaultdict(set)
    for h, t in train:
        known[h].add(t)
    return known, len(train), len(val_t)


# ================================================================================================
# Distances
# ================================================================================================
def ppi_distance(cfg, ref_df, known, max_hops):
    """Task A: PPI hops from the test protein to the nearest training target of the same drug."""
    g = pd.read_csv(cfg["tsv"], sep="\t", dtype=str, usecols=[0, 1, 2], compression="zip")
    g.columns = ["head", "interaction", "tail"]
    sub = g[g["interaction"] == cfg["ctx_rel"]]
    if sub.empty:
        raise SystemExit(f"[!] relation {cfg['ctx_rel']} not found in {cfg['tsv']}")
    adj = defaultdict(set)
    for a, b in zip(sub["head"], sub["tail"]):
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    print(f"[i] PPI graph: {len(adj):,} proteins, {sum(len(v) for v in adj.values()) // 2:,} edges", flush=True)

    memo = {}

    def bfs(drug):
        if drug not in memo:
            dist = {p: 0 for p in known.get(drug, ())}
            q = deque(dist)
            while q:
                u = q.popleft()
                if dist[u] >= max_hops:
                    continue
                for v in adj.get(u, ()):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            memo[drug] = dist
        return memo[drug]

    d = np.array([bfs(h).get(t, -1) for h, t in zip(ref_df["head"], ref_df["tail"])])
    return pd.DataFrame({"distance": d}), {"PPI proteins": len(adj),
                                           "PPI edges": sum(len(v) for v in adj.values()) // 2}


def extract_mondo(edges_zip, cache):
    """Stream the raw graph once; keep rdf:type edges between two Mondo classes (child, ancestor)."""
    if os.path.isfile(cache):
        e = pd.read_csv(cache, sep="\t", dtype=str)
        print(f"[i] Mondo hierarchy from cache {cache}: {len(e):,} edges", flush=True)
        return e
    import ijson
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    z = zipfile.ZipFile(edges_zip)
    name = [n for n in z.namelist() if n.endswith(".json")][0]
    rows, n, t0 = [], 0, time.time()
    with z.open(name) as f:
        for e in ijson.items(f, "item"):
            n += 1
            if n % 1_000_000 == 0:
                print(f"[i]   {n / 1e6:.0f}M edges read, {len(rows):,} Mondo rdf:type kept "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if e.get("predicate_uri") != RDF_TYPE:
                continue
            s, t = e["source_uri"], e["target_uri"]
            if "/MONDO_" in s and "/MONDO_" in t:
                rows.append((s.rsplit("/", 1)[-1], t.rsplit("/", 1)[-1]))
    e = pd.DataFrame(rows, columns=["child", "ancestor"]).drop_duplicates()
    e.to_csv(cache + ".part", sep="\t", index=False)
    os.replace(cache + ".part", cache)
    print(f"[i] Mondo hierarchy: {n:,} edges read, {len(e):,} kept -> {cache}", flush=True)
    return e


class Hierarchy:
    """Transitive reduction of the Mondo rdf:type DAG, with up-distances and descendant counts."""

    def __init__(self, edges):
        import networkx as nx
        self.nx = nx
        g = nx.DiGraph()
        g.add_edges_from((c, a) for c, a in zip(edges["child"], edges["ancestor"]) if c != a)
        self.stats = {"Mondo classes": g.number_of_nodes(), "rdf:type edges (raw)": g.number_of_edges()}
        cyc = [c for c in nx.strongly_connected_components(g) if len(c) > 1]
        if cyc:  # declared, not silently fixed: members of a cycle are made incomparable
            members = set().union(*cyc)
            g.remove_edges_from([(u, v) for u, v in list(g.edges()) if u in members and v in members
                                 and any(u in c and v in c for c in cyc)])
        self.stats["cycles removed (SCCs > 1)"] = len(cyc)
        t0 = time.time()
        self.g = nx.transitive_reduction(g)
        self.stats["edges after transitive reduction"] = self.g.number_of_edges()
        print(f"[i] Mondo DAG: {self.stats} ({time.time() - t0:.0f}s)", flush=True)
        self._up, self._ndesc = {}, {}

    def __contains__(self, x):
        return x in self.g

    def up(self, x):
        """{ancestor (self included): fewest steps up}."""
        if x not in self._up:
            self._up[x] = self.nx.single_source_shortest_path_length(self.g, x)
        return self._up[x]

    def ndesc(self, a):
        if a not in self._ndesc:
            self._ndesc[a] = len(self.nx.ancestors(self.g, a))  # edges child -> ancestor
        return self._ndesc[a]

    def distance(self, m, others, max_lca_desc):
        """(capped distance, kind, nearest known, lca, lca descendants, uncapped distance)."""
        um = self.up(m)
        best, best_any = None, None
        for k in others:
            if k not in self.g:
                continue
            uk = self.up(k)
            for a in um.keys() & uk.keys():
                d = um[a] + uk[a]
                lineal = a == m or a == k
                if best_any is None or d < best_any:
                    best_any = d
                if not lineal and self.ndesc(a) > max_lca_desc:
                    continue
                kind = ("descendant of a known indication" if a == k else
                        "ancestor of a known indication" if a == m else "collateral (common ancestor)")
                cand = (d, 0 if lineal else 1, kind, k, a)
                if best is None or cand[:2] < best[:2]:
                    best = cand
        if best is None:
            return -1, "no connection", "", "", -1, -1 if best_any is None else best_any
        return best[0], best[2], best[3], best[4], self.ndesc(best[4]), best_any


def mondo_distance(ref_df, known, max_lca_desc):
    edges = extract_mondo(EDGES_ZIP, MONDO_CACHE)
    H = Hierarchy(edges)
    mid = lambda e: e.split("::", 1)[-1]  # noqa: E731  Disease::MONDO_0000727 -> MONDO_0000727
    known_m = {d: {mid(t) for t in ts} for d, ts in known.items()}
    rows, t0 = [], time.time()
    for i, (h, t) in enumerate(zip(ref_df["head"], ref_df["tail"])):
        m = mid(t)
        if m not in H:
            rows.append((-2, "not in the Mondo hierarchy", "", "", -1, -2))
        else:
            rows.append(H.distance(m, known_m.get(h, ()), max_lca_desc))
        if (i + 1) % 5000 == 0:
            print(f"[i]   {i + 1:,}/{len(ref_df):,} test pairs ({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows, columns=["distance", "kind", "nearest_known", "lca", "lca_descendants",
                                      "distance_uncapped"])
    out["nearest_known"] = out["nearest_known"].map(lambda x: f"Disease::{x}" if x else "")
    H.stats["test diseases in the hierarchy"] = int((out["distance"] != -2).sum())
    return out, H.stats


# ================================================================================================
# Strata and tables
# ================================================================================================
def regime(ref_df, what):
    r = pd.Series("has known", index=ref_df.index)
    r[ref_df["head_target_deg_train"] == 0] = REG_NONE.format(what=what)
    r[ref_df["cold_start"]] = REG_COLD
    return r


def bin_distance(d, task, reg):
    if task == "DTI":
        edges, labels = [0, 1, 2, 3, 4], ["1 hop", "2 hops", "3 hops", "4 hops"]
        far = "> 4 hops or unreachable"
    else:
        edges, labels = [0, 1, 2, 3, 4, 6], ["1 (parent/child)", "2", "3", "4", "5-6"]
        far = "> 6 or no connection"
    out = []
    for x, r in zip(d, reg):
        if r != "has known":
            out.append(r)
        elif x == -2:
            out.append("not in the Mondo hierarchy")
        elif x == -1 or x > edges[-1]:
            out.append(far)
        else:
            for lo_i in range(len(labels)):
                lo, hi = edges[lo_i], edges[lo_i + 1]
                if lo < x <= hi:
                    out.append(labels[lo_i])
                    break
            else:
                out.append("0 (duplicate pair?)")
    order = labels + [far, "not in the Mondo hierarchy"]
    return pd.Series(out), order


def stratum_mrr(ranks, mask):
    """{model: (array of per-seed MRR, array of per-seed drug->target MRR)}."""
    res = {}
    for label, dfs in ranks.items():
        a = np.array([df.loc[mask, "rr_mean"].mean() for df in dfs])
        b = np.array([(1.0 / df.loc[mask, "rank_tail"]).mean() for df in dfs])
        res[label] = (a, b)
    return res


def table(ranks, col, order, title, note, per_seed_rows, task, metric=0):
    from stratified_multiseed import fmt, pval
    models = list(ranks)
    others = [m for m in models if m != REFERENCE]
    n_all = len(col)
    L = [f"\n### {title}\n", note + "\n" if note else ""]
    head = ["stratum", "n", "share"] + models + [f"Δ {m} − {REFERENCE}" for m in others] + \
           [f"p {m}" for m in others]
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))
    present = [s for s in order if (col == s).any()] + sorted(set(col) - set(order))
    for s in present:
        mask = (col == s).values
        n = int(mask.sum())
        res = stratum_mrr(ranks, mask)
        for m in models:
            for seed, (v, w) in enumerate(zip(*res[m])):
                per_seed_rows.append(dict(task=task, table=title, stratum=s, model=m, seed=seed, n=n,
                                          mrr=v, mrr_drug_to_target=w))
        cells = [s, f"{n:,}", f"{100 * n / n_all:.1f}%"] + [fmt(res[m][metric]) for m in models]
        cells += [f"{np.mean(res[m][metric]) - np.mean(res[REFERENCE][metric]):+.3f}" for m in others]
        cells += [pval(res[m][metric], res[REFERENCE][metric]) for m in others]
        L.append("| " + " | ".join(cells) + " |")
    return L


def cumulative(dist, reg, ranks, task, max_k):
    from stratified_multiseed import fmt
    L = ["\n### Cumulative share of the test set within distance k of the supervision\n",
         "Share of ALL test triples whose distance is ≤ k, and the MRR on that subset.\n",
         "| k | n | share | " + " | ".join(ranks) + " |", "|" + "---|" * (3 + len(ranks))]
    for k in range(1, max_k + 1):
        mask = ((reg == "has known") & (dist >= 1) & (dist <= k)).values
        if not mask.any():
            continue
        res = stratum_mrr(ranks, mask)
        L.append(f"| ≤ {k} | {int(mask.sum()):,} | {100 * mask.mean():.1f}% | " +
                 " | ".join(fmt(res[m][0]) for m in ranks) + " |")
    return L


# ================================================================================================
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state():
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
        return c or "unknown", bool(d)
    except Exception:  # noqa: BLE001
        return "unknown", None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(E1))
    p.add_argument("--out_dir", default=None, help="version folder, e.g. experiments/block1/results/X0a_v1")
    p.add_argument("--check", action="store_true", help="verify the requirements only; write nothing")
    p.add_argument("--max_hops", type=int, default=4, help="Task A: BFS depth in the PPI graph")
    p.add_argument("--max_lca_desc", type=int, default=250,
                   help="Task B: largest common ancestor (in descendants) that still connects two "
                        "diseases collaterally; fixed before the run (about 1%% of Mondo)")
    args = p.parse_args()
    os.chdir(ROOT)
    if not args.check and not args.out_dir:
        p.error("--out_dir is required unless --check")
    task, cfg = args.task, E1[args.task]
    t_start = time.time()
    print(f"[x0a] {datetime.datetime.now():%F %T} task {task} on {socket.gethostname()}"
          f"{' (check only)' if args.check else ''}", flush=True)
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("[i] PYTHONHASHSEED is not 0: harmless here (entities are matched by name), set it anyway "
              "for consistency with E1/E5", flush=True)

    chk = Checks()
    files = check_inputs(task, cfg, chk)
    if not chk.ok:
        raise SystemExit("[x0a] requirements not met (FAIL above): nothing computed")
    ranks = load_ranks(files, chk)
    if ranks is None:
        raise SystemExit("[x0a] rank dumps not aligned: nothing computed")
    ref = ranks[REFERENCE][0]
    for label, dfs in ranks.items():
        got = np.mean([df["rr_mean"].mean() for df in dfs])
        chk.add(f"{label}: mean MRR over seeds = E1", abs(got - cfg["e1_mrr"][label]) < 0.0015,
                f"{got:.4f} (E1 {cfg['e1_mrr'][label]:.3f})")
    known, n_train, n_val = rebuild_split(task, cfg, ref, chk)
    if not chk.ok:
        raise SystemExit("[x0a] the rebuilt split does not match the rank dumps: nothing computed")
    if args.check:
        print(f"[x0a] CHECK PASSED: all requirements met for {task} ({time.time() - t_start:.0f}s)")
        return

    # ---- distances (model-independent) ----
    if task == "DTI":
        dist_df, dstats = ppi_distance(cfg, ref, known, args.max_hops)
    else:
        dist_df, dstats = mondo_distance(ref, known, args.max_lca_desc)
    reg = regime(ref, cfg["what"])
    dbin, order = bin_distance(dist_df["distance"], task, reg)
    order = order + [REG_NONE.format(what=cfg["what"]), REG_COLD]

    strata = pd.concat([ref[["head", "tail", "head_target_deg_train", "tail_target_deg_train",
                             "cold_start"]].reset_index(drop=True),
                        pd.DataFrame({"regime": reg.values, "distance_stratum": dbin.values}),
                        dist_df.reset_index(drop=True)], axis=1)

    # ---- tables ----
    per_seed = []
    unit = "PPI hops" if task == "DTI" else "Mondo steps"
    L = [f"# X0a — MRR by distance from the supervision — {task}\n",
         f"{len(ref):,} test triples of the fixed E1 split (train/val/test target edges = "
         f"{n_train:,}/{n_val:,}/{len(ref):,}). Each cell: mean ± sd over {N_SEEDS} seeds of the MRR "
         f"inside the stratum (rr_mean, the E1 definition); Δ and p (Welch, two-sided, over seeds) "
         f"against {REFERENCE}. Distance = {unit} from the test {cfg['what']} to the nearest "
         f"{cfg['what']} the same drug has in the TRAINING split. Generated by "
         f"`experiments/block1/x0a_distance_curve.py`.\n"]
    L.append("## Graph used for the distance\n")
    L += [f"- {k}: {v:,}" if isinstance(v, int) else f"- {k}: {v}" for k, v in dstats.items()]
    if task == "TREATS":
        L.append(f"- collateral connections only through common ancestors with ≤ {args.max_lca_desc} "
                 f"descendants; lineal (ancestor/descendant) connections always count")
    L.append("\n## Results")
    L += table(ranks, dbin, order, f"MRR by distance ({unit})", "", per_seed, task)
    L += table(ranks, dbin, order, f"Drug → {cfg['what']} direction only, by distance ({unit})",
               "MRR of the rank of the true " + cfg["what"] + " among the candidates for the drug "
               "(the repurposing question), instead of the mean of both directions.",
               [], task, metric=1)
    L += cumulative(dist_df["distance"], reg, ranks, task, 6 if task == "TREATS" else args.max_hops)
    if task == "TREATS":
        kind = pd.Series([k if r == "has known" else r for k, r in zip(dist_df["kind"], reg)])
        korder = ["descendant of a known indication", "ancestor of a known indication",
                  "collateral (common ancestor)", "no connection", "not in the Mondo hierarchy"]
        L += table(ranks, kind, korder, "MRR by kind of hierarchical connection",
                   "Descendant = the test disease is more specific than one the drug already treats in "
                   "training; ancestor = more general; collateral = connected through an informative "
                   "common ancestor.", per_seed, task)
        dbin_u, _ = bin_distance(dist_df["distance_uncapped"], task, reg)
        L += table(ranks, dbin_u, order, "Sensitivity: distance through ANY common ancestor (no cap)",
                   "Same as the first table but top-level grouping classes also connect diseases; "
                   "expected to push almost everything to distance ≤ 4.", per_seed, task)

    # ---- write ----
    out = os.path.join(args.out_dir, "summary")
    os.makedirs(out, exist_ok=True)
    md = os.path.join(out, f"x0a_{task}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    pd.DataFrame(per_seed).to_csv(os.path.join(out, f"x0a_{task}_per_seed.csv"), index=False)
    strata.to_csv(os.path.join(out, f"x0a_{task}_strata.csv"), index=False)
    commit, dirty = git_state()
    prov = dict(experiment="X0a", task=task, finished=f"{datetime.datetime.now():%F %T}",
                seconds=round(time.time() - t_start), host=socket.gethostname(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), git_commit=commit, git_dirty_tracked=dirty,
                python=platform.python_version(),
                packages={m: getattr(importlib.import_module(m), "__version__", "") for m in
                          ["numpy", "pandas", "scipy", "torch"] + (["networkx", "ijson"] if task == "TREATS" else [])},
                params=dict(max_hops=args.max_hops, max_lca_desc=args.max_lca_desc, n_seeds=N_SEEDS,
                            reference=REFERENCE),
                inputs=dict(tsv=cfg["tsv"], tsv_sha256=sha256(cfg["tsv"]), models=cfg["models"],
                            mondo_source=(EDGES_ZIP if task == "TREATS" else None)),
                checks=[dict(item=i, ok=o, detail=d) for i, o, d in chk.rows], distance_graph=dstats)
    with open(os.path.join(out, f"x0a_{task}_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2, default=str)
    print(f"[x0a] written {md} (+ _per_seed.csv, _strata.csv, _provenance.json) "
          f"in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
