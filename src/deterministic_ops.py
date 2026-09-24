"""
deterministic_ops.py — bit-reproducible replacements for the two GPU operations that are not.

WHY. On CUDA, summing many values into the same output slot is done with atomic additions, whose
order depends on thread scheduling. Floating-point addition is not associative, so the result
changes in the last bits from one execution to the next. In this code base two operations do it:

  1. the message aggregation of R-GCN, `torch_sparse.spmm` (a `scatter_add` in the forward pass and
     another one in the backward pass);
  2. the backward pass of every `embedding[index]` read in the DistMult decoder and in the
     regulariser, where an entity that occurs in many triples accumulates many gradients.

The differences start around the fourth decimal of the loss after a few epochs. Usually they fade
out, but sometimes they move the epoch at which validation M peaks, and then the selected checkpoint
changes: in the Task A ablation the same recipe, seed and data gave test MRR 0.356 on one execution
and 0.386 on another. PyTorch's own deterministic mode (`torch.use_deterministic_algorithms`) does
fix it, but its deterministic scatter is much slower and heavier in memory, and on the full Task A
graph it does not fit a 4 GB card.

HOW. Both operations are sums over groups: output row r is the sum of the rows whose index is r.
Sorting the entries by index once and summing each contiguous segment with `torch.segment_reduce`
gives the same numbers every time, because each segment is summed in a fixed order. No atomics, no
extension beyond PyTorch itself, and the cost is one argsort per call. Verified on a 400k x 64 sum:
five repeated atomic sums were never bit-identical, five segment sums always were, and the two agree
to 1e-5.

The replacements are used only when enabled (train_and_eval.py --deterministic, or the environment
variable PKT_DETERMINISTIC=1). With the switch off the code path is exactly the original one, so
earlier results (E0, E1) are unaffected. Scope: R-GCN and the DistMult decoder, i.e. the models of
the ablation. CompGCN keeps its own scatter and is not covered.
"""
import os

import torch


def enabled():
    return os.environ.get("PKT_DETERMINISTIC") == "1"


def _segments(index, size):
    """Stable sort of `index` and the length of each of its `size` groups (empty ones included)."""
    perm = torch.argsort(index, stable=True)
    lengths = torch.bincount(index, minlength=size)   # integer counts: exact, order does not matter
    return perm, lengths


def _segment_sum(src, lengths):
    """Sum of consecutive groups of rows, in a fixed order (torch.segment_reduce, no atomics)."""
    return torch.segment_reduce(src, "sum", lengths=lengths, axis=0)


class _Gather(torch.autograd.Function):
    """src[index] whose backward sums the incoming gradients per row deterministically."""

    @staticmethod
    def forward(ctx, src, index):
        ctx.save_for_backward(index)
        ctx.n = src.size(0)
        return src.index_select(0, index)

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        perm, lengths = _segments(index, ctx.n)
        return _segment_sum(grad[perm], lengths), None


class _SpMM(torch.autograd.Function):
    """out[r] = sum over entries e with row[e] == r of val[e] * x[col[e]], deterministically.

    `val` is treated as a constant (in R-GCN it is the degree normalisation, computed from the
    graph and never trained), so no gradient is returned for it.
    """

    @staticmethod
    def forward(ctx, row, col, val, m, x):
        perm, lengths = _segments(row, m)
        out = _segment_sum(val[perm, None] * x.index_select(0, col[perm]), lengths)
        ctx.save_for_backward(row, col, val)
        ctx.n = x.size(0)
        return out

    @staticmethod
    def backward(ctx, grad):
        row, col, val = ctx.saved_tensors
        perm, lengths = _segments(col, ctx.n)
        grad_x = _segment_sum(val[perm, None] * grad.index_select(0, row[perm]), lengths)
        return None, None, None, None, grad_x


def gather(src, index):
    """Drop-in for `src[index]` on the first dimension."""
    return _Gather.apply(src, index) if enabled() else src[index]


def spmm(index, val, m, n, x):
    """Drop-in for `torch_sparse.spmm(index, val, m, n, x)`, with index = [row; col]."""
    if not enabled():
        import torch_sparse
        return torch_sparse.spmm(index, val, m, n, x)
    return _SpMM.apply(index[0], index[1], val, m, x)


def configure():
    """Everything else PyTorch needs to be reproducible on GPU (cuBLAS and cuDNN)."""
    os.environ["PKT_DETERMINISTIC"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
