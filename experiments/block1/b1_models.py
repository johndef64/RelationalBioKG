"""
b1_models.py — registry of the models that block 1 adds to the v1.0.0 ones.

train_b1.py builds a model of this registry when --model names one of its entries, and falls back
to train_and_eval.get_model (v1.0.0, unchanged) for the v1 names (rgcn, rgat, compgcn, distmult).

A model must have the interface of the v1 encoders, because it goes through the v1 training loop
and evaluation unchanged:
    forward(x_dict, edge_index) -> node embeddings (num_entities x d)
    distmult(embedding, triplets) -> raw scores (logits), one per triple
    reg_loss(embedding, triplets) -> scalar L2 penalty of the decoder inputs

An entry is a Spec:
    base      the v1 model whose hyperparameter block of --config the new model starts from
              (train_b1.py copies it into the run folder, then applies the entry's defaults and the
              --override flags, in that order);
    defaults  hyperparameters that the v1 block does not have (for example a gate initialisation);
    build     build(params, in_channels_dict, num_nodes_per_type, num_entities, num_relations,
                    device) -> nn.Module, with params = the merged hyperparameter block.

The entry `rgcn_v1` builds exactly the thesis R-GCN through this path. It is not an experiment: it
is the control of test_equivalence.py, which checks that a registered model goes through
train_b1.py with the same numbers as the v1 model it copies.
"""
from dataclasses import dataclass, field
from typing import Callable

import torch.nn.functional as F


@dataclass(frozen=True)
class Spec:
    base: str
    build: Callable
    defaults: dict = field(default_factory=dict)
    doc: str = ""


def _conv_channels(p):
    return {f"layer_{x}": p[f"layer_{x}"] for x in range(p["conv_layer_num"])}


def _build_rgcn_v1(p, in_channels_dict, num_nodes_per_type, num_entities, num_relations, device):
    # Same constructor call as train_and_eval.get_model for 'rgcn' (v1.0.0), argument by argument.
    from src.hetero_rgcn import HeterogeneousRGCN
    return HeterogeneousRGCN(in_channels_dict, p["mlp_out_layer"], p["mlp_out_layer"], _conv_channels(p),
                             num_nodes_per_type, num_entities, num_relations + 1, p["conv_layer_num"],
                             p["num_bases"], activation_function=F.relu, device=device)


REGISTRY = {
    "rgcn_v1": Spec(base="rgcn", build=_build_rgcn_v1,
                    doc="the thesis R-GCN, built through the block-1 registry (equivalence control)"),
}
