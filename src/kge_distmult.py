"""
Plain DistMult knowledge-graph embedding (NO message passing) — baseline for the GNN encoders.

Same interface as HeterogeneousRGCN / HeterogeneousCompGCN (forward, distmult, reg_loss), so it
runs through the identical training loop, loss, negative sampling and filtered evaluation of
train_and_eval.py. The graph passed to forward() is ignored: every entity has a free lookup
embedding and scores come from the DistMult decoder only. If a GNN does not beat this model,
message passing over the KG context is not adding anything.
"""
import torch
import torch.nn as nn


class DistMultKGE(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim, device="cuda:0"):
        super().__init__()
        self.device = device
        self.entity_embedding = nn.Embedding(num_entities, emb_dim)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, emb_dim))
        # KGE-style init (Bordes et al. 2013): U(-6/sqrt(d), 6/sqrt(d)). Xavier on a (num_entities x d)
        # table gives ~1e-2 entries, the 3-way DistMult product is ~0 and the model barely trains,
        # which would make the baseline artificially weak.
        bound = 6.0 / emb_dim ** 0.5
        nn.init.uniform_(self.entity_embedding.weight, -bound, bound)
        nn.init.uniform_(self.relation_embedding, -bound, bound)

    def forward(self, x_dict, edge_index):
        return self.entity_embedding.weight

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        return torch.sum(s * r * o, dim=1)

    def reg_loss(self, embedding, triplets):
        s_index, p_index, o_index = triplets.t()
        s, p, o = embedding[s_index, :], self.relation_embedding[p_index, :], embedding[o_index, :]
        return s.pow(2).mean() + p.pow(2).mean() + o.pow(2).mean()
