################# TRAIN AND EVAL ####################


# # Specify a different TASK (edge type) to train on
# python train_and_eval.py --task CMP_BIND

# # Use a different model
# python train_and_eval.py --model rgcn

# # Change number of runs and epochs
# python train_and_eval.py --runs 5 --epochs 200

# # Training with pretraining
# python train_and_eval.py --pretrain_epochs 50

# # Training with focal loss parameters
# python train_and_eval.py --alpha 0.25 --gamma 3.0 --alpha_adv 2.0

########################################################

# # Full configuration with multiple parameters
# python train_and_eval.py --target 83332 --model compgcn --runs 3 --epochs 300 --patience 100 --validation_size 0.15 --test_size 0.2 --evaluate_every 10 --negative_rate 2 --oversample_rate 5 --undersample_rate 0.5

# # With pretraining and frozen base layers
# python train_and_eval.py --model rgat --pretrain_epochs 100 --freeze_base --epochs 200

# # Training on target graph with custom sampling
# python train_and_eval.py --oversample_rate 10 --undersample_rate 0.3 --negative_rate 3

# # Quiet mode (minimal output)
# python train_and_eval.py --quiet --runs 5

"""
Train and evaluate heterogeneous GNN models (CompGCN, R-GCN, R-GAT) for link
prediction on the TARGET relation in PathogenKG.

The dataset is loaded as (head, interaction, tail) triples from a TSV file.
See README.md for usage examples.
"""



#%%
import warnings
warnings.simplefilter(action='ignore')

import os
import json
import time
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm, trange
import torch.nn.functional as F
from torcheval.metrics.functional import binary_auprc, binary_auroc

from src.utils import set_seed, load_data, select_target_triplets, entities2id_offset, rel2id_offset, edge_ind_to_id, entities_features_flattening,\
                      set_target_label, triple_sampling, graph_to_undirect, negative_sampling, negative_sampling_filtered, add_self_loops, evaluation_metrics_sampled, evaluation_metrics_full

from src.evaluation_metrics_filtered import evaluation_metrics_filtered, evaluation_metrics_filtered_typeconstrained, print_metrics

from src.hetero_rgcn import HeterogeneousRGCN as rgcn
from src.hetero_rgat import HeterogeneousRGAT as rgat
from src.hetero_compgcn import HeterogeneousCompGCN as compgcn
from src.kge_distmult import DistMultKGE

BASE_SEED = 42

# Single, pre-merged dataset ready for training.
dataset = 'PathogenKG_merged.tsv'
dataset = 'PathogenKG_n19.tsv'


dataset = 'PathogenKG_n31_core.tsv.zip'
DEFAULT_TRAIN_TSV = os.path.join('dataset', dataset)
models_params_path = './src/models_params.json'
CONFIG_NAME = 'pathogen31-cmp-gene'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEBUG = False
# Use alternative negative sampling strategy (filtered)
USE_ALTERNATIVE_NEG_SAMPLING = True
# Use filtered evaluation metrics (standard in KGE literature) instead of legacy pool-based evaluation
# Use Type Constrained Filtered Evaluation (more biologically plausible negatives) instead of standard filtered evaluation on full graph
USE_EVAL_TYPE_CONSTRAINED = True  

"""
If True, uses type-constrained filtered negative sampling:
- Corrupts heads with valid Compound entities and tails with valid ExtGene entities
- Filters out all known true triples (train+val+test) to avoid false negatives
- Deterministic seed for val/test (fixed per run), variable for train (seed+epoch)

If False, uses standard (unfiltered) negative sampling (kept for comparison).
"""

def _resolve_dataset_path(tsv_path: str) -> str:
  resolved = os.path.normpath(tsv_path)
  if not os.path.exists(resolved):
    raise FileNotFoundError(
      f"Training TSV not found: {resolved}. Expected the merged dataset at '{DEFAULT_TRAIN_TSV}'."
    )
  return resolved

# Focal loss for binary classification with logits
def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
  """
  Compute focal loss for binary classification with logits.
  Args:
    inputs: Raw model logits.
    targets: Binary targets (0 or 1), same shape as inputs.
    alpha: Weighting factor for positive examples.
    gamma: Focusing parameter to down-weight easy examples.
    reduction: 'mean', 'sum', or 'none'.
  """
  bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
  p_t = torch.exp(-bce)
  loss = alpha * (1 - p_t) ** gamma * bce
  if reduction == 'mean':
    return loss.mean()
  elif reduction == 'sum':
    return loss.sum()
  else:
    return loss


def mixed_metric(m):
  """Composite model-selection metric of PathogenKG: M = 0.2*AUROC + 0.4*AUPRC + 0.4*MRR."""
  return 0.2 * m["Auroc"] + 0.4 * m["Auprc"] + 0.4 * m["MRR"]


class SupervisionSampler:
  """
  Disjoint message-passing / supervision edges for the training target relation(s).

  Without it, every training target edge is BOTH in the message-passing graph and a positive
  in the loss, so the model can score an edge it has just propagated a message through, while
  val/test edges are never in the graph (train/test mismatch). Standard GNN link-prediction
  practice (e.g. PyG RandomLinkSplit(disjoint_train_ratio), NBFNet removing target edges)
  keeps the two disjoint.

  At every epoch the unique training target edges are reshuffled: a fraction `ratio` is
  REMOVED from the graph and used only as loss positives; the rest stays in the graph as
  context. Over the epochs every edge plays both roles. Validation/test still use the full
  training graph (all training target edges visible), exactly as before.
  """

  def __init__(self, train_index, train_triplets, num_relations, ratio, oversample_rate, seed):
    if not 0.0 < ratio < 1.0:
      raise ValueError("disjoint_supervision ratio must be in (0, 1)")
    self.ratio = ratio
    self.oversample_rate = max(1, int(oversample_rate))
    self.seed = seed
    self.num_rel_dir = (num_relations - 1) // 2          # relation2id = R forward + R reverse + self
    target_rels = torch.unique(torch.as_tensor(train_triplets)[:, 1]).to(train_index.device)
    rel = train_index[:, 1]
    is_target = torch.isin(rel, target_rels) | torch.isin(rel, target_rels + self.num_rel_dir)
    self.base_index = train_index[~is_target]            # context (both directions) + self-loops
    self.positives = torch.unique(torch.as_tensor(train_triplets).cpu(), dim=0)  # drop oversampling copies

  def sample(self, epoch):
    g = torch.Generator().manual_seed(int(self.seed) + int(epoch))
    n = self.positives.size(0)
    perm = torch.randperm(n, generator=g)
    n_sup = min(n - 1, max(1, int(round(n * self.ratio))))
    supervision = self.positives[perm[:n_sup]]
    message = self.positives[perm[n_sup:]]
    if self.oversample_rate > 1:
      supervision = supervision.repeat(self.oversample_rate, 1)
      message = message.repeat(self.oversample_rate, 1)
    message = message.to(self.base_index.device)
    reverse = message.clone()
    reverse[:, [0, 2]] = reverse[:, [2, 0]]
    reverse[:, 1] += self.num_rel_dir
    epoch_index = torch.cat([self.base_index, message, reverse], dim=0)
    return epoch_index, supervision


def cold_start_mask(triplets, train_index, num_relations):
  """
  True for test triples with at least one endpoint that has NO edge in the training
  message-passing graph (self-loops excluded): its embedding is never updated by message
  passing, so the triple is essentially unpredictable from the KG context.
  """
  self_rel = num_relations - 1
  edges = train_index[train_index[:, 1] != self_rel]
  num_nodes = int(max(train_index[:, 0].max().item(), train_index[:, 2].max().item())) + 1
  deg = torch.bincount(torch.cat([edges[:, 0], edges[:, 2]]).cpu(), minlength=num_nodes)
  t = torch.as_tensor(triplets).cpu()
  return (deg[t[:, 0]] == 0) | (deg[t[:, 2]] == 0)

def get_dataset_pretrain(tsv_path, quiet):
  """
  Load full graph for multi-relational pretraining.
  Returns same outputs as get_dataset, but using all relations as positives.
  """
  # Load the same edge_index and features
  edge_index, node_features_per_type = load_data(tsv_path, {}, quiet)
  ent2id, all_nodes_per_type = entities2id_offset(edge_index, node_features_per_type, quiet)
  relation2id = rel2id_offset(edge_index)

  # Convert to indexed triplets
  indexed_edge_index = edge_ind_to_id(edge_index, ent2id, relation2id)

  # Use all triplets as positives
  all_triplets = indexed_edge_index.loc[:, ["head","interaction","tail"]].values

  # Split into train/val/test
  train_triplets, val_triplets, test_triplets = triple_sampling(
    all_triplets, 0.1, 0.2, quiet, BASE_SEED
  )

  # Build full message-passing graph from all edges (undirected + self-loops)
  num_nodes = sum(len(v) for v in all_nodes_per_type.values())
  full_index = graph_to_undirect(all_triplets, len(relation2id))
  full_index = add_self_loops(full_index, num_nodes, len(relation2id))

  # Flatten features, move to tensors/DEV
  train_triplets = torch.tensor(train_triplets).to(device)
  val_triplets   = torch.tensor(val_triplets).to(device)
  test_triplets  = torch.tensor(test_triplets).to(device)

  train_val_triplets = torch.cat([train_triplets, val_triplets], 0).to(device)
  train_val_test_triplets = torch.cat([train_val_triplets, test_triplets], 0).to(device)
  flattened_features_per_type = entities_features_flattening(node_features_per_type, all_nodes_per_type)
  in_channels_dict = {nt:(feat.shape[1] if feat is not None else None)
                      for nt, feat in flattened_features_per_type.items()}
  
  num_nodes_per_type = {node_type:len(nodes) for node_type,nodes in all_nodes_per_type.items()}

  return (in_channels_dict, num_nodes_per_type, num_nodes, len(relation2id),
          train_triplets, full_index, flattened_features_per_type,
          val_triplets, train_val_triplets, test_triplets,
          train_val_test_triplets, edge_index, ent2id, relation2id)

def get_dataset(tsv_path, task, validation_size, test_size, quiet, seed, oversample_rate, undersample_rate):
  edge_index, node_features_per_type = load_data(tsv_path, {}, quiet, 
                                                #  debug=DEBUG
                                                 )
  
  # Label target edges BEFORE converting to IDs (need string entity names to derive edge types)
  #if task is a list keep is as it is if is a string apply this [ x for x in task.split(',')] 
  target_list = [ x for x in task.split(',')] if isinstance(task, str) else task
  edge_index = set_target_label(edge_index, target_list)
  
  ent2id, all_nodes_per_type  = entities2id_offset(edge_index, node_features_per_type, quiet)
  relation2id                 = rel2id_offset(edge_index)
  indexed_edge_index          = edge_ind_to_id(edge_index, ent2id, relation2id)
  flattened_features_per_type = entities_features_flattening(node_features_per_type, all_nodes_per_type)
  
  # Labels are already in edge_index, copy to indexed version
  indexed_edge_index["label"] = edge_index["label"].values

  non_target_triplets, target_triplets        = select_target_triplets(indexed_edge_index)
  train_triplets, val_triplets, test_triplets = triple_sampling(
                                                                target_triplets.loc[:,["head","interaction","tail"]].values, 
                                                                validation_size, 
                                                                test_size, 
                                                                quiet,
                                                                seed
                                                              )
  
  # under-sampling non-target
  if undersample_rate < 1.0:
    rnd = np.random.RandomState(seed)
    keep = int(len(non_target_triplets) * undersample_rate)
    idx = rnd.choice(len(non_target_triplets), size=keep, replace=False)
    non_target_triplets = non_target_triplets.iloc[idx]

  # over-sampling training positive edges
  if oversample_rate > 1:
    train_triplets = np.repeat(train_triplets, oversample_rate, axis=0)

  non_target_triplets = non_target_triplets.loc[:,["head","interaction","tail"]].values

  # print(f" non target {non_target_triplets.shape}")
  train_index = np.concatenate([non_target_triplets,train_triplets], axis=0)
  
  # print(f" target triplets{target_triplets.shape}")
  num_nodes_per_type = {node_type:len(nodes) for node_type,nodes in all_nodes_per_type.items()}

  num_entities = 0
  for features in num_nodes_per_type.values():
      num_entities += features

  # print(f" non target+target {train_index.shape}")
  train_index = graph_to_undirect(train_index, len(relation2id))
  # print(f" undirected {train_index.shape}")
  train_index = add_self_loops(train_index, num_entities, len(relation2id))
  # print(f" loops {train_index.shape}")
  
  train_triplets, val_triplets, test_triplets = torch.tensor(train_triplets), torch.tensor(val_triplets), torch.tensor(test_triplets)
  
  train_val_triplets = torch.cat([train_triplets, val_triplets], axis=0)
  train_val_test_triplets = torch.cat([train_val_triplets, test_triplets], axis=0)

  train_val_triplets = train_val_triplets.to(device)
  train_val_test_triplets = train_val_test_triplets.to(device)
        
  in_channels_dict = {node_type:(features.shape[1] if features!= None else None) for node_type, features in flattened_features_per_type.items() }
  num_relations = len(relation2id)

  return in_channels_dict, num_nodes_per_type, num_entities, num_relations, \
    train_triplets, train_index, flattened_features_per_type, val_triplets, \
    train_val_triplets, test_triplets, train_val_test_triplets, edge_index, \
    ent2id, relation2id

def get_model(model_name, task, in_channels_dict, num_nodes_per_type, num_entities, num_relations, config_name = "pathogen31-128"):

  with open(models_params_path, 'r') as f:
    models_params = json.load(f)
  
  # Select params by config (not by task). Fallback to "default" config if needed.
  
  if config_name not in models_params:
    print(f"[get_model] Config '{config_name}' not found, using 'default' config")
    config_name = "default"

  if config_name not in models_params:
    raise KeyError("[get_model] Neither requested config nor 'default' found in models_params.json")
  if model_name == 'distmult' and 'distmult' not in models_params[config_name]:
    # Baseline without message passing: reuse the R-GCN optimisation settings of the same
    # config (lr, weight decay, reg, ...) with embedding size = mlp_out_layer.
    base = models_params[config_name].get('rgcn')
    if base is None:
      raise KeyError(f"[get_model] config '{config_name}' has neither 'distmult' nor 'rgcn' params")
    p = dict(base)
    print(f"[get_model] DistMult baseline: using R-GCN optimisation params of '{config_name}' (dim={p['mlp_out_layer']})")
    model = DistMultKGE(num_entities, num_relations + 1, p['mlp_out_layer'], device=device)
    return model, p['learning_rate'], p['regularization'], p['grad_norm'], p['weight_decay'], p['scheduler_gamma']
  if model_name not in models_params[config_name]:
    raise KeyError(f"[get_model] Model '{model_name}' not found in config '{config_name}'")

  model_params = models_params[config_name][model_name]
  print(f"[get_model] Using parameters from config '{config_name}': {model_params}")

  conv_hidden_channels = {f'layer_{x}':model_params[f'layer_{x}']  for x in range(model_params['conv_layer_num'])} 

  if model_name == 'rgcn':
    model = rgcn(
      in_channels_dict,
      model_params['mlp_out_layer'],
      model_params['mlp_out_layer'],
      conv_hidden_channels,
      num_nodes_per_type,
      num_entities,
      num_relations + 1,
      model_params['conv_layer_num'],
      model_params['num_bases'],
      activation_function=F.relu,
      device=device
    )
  elif model_name == 'rgat':
    model = rgat(
      in_channels_dict,
      model_params['mlp_out_layer'],
      model_params['mlp_out_layer'],
      conv_hidden_channels,
      num_nodes_per_type,
      num_entities,
      num_relations + 1,
      conv_num_layers=model_params['conv_layer_num'],
      num_bases=model_params['num_bases'],
      activation_function=F.relu,
      device=device
    )
  elif model_name == 'compgcn':
    model = compgcn(
      in_channels_dict,
      mlp_out_emb_size=model_params['mlp_out_layer'],
      conv_hidden_channels=conv_hidden_channels,
      num_nodes_per_type=num_nodes_per_type,
      num_entities=num_entities,
      num_relations=num_relations,
      dropout=model_params['dropout'],
      conv_num_layers=model_params['conv_layer_num'],
      opn=model_params['opn'],
      use_layer_norm=model_params.get('use_layer_norm', True),
      device=device,
    )
  elif model_name == 'distmult':
    p = model_params
    model = DistMultKGE(num_entities, num_relations + 1, p['mlp_out_layer'], device=device)
  else:
    return None
  
  return model, model_params['learning_rate'], model_params['regularization'], model_params['grad_norm'], model_params['weight_decay'], model_params['scheduler_gamma']



def train(model, optimizer, gradnorm, reg_param, x_dict, index , target_triplets, target_labels, alpha, gamma, alpha_adv, change_points=None):
  metrics = {"Auroc":{},"Auprc":{},"Loss":{}}
  model.train()
  optimizer.zero_grad()
  if change_points != None:
    out = model(x_dict, index, change_points)
  else:
    out = model(x_dict, index)
  
  scores  = model.distmult(out, target_triplets)

  # loss    = model.score_loss(scores, target_labels)
  # reg_loss = (reg_param * model.reg_loss(out, target_triplets))
  # loss =  loss  + reg_loss

  # Compute focal loss instead of standard BCE
  # loss_fl = focal_loss(scores, target_labels)
  # reg_loss = reg_param * model.reg_loss(out, target_triplets)
  # loss = loss_fl + reg_loss

  ############################################
  # focal loss + hard-negative mining | NEW! #
  ############################################
  raw_logits = model.distmult(out, target_triplets)
  probs = torch.sigmoid(raw_logits)
  # Per-sample focal loss (no reduction)
  fl_all = focal_loss(raw_logits, target_labels, alpha, gamma, reduction='none')
  # Separate positives and negatives
  pos_mask = target_labels.bool()
  neg_mask = ~pos_mask
  # Positive focal loss averaged
  pos_loss = fl_all[pos_mask].mean()
  # Adversarial weighting for negatives
  neg_probs = probs[neg_mask]
  neg_weights = torch.softmax(neg_probs * alpha_adv, dim=0)
  # Weighted negative focal loss
  neg_loss = (neg_weights * fl_all[neg_mask]).sum()
  # Regularization loss
  reg_loss = reg_param * model.reg_loss(out, target_triplets)
  # Total loss
  loss = pos_loss + neg_loss + reg_loss
  # Use probabilities for metrics
  scores = probs
  
  scores = torch.sigmoid(scores)

  metrics["Loss"]   = loss.item()
  metrics["Auroc"]  = binary_auroc(scores, target_labels).item()
  metrics["Auprc"]  = binary_auprc(scores, target_labels).item()

  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), gradnorm)
  optimizer.step()

  return metrics

def test(model, reg_param, x_dict, index , target_triplets, target_labels, train_val_triplets, alpha, gamma, alpha_adv, change_points=None,
         use_filtered_eval=False, all_target_triplets=None, num_entities=None, standardized_negatives=False):    
  metrics = {"Auroc":{},"Auprc":{},"Loss":{}}

  model.eval()
  with torch.no_grad():
    if change_points != None:
      out = model(x_dict, index, change_points)
    else:
      out = model(x_dict, index)
    scores = model.distmult(out, target_triplets)

    # loss = model.score_loss(scores, target_labels)
    # reg_loss = (reg_param * model.reg_loss(out, target_triplets))
    # loss =  loss  + reg_loss

    # Compute focal loss instead of standard BCE
    # loss_fl = focal_loss(scores, target_labels)
    # reg_loss = reg_param * model.reg_loss(out, target_triplets)
    # loss = loss_fl + reg_loss

    ############################################
    # focal loss + hard-negative mining | NEW! #
    ############################################
    raw_logits = model.distmult(out, target_triplets)
    probs = torch.sigmoid(raw_logits)
    # Per-sample focal loss (no reduction)
    fl_all = focal_loss(raw_logits, target_labels, alpha, gamma, reduction='none')
    # Separate positives and negatives
    pos_mask = target_labels.bool()
    neg_mask = ~pos_mask
    # Positive focal loss averaged
    pos_loss = fl_all[pos_mask].mean()
    # Adversarial weighting for negatives
    neg_probs = probs[neg_mask]
    neg_weights = torch.softmax(neg_probs * alpha_adv, dim=0)
    # Weighted negative focal loss
    neg_loss = (neg_weights * fl_all[neg_mask]).sum()
    # Regularization loss
    reg_loss = reg_param * model.reg_loss(out, target_triplets)
    # Total loss
    loss = pos_loss + neg_loss + reg_loss
    # Use probabilities for metrics
    scores = probs

    scores = torch.sigmoid(scores)
  

  # Subito prima della chiamata evaluation_metrics in test()
  src, _, dst = train_val_triplets.T
  unique_nodes = torch.unique(torch.cat((src, dst)))
  # print(f"[DEBUG] unique_nodes: {unique_nodes.size(0)}, train_val_triplets shape: {train_val_triplets.shape}, device: {train_val_triplets.device}")
  
  if use_filtered_eval and all_target_triplets is not None and num_entities is not None:
    # --- NEW: Filtered evaluation (standard in KGE literature) ---
    # Usa tutti i nodi del grafo come candidati, con filtered setting
    all_graph_nodes = torch.arange(num_entities, device=target_triplets.device)
    # Estrai solo le triple positive dal batch (target_triplets contiene pos+neg)
    pos_mask = target_labels.bool()
    test_positives = target_triplets[pos_mask]
    
    if USE_EVAL_TYPE_CONSTRAINED:
      # print("[i] Using type-constrained filtered evaluation")
      filtered_results = evaluation_metrics_filtered_typeconstrained(
          model, out, all_target_triplets, test_positives,
          all_graph_nodes, target_triplets.device, hits_k=[1, 3, 10]
      )
    else:
      # print("[i] Using standard filtered evaluation")
        filtered_results = evaluation_metrics_filtered(
            model, out, all_target_triplets, test_positives,
            all_graph_nodes, target_triplets.device, hits_k=[1, 3, 10]
        )
    mrr = filtered_results['mrr']
    hits = {k: filtered_results[f'hits@{k}'] for k in [1, 3, 10]}
  else:
    # --- OLD: evaluation_metrics originale (mantenuta per backward compatibility) ---
    # è nomrale che copn sole 20 random avrò valori di mrr e hits molto variabili tra run!!!
    NUM_GENERATE = 200  # [100-200]  # 20 legacy
    mrr, hits = evaluation_metrics_sampled(model, out, train_val_triplets, target_triplets[...], NUM_GENERATE, 0, hits=[1,3,10], standardized_negatives=standardized_negatives)

  # # aggiungi MRR affidabile solo sul test set finale
  # all_graph_nodes = torch.arange(num_entities)
  # mrr_full, hits_full = evaluation_metrics_full(
  #     model, 
  #     model(flattened_features_per_type, train_index),  # ricalcola embeddings
  #     all_graph_nodes,
  #     testing_triplets[test_labels.bool()],  # solo i positivi
  #     device,
  #     hits=[1, 3, 10]
  # )

  # print(f"MRR (full graph): {mrr_full:.3f}, Hits@10 (full graph): {hits_full[10]:.3f}")

  metrics["Loss"]   = loss.item()
  metrics["MRR"]    = mrr
  metrics["Hits@"]  = hits
  metrics["Auroc"]  = binary_auroc(scores, target_labels).item()
  metrics["Auprc"]  = binary_auprc(scores, target_labels).item()
  
  return metrics


# Import the helper function from utils_v2
from src.utils import get_edge_type

def eval(model, flattened_features_per_type, train_index, edge_index, ent2id, relation2id, change_points=None, task=None):
    """
    Rank entities by prediction confidence for novel interactions of a given task type.
    
    Args:
        model: The trained model.
        flattened_features_per_type: Node features by type.
        train_index: Training edge index.
        edge_index: Original edge index DataFrame with 'head', 'interaction', 'tail' columns.
        ent2id: Entity to ID mapping.
        relation2id: Relation to ID mapping.
        change_points: Change points for RGAT (optional).
        task: The task/edge type to evaluate (e.g., 'Compound-ExtGene'). 
              If None, uses args.task.
              
    Note:
        Edge types are derived on-the-fly from entity prefixes (e.g., 'Compound::x' -> 'Compound').
        No 'type' column is needed in the edge_index DataFrame.
    """
    model.eval()
    
    # Use provided task or fall back to args.task
    eval_task = task if task is not None else args.task
    
    def normalize(s):
        """Normalize string for matching: lowercase, replace spaces with underscores"""
        return str(s).lower().replace(" ", "_")
    
    # Build normalized set of targets (including reversed edge types)
    eval_task_normalized = {normalize(eval_task)}
    if '-' in eval_task:
        parts = eval_task.split('-')
        if len(parts) == 2:
            eval_task_normalized.add(normalize(f"{parts[1]}-{parts[0]}"))
    
    print(f"[i] Evaluating model for novel {eval_task} interactions...")
    with torch.no_grad():
        # Get embeddings from the model
        if change_points != None:
            out = model(flattened_features_per_type, train_index, change_points)
        else:
            out = model(flattened_features_per_type, train_index)
        
        # Filter edges by task - match either edge type OR interaction name
        task_mask = edge_index.apply(
            lambda row: (
                normalize(get_edge_type(row['head'], row['tail'])) in eval_task_normalized or
                normalize(row['interaction']) in eval_task_normalized
            ),
            axis=1
        )
        task_edges = edge_index[task_mask]
        
        if len(task_edges) == 0:
            # Show available options for debugging
            available_types = edge_index.apply(
                lambda row: get_edge_type(row['head'], row['tail']), axis=1
            ).unique().tolist()
            available_interactions = edge_index['interaction'].unique().tolist()
            print(f"[WARNING] No edges found for task '{eval_task}'.")
            print(f"  Available edge types: {available_types}")
            print(f"  Available interactions: {available_interactions}")
            return []
        
        # Get unique head and tail entities for this task
        heads = set(task_edges['head'].unique())
        tails = set(task_edges['tail'].unique())
        
        # Determine the relation name dynamically from the dataset
        # Use the most common interaction type for this edge type
        relation_name = task_edges['interaction'].mode().iloc[0] if len(task_edges) > 0 else "TARGET"
        relation_name = relation_name.replace(" ", "_")  # Normalize whitespace
        
        if relation_name not in relation2id:
            print(f"[WARNING] Relation '{relation_name}' not found in relation2id. Available: {list(relation2id.keys())}")
            # Try to find a matching relation
            possible_relations = [r for r in relation2id.keys() if not r.startswith('rev_')]
            if possible_relations:
                relation_name = possible_relations[0]
                print(f"[i] Using fallback relation: {relation_name}")
            else:
                return []
        
        rel_id = relation2id[relation_name]
        
        # Generate all possible triplets for this task
        all_triplets = []
        for head_entity in heads:
            for tail_entity in tails:
                if head_entity in ent2id and tail_entity in ent2id:
                    head_id = ent2id[head_entity]
                    tail_id = ent2id[tail_entity]
                    all_triplets.append([head_id, rel_id, tail_id])
        
        if not all_triplets:
            return []
        
        all_triplets = torch.tensor(all_triplets)

        # Score in batches and offload to CPU to avoid OOM on large head×tail grids
        # (e.g. Task A ~3.7k compounds × ~3.5k proteins ≈ 13M triplets → ~6 GB in one shot).
        score_batch = 200000
        score_chunks = []
        for start in range(0, all_triplets.size(0), score_batch):
            batch_t = all_triplets[start:start + score_batch].to(device)
            s = torch.sigmoid(model.distmult(out, batch_t))
            score_chunks.append(s.detach().cpu())
            del batch_t
        scores = torch.cat(score_chunks)
        
        # Create ranked list of predictions
        ranked_predictions = []
        id2ent = {v: k for k, v in ent2id.items()}
        
        for score, triplet in zip(scores, all_triplets):
            head_id = triplet[0].item()
            tail_id = triplet[2].item()
            
            head = id2ent.get(head_id, f"unknown_head_{head_id}")
            tail = id2ent.get(tail_id, f"unknown_tail_{tail_id}")
            
            ranked_predictions.append({
                'head': head,
                'tail': tail,
                'confidence': score.item(),
                'relation': eval_task
            })
        
        # Sort by confidence (descending)
        ranked_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        print(f"[i] Total predictions made: {len(ranked_predictions)}")
        
        return ranked_predictions

def main(model_name, dataset_tsv, task, runs, epochs, patience, validation_size, test_size, quiet, \
  evaluate_every, negative_rate, model_save_path, oversample_rate, undersample_rate, \
  pretrain_epochs, freeze_base, alpha, gamma, alpha_adv, early_stopping, min_delta, eval_filtered=False,
  split_seed=None, select_metric='loss', train_negative_rate=None, disjoint_supervision=0.0,
  detect_anomaly=False, warm_eval=False, learning_rate=None):
  """
  Protocol knobs (defaults = legacy PathogenKG behaviour, "v1"):
    split_seed           None -> split/undersampling/val-test negatives follow the run seed (v1);
                         int  -> fixed data split across runs, only model init varies.
    select_metric        'loss' -> checkpoint/early stopping on validation loss (v1);
                         'mixed' -> on validation M = 0.2 AUROC + 0.4 AUPRC + 0.4 MRR (as the HPO).
    train_negative_rate  None -> same as negative_rate (v1); int -> negatives per positive in
                         TRAINING only (validation/test keep negative_rate, e.g. 1:1).
    disjoint_supervision 0 -> training target edges are also in the message-passing graph (v1);
                         r in (0,1) -> see SupervisionSampler.
    detect_anomaly       autograd anomaly detection (debug only; was always on, very slow).
    warm_eval            also report test metrics excluding cold-start triples.
    learning_rate        None -> the config's learning rate; float -> override (e.g. baselines).
  """
  all_run_metrics = []
  if disjoint_supervision > 0 and model_name == 'rgat':
    raise ValueError("--disjoint_supervision is not supported for rgat (needs relation-sorted change points)")
  train_neg_rate = negative_rate if train_negative_rate is None else train_negative_rate
  protocol = {
    "split_seed": split_seed, "select_metric": select_metric, "negative_rate_eval": negative_rate,
    "negative_rate_train": train_neg_rate, "oversample_rate": oversample_rate,
    "undersample_rate": undersample_rate, "disjoint_supervision": disjoint_supervision,
    "early_stopping": early_stopping, "patience": patience, "epochs": epochs,
    "alpha": alpha, "gamma": gamma, "alpha_adv": alpha_adv, "config": CONFIG_NAME, "model": model_name,
    "learning_rate_override": learning_rate,
  }
  print(f"[i] Protocol: {protocol}")

  # Select negative sampler without shadowing imported function names.
  if args.negative_sampling == 'filtered' and USE_ALTERNATIVE_NEG_SAMPLING:
    print("[i] Using filtered negative sampling (only non-target edges) for training.")
    neg_sampler = negative_sampling_filtered
  else:
    print("[i] Using standard negative sampling (all edges) for training.")
    neg_sampler = negative_sampling

  if eval_filtered:
    print("[i] Using FILTERED evaluation metrics (standard KGE setting).")
  else:
    print("[i] Using LEGACY evaluation metrics (pool-based, unfiltered).")

    
  # Multi-relational pretraining phase
  model_state = None
  if pretrain_epochs > 0:
      print(f'[i] Pretraining for {pretrain_epochs} epochs on all relations...')

      # Prepare pretrain dataset
      (pt_in_channels, pt_num_nodes_per_type, pt_num_entities, pt_num_relations,
       pt_train_triplets, pt_full_index, pt_flattened_feats,
        _, _, _, _, _, _, _) = get_dataset_pretrain(dataset_tsv, quiet)
      
      # Initialize model
      pre_model, lr, regularization, grad_norm = get_model(model_name, 
                                                           task, 
                                                           pt_in_channels,
                                                           pt_num_nodes_per_type,
                                                           pt_num_entities, 
                                                           pt_num_relations,
                                                           config_name=CONFIG_NAME
                                                           )
      pre_model = pre_model.to(device)

      # Pretraining loop
      optimizer_pre = torch.optim.Adam(pre_model.parameters(), lr=lr)


      change_points_pre = None
      if model_name == 'rgat':
        # pt_full_index è un numpy array [src, rel, dst]
        rel_ids = torch.from_numpy(pt_full_index[:,1]).to(device)
        change_points_pre = torch.cat([
            torch.tensor([0], device=device),
            (rel_ids[1:] != rel_ids[:-1]).nonzero(as_tuple=False).view(-1) + 1,
            torch.tensor([rel_ids.size(0)], device=device)
        ])

      pt_full_index = torch.tensor(pt_full_index).to(device)
      pt_flattened_feats = {node_type:(features.to(device) if features is not None else None) 
                      for node_type, features in pt_flattened_feats.items()}

      for epoch in tqdm(range(1, pretrain_epochs+1)):
          # negative sampling on all relations
          neg_triplets, neg_labels = neg_sampler(pt_train_triplets.cpu(), negative_rate)
          neg_triplets, neg_labels = neg_triplets.to(device), neg_labels.to(device)

          train(pre_model, optimizer_pre, grad_norm, \
                regularization, pt_flattened_feats, \
                pt_full_index, neg_triplets, neg_labels, \
                alpha, gamma, alpha_adv, change_points_pre)

      # Save pretrained weights
      pretrain_path = model_save_path.replace('.pt', '_pretrained.pt')
      torch.save(pre_model.state_dict(), pretrain_path)
      print(f'[i] Pretrained model saved to {pretrain_path}')
      # For fine-tuning: load weights
      model_state = torch.load(pretrain_path)

  for i in range(runs):
    # --dry_run leaves model_save_path unset: nothing is written, so keep the path as None
    run_model_save_path = model_save_path.replace(".pt", f"_run{i}.pt") if model_save_path else None
    random_seed = BASE_SEED + i
    set_seed(random_seed)
    print(f'[i] Using random seed {random_seed}')
    
    # Get dataset
    print('[i] Getting dataset...', end='', flush=True)
    start_time_dataset = time.time()

    # data seed: split + undersampling mask + val/test negatives. Legacy = run seed (changes per run).
    data_seed = random_seed if split_seed is None else int(split_seed)
    in_channels_dict, num_nodes_per_type, num_entities, num_relations, \
    train_triplets, train_index, flattened_features_per_type, val_triplets, \
    train_val_triplets, test_triplets, train_val_test_triplets, \
    edge_index, ent2id, relation2id = get_dataset(dataset_tsv, task, validation_size, test_size, quiet, data_seed, oversample_rate, undersample_rate)
    if split_seed is not None:
      # fixed split: make model init depend only on the run seed, whatever get_dataset consumed
      set_seed(random_seed)
  
    # --- preparazione parametri per neg sampling corretto ---
    all_entities_arr = np.arange(num_entities)
    all_true_arr = train_val_test_triplets.cpu().numpy()
      
    end_time_dataset = round(time.time() - start_time_dataset, 2)
    # print(f'ok ({end_time_dataset} seconds)')
    
    # Model definition
    model, lr, regularization, grad_norm, weight_decay, scheduler_gamma = get_model(model_name, task, in_channels_dict, num_nodes_per_type, num_entities, num_relations, config_name=CONFIG_NAME)
    if learning_rate is not None:
      lr = learning_rate

    # Move to device
    # model                       = model.to(device)
    # train_index                 = train_index.to(device)
    if model_state is not None:
      model.load_state_dict(model_state, strict=False)
      if freeze_base:
          for name, param in model.named_parameters():
              if 'conv_layers' in name or 'relation_embeddings_per_layer' in name:
                  param.requires_grad = False
    model = model.to(device)
    train_index = train_index.to(device)
    flattened_features_per_type = {node_type:(features.to(device) if features!= None else None) for node_type,features in flattened_features_per_type.items()}
    
    change_points = None
    if model_name == 'rgat':
      rel_ids_sorted = train_index[:, 1]
      change_points = torch.cat([
          torch.tensor([0], device=device),
          (rel_ids_sorted[1:] != rel_ids_sorted[:-1]).nonzero(as_tuple=False).view(-1) + 1,
          torch.tensor([rel_ids_sorted.size(0)], device=device) 
      ])
      
    # Optimizer definition
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)




    # Training runs
    # Anomaly detection is a debugging tool that slows autograd considerably; it used to be
    # always on. It does not change the computed values.
    torch.autograd.set_detect_anomaly(bool(detect_anomaly))
    val_metrics       = {"Auroc":0, "Auprc":0, "Loss":0, "MRR":0, "Hits@":0}
    last_improvement_epoch = 0
    select_on_loss = (select_metric == 'loss')
    best_val_loss = float("inf") if select_on_loss else -float("inf")   # best value of the selection metric
    best_epoch = None
    best_model_found  = False
    sup_sampler = None
    if disjoint_supervision and disjoint_supervision > 0:
      sup_sampler = SupervisionSampler(train_index, train_triplets, num_relations, disjoint_supervision,
                                       oversample_rate, random_seed)
    run_start = time.time()
    with trange(1, (epochs + 1), desc=f'Run {i} | Epochs', position=0) as epochs_tqdm:
      for epoch in epochs_tqdm:
        if sup_sampler is not None:
          epoch_index, epoch_positives = sup_sampler.sample(epoch)
        else:
          epoch_index, epoch_positives = train_index, train_triplets
        if not USE_ALTERNATIVE_NEG_SAMPLING:
          training_triplets, train_labels = neg_sampler(epoch_positives, train_neg_rate)
        else:
          training_triplets, train_labels = neg_sampler(epoch_positives, all_entities_arr, train_neg_rate, all_true_arr, seed=random_seed + epoch)

        training_triplets, train_labels = training_triplets.to(device), train_labels.to(device)
        # Train
        train_metrics = train(
            model,
            optimizer,
            grad_norm,
            regularization,
            flattened_features_per_type,
            epoch_index,
            training_triplets,
            train_labels,
            alpha,
            gamma,
            alpha_adv,
            change_points
          )
        scheduler.step()
        # Validate
        if epoch%evaluate_every==0:
          if not USE_ALTERNATIVE_NEG_SAMPLING:
             validation_triplets, val_labels = neg_sampler(val_triplets, negative_rate)
          else:
             validation_triplets, val_labels = neg_sampler(val_triplets, all_entities_arr, negative_rate, all_true_arr, seed=data_seed + 1000)

          validation_triplets, val_labels = validation_triplets.to(device), val_labels.to(device)
          val_metrics = test(
            model,
            regularization,
            flattened_features_per_type,
            train_index,
            validation_triplets,
            val_labels,
            train_val_triplets,
            alpha,
            gamma,
            alpha_adv,
            change_points,
            use_filtered_eval=eval_filtered,
            all_target_triplets=train_val_test_triplets if eval_filtered else None,
            num_entities=num_entities if eval_filtered else None
          )
          # Model selection / early stopping: 'loss' (legacy) or validation M (as in the HPO)
          if select_on_loss:
            val_loss = val_metrics["Loss"]
            improved = val_loss < (best_val_loss - min_delta)
          else:
            val_loss = mixed_metric(val_metrics)
            improved = val_loss > (best_val_loss + min_delta)
          print(f"[val] run {i} epoch {epoch} | loss {val_metrics['Loss']:.4f} | AUROC {val_metrics['Auroc']:.4f} "
                f"| AUPRC {val_metrics['Auprc']:.4f} | MRR {val_metrics['MRR']:.4f} | M {mixed_metric(val_metrics):.4f}")
          if improved:
            best_val_loss = val_loss
            last_improvement_epoch = epoch
            best_epoch = epoch
            # model is saved only if validation improves (and only when not a dry run)
            if run_model_save_path:
              torch.save(model.state_dict(), run_model_save_path)
              print("[i] Best model updated.")
              best_model_found = True
          elif early_stopping:
            if (epoch - last_improvement_epoch) >= patience:
              print(f"[i] Early stopping triggered at epoch {epoch} (patience={patience} epochs, min_delta={min_delta}).")
              break
        else:
          val_loss = val_metrics["Loss"] if select_on_loss else mixed_metric(val_metrics)

        epochs_tqdm.set_postfix(loss=train_metrics["Loss"], Tr_Auroc=train_metrics["Auroc"], Tr_Auprc=train_metrics["Auprc"],
                                Val_Auroc=val_metrics["Auroc"], Val_Auprc=val_metrics["Auprc"], Val_mrr=val_metrics["MRR"], Val_hits=val_metrics["Hits@"],
                                metric=val_loss, best_metric=best_val_loss)

      train_time = time.time() - run_start
      # Load best model
      if best_model_found:
        model.load_state_dict(torch.load(run_model_save_path))
      # Test best model
      model.eval()
      with torch.no_grad():
          if not USE_ALTERNATIVE_NEG_SAMPLING:
            testing_triplets, test_labels = neg_sampler(test_triplets, negative_rate)
          else:
            testing_triplets, test_labels = neg_sampler(test_triplets, all_entities_arr, negative_rate, all_true_arr, seed=data_seed + 2000)

          testing_triplets, test_labels = testing_triplets.to(device), test_labels.to(device)
          metrics = test(
            model,
            regularization,
            flattened_features_per_type,
            train_index,
            testing_triplets,
            test_labels,
            train_val_test_triplets,
            alpha,
            gamma,
            alpha_adv,
            change_points,
            use_filtered_eval=eval_filtered,
            all_target_triplets=train_val_test_triplets if eval_filtered else None,
            num_entities=num_entities if eval_filtered else None
          )  
      print(f"Run {i} | Test Auroc: {metrics['Auroc']:.3f}, Test Auprc: {metrics['Auprc']:.3f}, Test MRR: {metrics['MRR']:.3f}, TEST HITS: {metrics['Hits@']}")
      if eval_filtered:
        pool = "type-constrained: candidates = nodes occurring in the target relation" if USE_EVAL_TYPE_CONSTRAINED \
               else f"all {num_entities} nodes"
        print(f"         (filtered setting — ranking against {pool})")
      run_record = {
          "Auroc": metrics["Auroc"],
          "Auprc": metrics["Auprc"],
          "MRR": metrics["MRR"],
          "Hits@1": metrics["Hits@"][1],
          "Hits@3": metrics["Hits@"][3],
          "Hits@10": metrics["Hits@"][10],
          "eval_mode": "filtered" if eval_filtered else "legacy",
          "seed": random_seed, "data_seed": data_seed,
          "best_epoch": best_epoch, "train_time_sec": round(train_time, 1),
      }
      print(f"Run {i} | best_epoch: {best_epoch} | train_time_sec: {train_time:.1f}")

      if warm_eval and eval_filtered:
        cold = cold_start_mask(test_triplets, train_index, num_relations)
        n_cold = int(cold.sum().item())
        warm_triplets = torch.as_tensor(test_triplets)[~cold]
        print(f"Run {i} | cold-start test triples: {n_cold}/{len(cold)} ({n_cold / max(1, len(cold)):.2%})")
        if warm_triplets.size(0) > 0:
          with torch.no_grad():
            if not USE_ALTERNATIVE_NEG_SAMPLING:
              warm_tt, warm_labels = neg_sampler(warm_triplets, negative_rate)
            else:
              warm_tt, warm_labels = neg_sampler(warm_triplets, all_entities_arr, negative_rate, all_true_arr, seed=data_seed + 3000)
            warm = test(model, regularization, flattened_features_per_type, train_index,
                        warm_tt.to(device), warm_labels.to(device), train_val_test_triplets,
                        alpha, gamma, alpha_adv, change_points, use_filtered_eval=True,
                        all_target_triplets=train_val_test_triplets, num_entities=num_entities)
          print(f"Run {i} | WARM Test Auroc: {warm['Auroc']:.3f}, Test Auprc: {warm['Auprc']:.3f}, Test MRR: {warm['MRR']:.3f}, TEST HITS: {warm['Hits@']}")
          run_record.update({"warm_Auroc": warm["Auroc"], "warm_Auprc": warm["Auprc"], "warm_MRR": warm["MRR"],
                             "warm_Hits@1": warm["Hits@"][1], "warm_Hits@3": warm["Hits@"][3],
                             "warm_Hits@10": warm["Hits@"][10], "cold_start_fraction": n_cold / max(1, len(cold))})
      all_run_metrics.append(run_record)
      print(f"[i] Completed run {i}/{runs}")

      # print(f"[i] Saving best model of run {i} to {run_model_save_path}")

      # Optional: full head×tail ranking after each run (heavy). Off by default —
      # use drug_eval.py for batched, compound-centric ranking instead.
      if getattr(args, 'rank_after_train', False):
          print(f"[i] Evaluating model on entire dataset for ranking...")
          # Eval model on entire dataset and get ranking
          rank = eval(model, flattened_features_per_type, train_index, edge_index, ent2id, relation2id, change_points, task=task)
          # Show better print predictions
          top_p_num = min(10, len(rank))
          print(f"\n[i] Top {top_p_num} predictions:")
          for idx, pred in enumerate(rank[:top_p_num], 1):
              print(f"  {idx}. Head: {pred['head']:30s} -> Tail: {pred['tail']:20s} | Confidence: {pred['confidence']:.4f}")
          # Save ranking to JSON
          ranking_save_path = run_model_save_path.replace(".pt", "_ranking.json")
          with open(ranking_save_path, "w") as f:
              json.dump(rank, f, indent=4)
          print(f"[i] Saved ranking to {ranking_save_path}")


  metric_keys = ["Auroc", "Auprc", "MRR", "Hits@1", "Hits@3", "Hits@10"]
  avg_metrics = {key: float(np.mean([m[key] for m in all_run_metrics])) for key in metric_keys}
  std_metrics = {key: float(np.std([m[key] for m in all_run_metrics])) for key in metric_keys}

  # Salvataggio su JSON
  result_to_save = {
      "protocol": protocol,
      "individual_runs": all_run_metrics,
      "average_metrics": avg_metrics,
      "std_metrics": std_metrics
  }
  if model_save_path:      # --dry_run writes nothing to disk
    json_save_path = model_save_path.replace(".pt", "_metrics.json")
    with open(json_save_path, "w") as f:
        json.dump(result_to_save, f, indent=4)
    # print(f"Saved metrics to {json_save_path}")

  return None

if __name__ == '__main__':
  print(f'[i] Running on {device}')

  parser = argparse.ArgumentParser(description='Ablation study on Vitagraph generation process.')
  # add 

  parser.add_argument('--tsv', type=str, default=DEFAULT_TRAIN_TSV,
                      help=f"Path to the training TSV (default: {DEFAULT_TRAIN_TSV})")
  parser.add_argument('-m', '--model', type=str, default='compgcn', choices=['rgcn', 'rgat', 'compgcn', 'distmult'], help='Model to use (distmult = embedding-only baseline, no message passing).')
  parser.add_argument('-r', '--runs', type=int, default=1, help='Number of runs for the ablation study.')
  parser.add_argument('-e', '--epochs', type=int, default=400, help='Number of epochs for the ablation study.')
  parser.add_argument('-p', '--patience', type=int, default=20, help='Patience in epochs for early stopping (used only with --early_stopping).')
  parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping based on validation performance.')
  parser.add_argument('--min_delta', type=float, default=0.0, help='Minimum validation-loss decrease required to reset early-stopping patience.')
  parser.add_argument('--validation_size', type=float, default=0.1, help='Validation size for the ablation study.')
  parser.add_argument('--test_size', type=float, default=0.2, help='Test size for the ablation study.')
  parser.add_argument('--quiet', action='store_true', help='If set, the ablation study will print debug output.')
  parser.add_argument('--evaluate_every', type=int, default=5, help='Evaluate every n epochs.')
  parser.add_argument('--negative_sampling', type=str, default='filtered', help='')  
  parser.add_argument('--negative_rate', type=float, default=1, help='Negative sampling rate for the ablation study.')
  parser.add_argument('--oversample_rate', type=int, default=5, help='how many times to repeat the positive training triplets')
  parser.add_argument('--undersample_rate', type=float, default=0.5, help='fraction [0,1] of non-target triplets to keep in the training graph')
  parser.add_argument('--pretrain_epochs', type=int, default=0, help='Number of epochs for multi-relational pretraining')
  parser.add_argument('--freeze_base', action='store_true', help='Freeze pre-trained conv layers during fine-tuning')
  parser.add_argument('--alpha', type=float, default=0.25, help='Alpha value of the focal loss')
  parser.add_argument('--gamma', type=float, default=3.0, help='Gamma value of the focal loss')
  parser.add_argument('--alpha_adv', type=float, default=2.0, help='Alpha value for the hard-negative mining loss'), 
  parser.add_argument('--dry_run', action='store_true', help='If set, the ablation study will not save models or rankings, and will skip final evaluation.')
  parser.add_argument('--rank_after_train', action='store_true',
                      help='After each run, rank ALL task head×tail triplets and save a *_ranking.json. '
                           'Off by default (heavy: materialises the full grid, e.g. ~13M for DTI). '
                           'Prefer drug_eval.py for batched, compound-centric ranking.')
  # ---- protocol v2 knobs (defaults reproduce the legacy v1 behaviour; see main() docstring) ----
  parser.add_argument('--split_seed', type=int, default=None,
                      help='Fixed seed for the data split / undersampling / val-test negatives. '
                           'Default None = legacy: follows the run seed, so the split changes at every run.')
  parser.add_argument('--select_metric', type=str, default='loss', choices=['loss', 'mixed'],
                      help="Checkpoint selection and early stopping on validation 'loss' (legacy) or "
                           "'mixed' = M = 0.2 AUROC + 0.4 AUPRC + 0.4 MRR (same criterion as the HPO).")
  parser.add_argument('--train_negative_rate', type=int, default=None,
                      help='Negatives per positive in TRAINING only (val/test keep --negative_rate). '
                           'Default None = same as --negative_rate.')
  parser.add_argument('--disjoint_supervision', type=float, default=0.0,
                      help='Fraction of training target edges removed from the message-passing graph at '
                           'each epoch and used only as loss positives (0 = legacy: all in the graph).')
  parser.add_argument('--detect_anomaly', action='store_true',
                      help='Enable torch autograd anomaly detection (debug only; slow). Legacy code had it always on.')
  parser.add_argument('--learning_rate', type=float, default=None,
                      help='Override the learning rate of the selected config (default: use the config value).')
  parser.add_argument('--warm_eval', action='store_true',
                      help='Also report test metrics excluding cold-start triples (endpoint without edges in the training graph).')
  parser.add_argument('--eval_filtered', action='store_true', default=True,
                      help='Use filtered evaluation metrics (standard KGE setting). '
                           'Ranks against all graph nodes with known positives masked. '
                           'If not set, uses legacy pool-based evaluation.')

  # add task as argument
  parser.add_argument('--task', type=str,
                      #default='Compound-ExtGene',
                      default='TARGET',
                      help='Task to perform. Could be a comma-separated list of edge types or interaction names (e.g., "CMP_BIND,ENZYME"). If not specified, defaults to "TARGET".')

  # hyperparameter config selector (key in src/models_params.json). For the PKT
  # subgraphs (DRKG-scale biomedical KG, ~66-94k nodes) use e.g. BIOKG-128 / BIOKG-200.
  parser.add_argument('--config', type=str, default=CONFIG_NAME,
                      help=f'Hyperparameter config name from src/models_params.json (default: {CONFIG_NAME}). '
                           'For PKT subgraphs try BIOKG-64 / BIOKG-128 / BIOKG-200.')


  args            = parser.parse_args()
  model           = args.model
  runs            = args.runs
  epochs          = args.epochs
  patience        = args.patience
  validation_size = args.validation_size
  test_size       = args.test_size
  quiet           = args.quiet
  evaluate_every  = args.evaluate_every
  negative_rate   = args.negative_rate
  dataset_tsv     = _resolve_dataset_path(args.tsv)
  oversample_rate = args.oversample_rate
  undersample_rate = args.undersample_rate
  pretrain_epochs = args.pretrain_epochs
  freeze_base     = args.freeze_base
  alpha           = args.alpha
  gamma           = args.gamma
  alpha_adv       = args.alpha_adv
  early_stopping  = args.early_stopping
  min_delta       = args.min_delta
  eval_filtered   = args.eval_filtered
  CONFIG_NAME     = args.config  # override the module-level hyperparameter config (used by get_model)

  # get dataset_name (use os.path for cross-platform compatibility)
  dataset_basename = os.path.basename(args.tsv)
  dataset_name = os.path.splitext(dataset_basename)[0]
  print(f'[i] Using dataset: {dataset_name}')

  task            = args.task  # needed by main() in both dry_run and normal paths

  if args.dry_run:
      print("[i] Dry run enabled: models and rankings will not be saved, and final evaluation will be skipped.")
      model_save_path = None  # No saving
  else:
    time_stamp      = time.strftime('%Y%m%d_%H%M%S')
    task_clean      = task.lower().replace('-', '_').replace(',', '_')
    folder_name = task_clean + '_' + dataset_name + '_' + time_stamp
    model_save_dir = os.path.join('models', folder_name)
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'{model.lower()}.pt')

    # save parameters used for the run
    params_save_path = model_save_path.replace('.pt', '_params.json')
    with open(params_save_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

  if not quiet: print(f'Running training of model: {model}, runs: {runs} | device: {device}')



  """
  Example commands:
  python train_and_eval.py --model rgcn --epochs 1
  python train_and_eval.py --model rgcn --runs 1 --epochs 100
  python train_and_eval.py --model rgcn --runs 3 --epochs 100
  python train_and_eval.py --model rgcn --runs 3 --epochs 100
  python train_and_eval.py --model compgcn --epochs 100 --negative_sampling filtered
  python train_and_eval.py --model compgcn --epochs 100 --negative_sampling filtered --eval_filtered
  python train_and_eval.py --model compgcn --pretrain_epochs 100 --freeze_base --epochs 200

  -----------------------------------------------------------
  # optimized training commands: 
  python train_and_eval.py --model compgcn --run 3 --epochs 400  --tsv dataset/PathogenKG_n31_core.tsv.zip --early_stopping --negative_sampling filtered

  -----------------------------------------------------------
  
  -- on DRKG (simple triplets):
  python train_and_eval.py --model compgcn --epochs 100 --task CMP_BIND --tsv dataset/drkg/drkg_reduced.zip 
  
  -- on DRKG dataset:
  python train_and_eval.py --model compgcn --epochs 300 --task Compound-Gene --tsv dataset/drkg/drkg.tsv
  
  -- on DRKG_reduced dataset:
  python train_and_eval.py --model compgcn --epochs 300 --task Compound-Gene --tsv dataset/drkg/drkg_reduced.tsv
  python train_and_eval.py --model compgcn --epochs 3 --task Compound-Gene --tsv dataset/drkg/drkg_reduced.tsv

  """

  main(model, dataset_tsv, task, runs, epochs, patience, validation_size, test_size, quiet, \
      evaluate_every, negative_rate, model_save_path, oversample_rate, undersample_rate, \
      pretrain_epochs, freeze_base, alpha, gamma, alpha_adv, early_stopping, min_delta, eval_filtered,
      split_seed=args.split_seed, select_metric=args.select_metric,
      train_negative_rate=args.train_negative_rate, disjoint_supervision=args.disjoint_supervision,
      detect_anomaly=args.detect_anomaly, warm_eval=args.warm_eval, learning_rate=args.learning_rate)
