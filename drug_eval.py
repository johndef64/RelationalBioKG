"""
Given a pre-trained link prediction model (folder_path), create with @train_and_eval.py.

The purpose of this script is to create a ranking for a compound (drug) and all possible protein targets (ExtGene) in the reference dataset: dataset = 'PathogenKG_n34_core.tsv.zip'


The current eval scripts rank compounds using the trained model among the compounds and ExtGens present in the training edges ("interaction"== TARGET). In this script, however, we need to rank a compound among all possible protein targets (ExtGens) present in the reference dataset, regardless of whether or not they are present in the training edges.

Finally, the targets with the highest scores will be evaluated and compared with the targets present in the test edges ("interaction"== TARGET) to assess the model's ability to predict new protein targets for a given compound.


# All compounds
python drug_eval.py --model_folder models/target_PathogenKG_n34_core.tsv_20260225_184723

# Single compound
python drug_eval.py --model_folder models/target_PathogenKG_n34_core.tsv_20260225_184723 --compound "Compound::Pubchem:19"

"""

################# DRUG EVAL ####################

# Evaluate a single compound against ALL ExtGene targets in the full graph.
# Unlike model_eval.py / train_and_eval.py which rank among known TARGET edges,
# this script ranks a compound against every ExtGene in the reference dataset
# and then evaluates recall against the held-out test TARGET edges.

# Example commands:
# python drug_eval.py --model_folder models/target_PathogenKG_n34_core.tsv_20260225_184723
# python drug_eval.py --model_folder models/target_PathogenKG_n34_core.tsv_20260225_184723 --compound "Compound::Pubchem:19"
# python drug_eval.py --model_folder models/target_PathogenKG_n34_core.tsv_20260225_184723 --compound all --topk 50

########################################################

#%%
import warnings
warnings.simplefilter(action='ignore')

import os
import json
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F

from src.utils import (
    set_seed, load_data, entities2id_offset, rel2id_offset,
    edge_ind_to_id, entities_features_flattening,
    set_target_label, triple_sampling, select_target_triplets,
    graph_to_undirect, add_self_loops, get_edge_type
)

from src.hetero_rgcn import HeterogeneousRGCN as rgcn
from src.hetero_rgat import HeterogeneousRGAT as rgat
from src.hetero_compgcn import HeterogeneousCompGCN as compgcn

BASE_SEED = 42

dataset = 'PathogenKG_n31_core.tsv.zip'
DEFAULT_TRAIN_TSV = os.path.join('dataset', dataset)
models_params_path = './src/models_params.json'
CONFIG_NAME = 'pathogen31-cmp-gene'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _resolve_dataset_path(tsv_path: str) -> str:
  resolved = os.path.normpath(tsv_path)
  if not os.path.exists(resolved):
    raise FileNotFoundError(
      f"Dataset TSV not found: {resolved}."
    )
  return resolved


def get_dataset_for_drug_eval(tsv_path, task, validation_size, test_size, quiet, seed, undersample_rate):
  """
  Load the full dataset and prepare the graph for inference.
  Returns everything needed to rebuild the model and score arbitrary triplets,
  plus the train/val/test splits of TARGET edges for evaluation.
  """
  edge_index, node_features_per_type = load_data(tsv_path, {}, quiet)
  edge_index = set_target_label(edge_index, [x for x in task.split(',')])

  ent2id, all_nodes_per_type = entities2id_offset(edge_index, node_features_per_type, quiet)
  relation2id = rel2id_offset(edge_index)
  indexed_edge_index = edge_ind_to_id(edge_index, ent2id, relation2id)
  flattened_features_per_type = entities_features_flattening(node_features_per_type, all_nodes_per_type)

  indexed_edge_index["label"] = edge_index["label"].values

  non_target_triplets, target_triplets = select_target_triplets(indexed_edge_index)
  train_triplets, val_triplets, test_triplets = triple_sampling(
    target_triplets.loc[:, ["head", "interaction", "tail"]].values,
    validation_size, test_size, quiet, seed
  )

  # under-sampling non-target for message-passing graph
  if undersample_rate < 1.0:
    rnd = np.random.RandomState(seed)
    keep = int(len(non_target_triplets) * undersample_rate)
    idx = rnd.choice(len(non_target_triplets), size=keep, replace=False)
    non_target_triplets = non_target_triplets.iloc[idx]

  non_target_triplets = non_target_triplets.loc[:, ["head", "interaction", "tail"]].values
  train_index = np.concatenate([non_target_triplets, train_triplets], axis=0)

  num_nodes_per_type = {node_type: len(nodes) for node_type, nodes in all_nodes_per_type.items()}
  num_entities = sum(num_nodes_per_type.values())

  train_index = graph_to_undirect(train_index, len(relation2id))
  train_index = add_self_loops(train_index, num_entities, len(relation2id))

  train_triplets = torch.tensor(train_triplets)
  val_triplets = torch.tensor(val_triplets)
  test_triplets = torch.tensor(test_triplets)

  in_channels_dict = {
    node_type: (features.shape[1] if features is not None else None)
    for node_type, features in flattened_features_per_type.items()
  }
  num_relations = len(relation2id)

  return (in_channels_dict, num_nodes_per_type, num_entities, num_relations,
          train_triplets, train_index, flattened_features_per_type,
          val_triplets, test_triplets, edge_index, ent2id, relation2id,
          all_nodes_per_type)


def get_model(model_name, task, in_channels_dict, num_nodes_per_type, num_entities, num_relations, config=CONFIG_NAME):
  """Build the model exactly as train_and_eval.py did (same config handling, CompGCN options,
  DistMult baseline), so the saved state_dict always matches the architecture."""
  from train_and_eval import get_model as _train_get_model
  model, lr, regularization, grad_norm, _, _ = _train_get_model(
    model_name, task, in_channels_dict, num_nodes_per_type, num_entities, num_relations, config_name=config)
  return model, lr, regularization, grad_norm


def _legacy_get_model(model_name, task, in_channels_dict, num_nodes_per_type, num_entities, num_relations, config=CONFIG_NAME):
  """Previous local copy of get_model (kept for reference, not used)."""
  with open(models_params_path, 'r') as f:
    models_params = json.load(f)

  if config not in models_params:
    print(f"[get_model] Sweep '{config}' not found, using 'default' config")
    config = "default"

  if config not in models_params:
    raise KeyError("[get_model] Neither requested config nor 'default' found in models_params.json")
  if model_name not in models_params[config]:
    raise KeyError(f"[get_model] Model '{model_name}' not found in config '{config}'")

  model_params = models_params[config][model_name]
  print(f"[get_model] Using parameters from config '{config}': {model_params}")

  conv_hidden_channels = {f'layer_{x}': model_params[f'layer_{x}'] for x in range(model_params['conv_layer_num'])}

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
    )
  else:
    return None

  return model, model_params['learning_rate'], model_params['regularization'], model_params['grad_norm']


def resolve_model_folder(folder_path):
  """
  Given a model folder, find the .pt model file and the _params.json to
  determine model type and training parameters.
  """
  folder_path = os.path.normpath(folder_path)
  if not os.path.isdir(folder_path):
    raise FileNotFoundError(f"Model folder not found: {folder_path}")

  # Find params file
  params_files = [f for f in os.listdir(folder_path) if f.endswith('_params.json')]
  if not params_files:
    raise FileNotFoundError(f"No *_params.json found in {folder_path}")
  params_path = os.path.join(folder_path, params_files[0])

  with open(params_path, 'r') as f:
    train_params = json.load(f)

  model_name = train_params.get('model', 'compgcn')

  # Find model .pt files (exclude pretrained)
  pt_files = [f for f in os.listdir(folder_path)
              if f.endswith('.pt') and 'pretrained' not in f]
  if not pt_files:
    raise FileNotFoundError(f"No .pt model file found in {folder_path}")

  # Select best run by MRR from *_metrics.json, fallback to run0
  best_run_idx = 0
  metrics_files = [f for f in os.listdir(folder_path) if f.endswith('_metrics.json')]
  if metrics_files:
    metrics_path = os.path.join(folder_path, metrics_files[0])
    with open(metrics_path, 'r') as f:
      metrics_data = json.load(f)
    individual_runs = metrics_data.get('individual_runs', [])
    if individual_runs:
      best_run_idx = max(range(len(individual_runs)), key=lambda i: individual_runs[i].get('MRR', 0.0))
      print(f"[i] Best run by MRR: run{best_run_idx} (MRR={individual_runs[best_run_idx].get('MRR', 0.0):.4f})")
  else:
    print(f"[i] No *_metrics.json found, defaulting to run0")

  best_run_files = [f for f in pt_files if f'run{best_run_idx}' in f]
  if not best_run_files:
    run0 = [f for f in pt_files if 'run0' in f]
    model_file = run0[0] if run0 else pt_files[0]
    print(f"[WARNING] run{best_run_idx}.pt not found, falling back to {model_file}")
  else:
    model_file = best_run_files[0]
  model_path = os.path.join(folder_path, model_file)
  train_params['_selected_run'] = best_run_idx if best_run_files else 0

  return model_path, model_name, train_params


def rank_compound_against_all_extgenes(
    model, embeddings, compound_id, all_extgene_ids, rel_id, batch_size=4096
):
  """
  Score a single compound against every ExtGene in the dataset.

  Args:
    model: Trained GNN model.
    embeddings: Node embeddings from model forward pass.
    compound_id: Integer ID of the compound node.
    all_extgene_ids: Tensor of all ExtGene integer IDs.
    rel_id: Integer ID of the TARGET relation.
    batch_size: Batch size for scoring (to avoid OOM).

  Returns:
    scores: Tensor of sigmoid scores, one per ExtGene.
  """
  num_targets = all_extgene_ids.size(0)
  all_scores = []

  for start in range(0, num_targets, batch_size):
    end = min(start + batch_size, num_targets)
    batch_tails = all_extgene_ids[start:end]
    batch_size_actual = batch_tails.size(0)

    triplets = torch.stack([
      torch.full((batch_size_actual,), compound_id, dtype=torch.long, device=device),
      torch.full((batch_size_actual,), rel_id, dtype=torch.long, device=device),
      batch_tails
    ], dim=1)

    scores = model.distmult(embeddings, triplets)
    scores = torch.sigmoid(scores)
    all_scores.append(scores)

  return torch.cat(all_scores, dim=0)


def evaluate_compound(
    compound_name, compound_id, model, embeddings,
    all_extgene_ids, all_extgene_names, rel_id,
    test_target_tails, topk, id2ent,
    known_positive_tails=None
):
  """
  Rank a compound against all ExtGenes and compute filtered type-constrained
  metrics against the test set.

  Filtered evaluation (standard in KGE literature): when computing the rank
  of a test target, known positive tails (train+val) are excluded from the
  ranking so they don't artificially inflate the rank.

  Args:
    compound_name: String name of the compound.
    compound_id: Integer ID.
    model: Trained model.
    embeddings: Precomputed embeddings.
    all_extgene_ids: Tensor of all ExtGene IDs.
    all_extgene_names: List of all ExtGene string names (aligned with all_extgene_ids).
    rel_id: TARGET relation ID.
    test_target_tails: Set of ExtGene IDs that are true targets in the test set for this compound.
    topk: Number of top predictions to report.
    id2ent: ID to entity name mapping.
    known_positive_tails: Set of ExtGene IDs that are known positives (train+val)
        for this compound. Used for filtered ranking. If None, no filtering is applied.

  Returns:
    Dict with ranking results and evaluation metrics.
  """
  if known_positive_tails is None:
    known_positive_tails = set()

  scores = rank_compound_against_all_extgenes(
    model, embeddings, compound_id, all_extgene_ids, rel_id
  )

  # Sort by descending score
  sorted_indices = torch.argsort(scores, descending=True)
  sorted_scores = scores[sorted_indices]
  sorted_extgene_ids = all_extgene_ids[sorted_indices]

  # Build full ranking (unfiltered, for display / top-k inspection)
  ranking = []
  for i in range(len(sorted_indices)):
    ext_id = sorted_extgene_ids[i].item()
    ranking.append({
      'rank': i + 1,
      'head': compound_name,
      'tail': id2ent.get(ext_id, f"unknown_{ext_id}"),
      'tail_id': ext_id,
      'confidence': sorted_scores[i].item(),
      'is_test_target': ext_id in test_target_tails,
      'is_known_positive': ext_id in known_positive_tails,
      'relation': 'TARGET'
    })

  # Compute filtered type-constrained metrics against test targets
  num_test_targets = len(test_target_tails)
  metrics = {
    'compound': compound_name,
    'num_extgenes_scored': len(all_extgene_ids),
    'num_test_targets': num_test_targets,
  }

  if num_test_targets > 0:
    # Filtered ranking: for each test target, compute its rank after
    # removing known positives (train+val) from the ranking.
    # This is the standard "filtered" setting in KGE evaluation.
    filtered_ranks_of_true = []
    for test_tail in test_target_tails:
      # Score of the true test target
      test_tail_tensor = torch.tensor(test_tail, device=sorted_extgene_ids.device)
      mask = sorted_extgene_ids == test_tail_tensor
      if not mask.any():
        continue
      true_score = sorted_scores[mask][0]

      # Filtered: exclude known positives (train+val) AND other test targets
      # from corrupting the ranking, keeping only the current test target
      filter_set = (known_positive_tails | test_target_tails) - {test_tail}
      # Count how many non-filtered entities score >= true_score
      filtered_rank = 1
      for i in range(len(sorted_extgene_ids)):
        ext_id = sorted_extgene_ids[i].item()
        if ext_id == test_tail:
          break
        if ext_id not in filter_set and sorted_scores[i] >= true_score:
          filtered_rank += 1
      filtered_ranks_of_true.append(filtered_rank)

    ranks_tensor = torch.tensor(filtered_ranks_of_true, dtype=torch.float)
    metrics['mrr'] = (1.0 / ranks_tensor).mean().item() if len(filtered_ranks_of_true) > 0 else 0.0
    for k in [1, 3, 5, 10, 20, 50, 100]:
      hits = sum(1 for r in filtered_ranks_of_true if r <= k)
      metrics[f'hits@{k}'] = hits / num_test_targets
    metrics['median_rank'] = float(np.median(filtered_ranks_of_true)) if filtered_ranks_of_true else float('inf')
    metrics['mean_rank'] = float(np.mean(filtered_ranks_of_true)) if filtered_ranks_of_true else float('inf')
    metrics['test_target_ranks'] = filtered_ranks_of_true
  else:
    metrics['mrr'] = None
    for k in [1, 3, 5, 10, 20, 50, 100]:
      metrics[f'hits@{k}'] = None
    metrics['median_rank'] = None
    metrics['mean_rank'] = None
    metrics['test_target_ranks'] = []

  return ranking, metrics


def evaluate_compound_legacy(
    compound_name, compound_id, model, embeddings,
    all_extgene_ids, all_extgene_names, rel_id,
    test_target_tails, topk, id2ent
):
  """
  Rank a compound against all ExtGenes and evaluate against test set.

  Args:
    compound_name: String name of the compound.
    compound_id: Integer ID.
    model: Trained model.
    embeddings: Precomputed embeddings.
    all_extgene_ids: Tensor of all ExtGene IDs.
    all_extgene_names: List of all ExtGene string names (aligned with all_extgene_ids).
    rel_id: TARGET relation ID.
    test_target_tails: Set of ExtGene IDs that are true targets in the test set for this compound.
    topk: Number of top predictions to report.
    id2ent: ID to entity name mapping.

  Returns:
    Dict with ranking results and evaluation metrics.
  """
  scores = rank_compound_against_all_extgenes(
    model, embeddings, compound_id, all_extgene_ids, rel_id
  )

  # Sort by descending score
  sorted_indices = torch.argsort(scores, descending=True)
  sorted_scores = scores[sorted_indices]
  sorted_extgene_ids = all_extgene_ids[sorted_indices]

  # Build full ranking
  ranking = []
  for i in range(len(sorted_indices)):
    ext_id = sorted_extgene_ids[i].item()
    ranking.append({
      'rank': i + 1,
      'head': compound_name,
      'tail': id2ent.get(ext_id, f"unknown_{ext_id}"),
      'tail_id': ext_id,
      'confidence': sorted_scores[i].item(),
      'is_test_target': ext_id in test_target_tails,
      'relation': 'TARGET'
    })

  # Compute retrieval metrics against test targets
  num_test_targets = len(test_target_tails)
  metrics = {
    'compound': compound_name,
    'num_extgenes_scored': len(all_extgene_ids),
    'num_test_targets': num_test_targets,
  }

  if num_test_targets > 0:
    # Find ranks of true test targets
    ranks_of_true = []
    for i, ext_id in enumerate(sorted_extgene_ids):
      if ext_id.item() in test_target_tails:
        ranks_of_true.append(i + 1)  # 1-indexed

    ranks_tensor = torch.tensor(ranks_of_true, dtype=torch.float)
    metrics['mrr'] = (1.0 / ranks_tensor).mean().item() if len(ranks_of_true) > 0 else 0.0
    for k in [1, 3, 5, 10, 20, 50, 100]:
      hits = sum(1 for r in ranks_of_true if r <= k)
      metrics[f'hits@{k}'] = hits / num_test_targets
    metrics['median_rank'] = float(np.median(ranks_of_true)) if ranks_of_true else float('inf')
    metrics['mean_rank'] = float(np.mean(ranks_of_true)) if ranks_of_true else float('inf')
    metrics['test_target_ranks'] = ranks_of_true
  else:
    metrics['mrr'] = None
    for k in [1, 3, 5, 10, 20, 50, 100]:
      metrics[f'hits@{k}'] = None
    metrics['median_rank'] = None
    metrics['mean_rank'] = None
    metrics['test_target_ranks'] = []

  return ranking, metrics


def main(model_folder, dataset_tsv, task, compound_query, topk,
         validation_size, test_size, quiet, undersample_rate, batch_size,
         target_type='ExtGene', candidate_pool='all'):

  set_seed(BASE_SEED)

  # Resolve model folder
  model_path, model_name, train_params = resolve_model_folder(model_folder)
  print(f'[i] Model file: {model_path}')
  print(f'[i] Model type: {model_name}')
  print(f'[i] Training params: {json.dumps(train_params, indent=2)}')

  # Rebuild EXACTLY the data the selected model was trained on (values saved in *_params.json):
  #  - split / undersampling seed: --split_seed if the run used a fixed split (protocol v2),
  #    otherwise the seed of the selected run (legacy: BASE_SEED + run index);
  #  - undersample rate, validation/test sizes and hyperparameter config of training.
  # Explicit CLI values (not None) still override.
  selected_run = int(train_params.get('_selected_run', 0))
  split_seed = train_params.get('split_seed')
  data_seed = int(split_seed) if split_seed is not None else BASE_SEED + selected_run
  if undersample_rate is None:
    undersample_rate = float(train_params.get('undersample_rate', 0.5))
  if validation_size is None:
    validation_size = float(train_params.get('validation_size', 0.1))
  if test_size is None:
    test_size = float(train_params.get('test_size', 0.2))
  config_name = train_params.get('config', CONFIG_NAME)
  print(f'[i] Rebuilding training data: data_seed={data_seed} (run {selected_run}), '
        f'undersample_rate={undersample_rate}, val={validation_size}, test={test_size}, config={config_name}')

  # Load dataset
  print('[i] Loading dataset...', end='', flush=True)
  start_time = time.time()
  drug_code = compound_query.split(":")[-1] if compound_query.lower() != 'all' else 'all_compounds'

  (in_channels_dict, num_nodes_per_type, num_entities, num_relations,
   train_triplets, train_index, flattened_features_per_type,
   val_triplets, test_triplets, edge_index, ent2id, relation2id,
   all_nodes_per_type) = get_dataset_for_drug_eval(
    dataset_tsv, task, validation_size, test_size, quiet, data_seed, undersample_rate
  )

  print(f' ok ({time.time() - start_time:.2f}s)')

  # Build reverse mapping
  id2ent = {v: k for k, v in ent2id.items()}

  # Collect ALL target entity IDs from the full graph (target_type is configurable:
  # ExtGene for bacterial PathogenKG, Protein for PKT Task A, Disease for PKT Task B).
  all_extgene_names = sorted(all_nodes_per_type.get(target_type, []))
  all_extgene_ids = torch.tensor(
    [ent2id[name] for name in all_extgene_names if name in ent2id],
    dtype=torch.long, device=device
  )
  # Align names with IDs (filter out any that weren't in ent2id)
  all_extgene_names = [name for name in all_extgene_names if name in ent2id]

  # Candidate pool. 'all' = every node of target_type (PathogenKG case study). 'relation' = only the
  # nodes that occur as tails of the target relation (train+val+test) — the same pool used by the
  # training/HPO filtered evaluation and by the negative sampler. Nodes outside that pool are never
  # negatives during training, so their scores are not calibrated against true targets.
  relation_pool = set()
  for trip in (train_triplets, val_triplets, test_triplets):
    if trip.numel() > 0:
      relation_pool.update(int(t) for t in trip[:, 2].tolist())
  if candidate_pool == 'relation':
    keep = [i for i, tid in enumerate(all_extgene_ids.tolist()) if tid in relation_pool]
    all_extgene_ids = all_extgene_ids[torch.tensor(keep, dtype=torch.long, device=all_extgene_ids.device)]
    all_extgene_names = [all_extgene_names[i] for i in keep]
  print(f'[i] Candidate pool: {candidate_pool} ({len(relation_pool)} {target_type} nodes occur in the target relation)')
  print(f'[i] Total {target_type} targets to score against: {len(all_extgene_ids)}')

  # Collect all compound names
  all_compound_names = sorted(all_nodes_per_type.get('Compound', []))
  print(f'[i] Total Compounds in graph: {len(all_compound_names)}')

  # Determine the target relation ID: 'TARGET' (PathogenKG) or the task relation (PKT: DTI / TREATS)
  wanted = ['TARGET'] + [t.strip() for t in task.split(',') if t.strip()]
  relation_name = None
  for w in wanted:
    for rn in relation2id:
      if rn.lower().replace(' ', '_') == w.lower().replace(' ', '_'):
        relation_name = rn
        break
    if relation_name:
      break
  if relation_name is None:
    raise KeyError(f"Target relation {wanted} not found in relation2id. Available: {list(relation2id.keys())}")
  rel_id = relation2id[relation_name]
  print(f'[i] TARGET relation ID: {rel_id}')

  # Build ground truth per compound: {compound_id -> set of ExtGene_ids}
  # We need ALL target triples (train+val+test) for the pool of true positives,
  # and test triples specifically for evaluation.
  all_targets_per_compound = {}
  for triplets in [train_triplets, val_triplets, test_triplets]:
    triplets_np = triplets.numpy() if triplets.numel() > 0 else np.empty((0, 3))
    for h, r, t in triplets_np:
      all_targets_per_compound.setdefault(int(h), set()).add(int(t))

  test_targets_per_compound = {}
  test_triplets_np = test_triplets.numpy() if test_triplets.numel() > 0 else np.empty((0, 3))
  for h, r, t in test_triplets_np:
    test_targets_per_compound.setdefault(int(h), set()).add(int(t))

  # train+val targets (for filtered evaluation)
  train_targets_per_compound = {}
  for triplets in [train_triplets, val_triplets]:
    triplets_np = triplets.numpy() if triplets.numel() > 0 else np.empty((0, 3))
    for h, r, t in triplets_np:
      train_targets_per_compound.setdefault(int(h), set()).add(int(t))

  # --- Diagnostic: show split sizes ---
  print(f'[i] TARGET triple split: train={len(train_triplets)}, val={len(val_triplets)}, test={len(test_triplets)}')
  print(f'[i] Compounds with test targets: {len(test_targets_per_compound)} / {len(all_compound_names)}')
  if test_targets_per_compound:
    total_test_tails = sum(len(v) for v in test_targets_per_compound.values())
    print(f'[i] Total test target edges: {total_test_tails}')
    # Show a few compounds with test targets
    sample = list(test_targets_per_compound.items())[:5]
    for cid, tails in sample:
      print(f'    compound_id={cid} ({id2ent.get(cid, "?")}) -> {len(tails)} test target(s)')
  else:
    print(f'[WARNING] No compounds have test targets! Check split sizes and dataset.')

  # Create model and load weights
  model, lr, regularization, grad_norm = get_model(
    model_name, task, in_channels_dict, num_nodes_per_type, num_entities, num_relations, config=config_name
  )
  model.load_state_dict(torch.load(model_path, map_location=device))
  model = model.to(device)
  model.eval()

  # Move data to device
  train_index = train_index.to(device)
  flattened_features_per_type = {
    node_type: (features.to(device) if features is not None else None)
    for node_type, features in flattened_features_per_type.items()
  }

  # Compute change_points for RGAT
  change_points = None
  if model_name == 'rgat':
    rel_ids_sorted = train_index[:, 1]
    change_points = torch.cat([
      torch.tensor([0], device=device),
      (rel_ids_sorted[1:] != rel_ids_sorted[:-1]).nonzero(as_tuple=False).view(-1) + 1,
      torch.tensor([rel_ids_sorted.size(0)], device=device)
    ])

  # Compute embeddings once
  print('[i] Computing embeddings...')
  with torch.no_grad():
    if change_points is not None:
      embeddings = model(flattened_features_per_type, train_index, change_points)
    else:
      embeddings = model(flattened_features_per_type, train_index)

  # Determine which compounds to evaluate
  if compound_query.lower() == 'all':
    compounds_to_eval = all_compound_names
  else:
    # Match by exact name or partial match
    compounds_to_eval = []
    for name in all_compound_names:
      if compound_query in name or name == compound_query:
        compounds_to_eval.append(name)
    if not compounds_to_eval:
      print(f"[WARNING] Compound '{compound_query}' not found in dataset.")
      print(f"  Available compounds (first 10): {all_compound_names[:10]}")
      return

  print(f'[i] Evaluating {len(compounds_to_eval)} compound(s)...\n')

  all_rankings = {}
  all_metrics = []

  for compound_name in compounds_to_eval:
    if compound_name not in ent2id:
      print(f"[WARNING] Compound '{compound_name}' not in ent2id, skipping.")
      continue

    compound_id = ent2id[compound_name]
    test_target_tails = test_targets_per_compound.get(compound_id, set())
    train_val_tails = train_targets_per_compound.get(compound_id, set())

    with torch.no_grad():
      ranking, metrics = evaluate_compound(
        compound_name, compound_id, model, embeddings,
        all_extgene_ids, all_extgene_names, rel_id,
        test_target_tails, topk, id2ent,
        known_positive_tails=train_val_tails
      )

    metrics['num_train_val_targets'] = len(train_val_tails)
    # Share of the top-k predictions that are OUTSIDE the target-relation pool (nodes that never
    # occur in a train/val/test target edge, hence never used as training negatives either).
    top_ids = [p['tail_id'] for p in ranking[:topk]]
    metrics[f'top{topk}_outside_relation_pool'] = (
      sum(1 for t in top_ids if t not in relation_pool) / len(top_ids) if top_ids else None)
    all_rankings[compound_name] = ranking
    all_metrics.append(metrics)

    # Print per-compound summary
    print(f"  {compound_name}")
    print(f"    Train/Val targets: {len(train_val_tails)}, Test targets: {len(test_target_tails)}")
    if metrics['mrr'] is not None:
      print(f"    MRR: {metrics['mrr']:.4f}, Median rank: {metrics['median_rank']:.0f}, Mean rank: {metrics['mean_rank']:.0f}")
      print(f"    Hits@1: {metrics['hits@1']:.4f}, Hits@10: {metrics['hits@10']:.4f}, Hits@50: {metrics['hits@50']:.4f}, Hits@100: {metrics['hits@100']:.4f}")
      if metrics['test_target_ranks']:
        print(f"    Test target ranks: {metrics['test_target_ranks']}")
    else:
      print(f"    No test targets for this compound (ranking only).")

    # Show top-k predictions
    top_predictions = ranking[:topk]
    print(f"    Top {min(topk, len(top_predictions))} predictions:")
    for pred in top_predictions:
      marker = " *TEST*" if pred['is_test_target'] else ""
      train_marker = " [train/val]" if pred['tail_id'] in train_val_tails else ""
      print(f"      #{pred['rank']:5d}  {pred['tail'][:40]:40s}  conf: {pred['confidence']:.4f}{marker}{train_marker}")
    print()

  # Aggregate metrics across all compounds with test targets
  compounds_with_test = [m for m in all_metrics if m['mrr'] is not None]
  if compounds_with_test:
    print(f"\n{'='*70}")
    print(f"AGGREGATE METRICS ({len(compounds_with_test)} compounds with test targets)")
    print(f"{'='*70}")

    for key in ['mrr', 'hits@1', 'hits@3', 'hits@5', 'hits@10', 'hits@20', 'hits@50', 'hits@100', 'median_rank', 'mean_rank',
                f'top{topk}_outside_relation_pool']:
      values = [m[key] for m in compounds_with_test if m[key] is not None]
      if values:
        print(f"  {key:15s}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

    # Micro-average: pool all test target ranks together
    all_ranks = []
    total_test_targets = 0
    for m in compounds_with_test:
      all_ranks.extend(m['test_target_ranks'])
      total_test_targets += m['num_test_targets']

    if all_ranks:
      all_ranks_t = torch.tensor(all_ranks, dtype=torch.float)
      print(f"\n  Micro-averaged (over {total_test_targets} test targets):")
      print(f"    MRR:         {(1.0 / all_ranks_t).mean().item():.4f}")
      for k in [1, 3, 5, 10, 20, 50, 100]:
        print(f"    Hits@{k:<3d}:    {(all_ranks_t <= k).float().mean().item():.4f}")
      print(f"    Median rank: {np.median(all_ranks):.0f}")
      print(f"    Mean rank:   {np.mean(all_ranks):.0f}")

    print(f"{'='*70}\n")

  # Save results
  output_dir = model_folder + "/drug_eval_results"
  os.makedirs(output_dir, exist_ok=True)
  timestamp = time.strftime('%Y%m%d_%H%M%S')

  # Save per-compound rankings (top predictions only, to keep file size reasonable)
  rankings_to_save = {}
  for compound_name, ranking in all_rankings.items():
    rankings_to_save[compound_name] = ranking[:max(topk, 1000)]

  ranking_path = os.path.join(output_dir, f"pubchem-{drug_code}_eval_rankings_{timestamp}.json")
  with open(ranking_path, 'w') as f:
    json.dump(rankings_to_save, f, indent=2, default=str)
  print(f"[i] Rankings saved to: {ranking_path}")

  # Save metrics
  metrics_to_save = []
  for m in all_metrics:
    m_copy = {k: v for k, v in m.items() if k != 'test_target_ranks'}
    m_copy['test_target_ranks'] = m.get('test_target_ranks', [])
    metrics_to_save.append(m_copy)

  metrics_path = os.path.join(output_dir, f"pubchem-{drug_code}_eval_metrics_{timestamp}.json")
  with open(metrics_path, 'w') as f:
    json.dump(metrics_to_save, f, indent=2, default=str)
  print(f"[i] Metrics saved to: {metrics_path}")

  # Save summary CSV
  summary_rows = []
  for m in all_metrics:
    row = {
      'compound': m['compound'],
      'num_extgenes_scored': m['num_extgenes_scored'],
      'num_train_val_targets': m.get('num_train_val_targets', 0),
      'num_test_targets': m['num_test_targets'],
      'mrr': m['mrr'],
      'median_rank': m['median_rank'],
      'mean_rank': m['mean_rank'],
    }
    for k in [1, 3, 5, 10, 20, 50, 100]:
      row[f'hits@{k}'] = m.get(f'hits@{k}')
    summary_rows.append(row)

  summary_df = pd.DataFrame(summary_rows)

  
  summary_path = os.path.join(output_dir, f"pubchem-{drug_code}_eval_summary_{timestamp}.csv")
  summary_df.to_csv(summary_path, index=False)
  print(f"[i] Summary CSV saved to: {summary_path}")

  return all_metrics, all_rankings


if __name__ == '__main__':
  print(f'[i] Drug Evaluation Script')
  print(f'[i] Running on {device}')

  parser = argparse.ArgumentParser(
    description='Evaluate a trained model by ranking a compound against ALL ExtGene targets.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Evaluate all compounds in the dataset
  python drug_eval.py --model_folder models/target_PathogenKG_n34_core.tsv_20260225_184723

  # Evaluate a specific compound
  python drug_eval.py --model_folder models/target_PathogenKG_n34_core.tsv_20260225_184723 --compound "Compound::Pubchem:19"

  # Evaluate all compounds with top-50 predictions
  python drug_eval.py --model_folder models/target_PathogenKG_n34_core.tsv_20260225_184723 --compound all --topk 50

  # Use a different dataset
  python drug_eval.py --model_folder models/myfolder --tsv dataset/PathogenKG_n34_core.tsv.zip --compound all
    """
  )

  parser.add_argument('--model_folder', type=str, required=True,
                      help='Path to the model folder (containing .pt and _params.json files)')
  parser.add_argument('--compound', type=str, default='all',
                      help='Compound to evaluate. Use "all" for all compounds, or a specific compound ID (e.g., "Compound::Pubchem:19")')
  parser.add_argument('--tsv', type=str, default=DEFAULT_TRAIN_TSV,
                      help=f'Path to the dataset TSV (default: {DEFAULT_TRAIN_TSV})')
  parser.add_argument('--task', type=str, default='TARGET',
                      help='Task / interaction type (default: TARGET)')
  parser.add_argument('--target_type', type=str, default='ExtGene',
                      help='Node type to rank each compound against (default: ExtGene). '
                           'For PKT subgraphs use Protein (Task A / DTI) or Disease (Task B / TREATS).')
  parser.add_argument('--topk', type=int, default=20,
                      help='Number of top predictions to display per compound (default: 20)')
  parser.add_argument('--validation_size', type=float, default=None,
                      help='Validation split size (default: value used in training, from *_params.json)')
  parser.add_argument('--test_size', type=float, default=None,
                      help='Test split size (default: value used in training, from *_params.json)')
  parser.add_argument('--undersample_rate', type=float, default=None,
                      help='Fraction of non-target triplets to keep (default: value used in training)')
  parser.add_argument('--candidate_pool', type=str, default='all', choices=['all', 'relation'],
                      help="'all' = rank every node of --target_type (default, PathogenKG case study); "
                           "'relation' = only nodes occurring in the target relation, i.e. the pool of the "
                           "training evaluation and of the negative sampler.")
  parser.add_argument('--batch_size', type=int, default=4096,
                      help='Batch size for scoring triplets (default: 4096)')
  parser.add_argument('--quiet', action='store_true',
                      help='Suppress verbose output')

  args = parser.parse_args()

  dataset_tsv = _resolve_dataset_path(args.tsv)

  print(f'[i] Dataset: {os.path.basename(dataset_tsv)}')
  print(f'[i] Model folder: {args.model_folder}')
  print(f'[i] Compound query: {args.compound}')
  print(f'[i] Top-k: {args.topk}')
  print()

  main(
    model_folder=args.model_folder,
    dataset_tsv=dataset_tsv,
    task=args.task,
    compound_query=args.compound,
    topk=args.topk,
    validation_size=args.validation_size,
    test_size=args.test_size,
    quiet=args.quiet,
    undersample_rate=args.undersample_rate,
    batch_size=args.batch_size,
    target_type=args.target_type,
    candidate_pool=args.candidate_pool,
  )
