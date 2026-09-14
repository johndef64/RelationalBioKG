#%%
import gc
import os
import copy
import wandb
import torch
import numpy as np
from src.utils import set_seed
import torch.nn.functional as F
from train_and_eval import get_dataset, train, test, negative_sampling, negative_sampling_filtered, \
	mixed_metric, SupervisionSampler

from src.hetero_rgcn import HeterogeneousRGCN as rgcn
from src.hetero_rgat import HeterogeneousRGAT as rgat
from src.hetero_compgcn import HeterogeneousCompGCN as compgcn
from src.kge_distmult import DistMultKGE

# WandB configuration — logs to the RelationalPKT project (NOT pathogenkg).
# Overridable via env vars (WANDB_ENTITY / WANDB_PROJECT).
# ---- anonymized version (restore before publishing) ----
# ENTITY = os.environ.get("WANDB_ENTITY", "YOUR_WANDB_ENTITY")
# ---- real coordinates (in clear for now, anonymize later) ----
ENTITY = os.environ.get("WANDB_ENTITY", "giovannimaria-defilippis-university-of-naples-federico-ii")
PROJECT_NAME = os.environ.get("WANDB_PROJECT", "RelationalPKT")
# Dataset / task for the sweep — override for PKT via env (see experiments/e2_hpo_sweep.sh)
HPO_TSV   = os.environ.get("PKT_TSV",  "dataset/PathogenKG_n31_core.tsv.zip")
HPO_TASK  = os.environ.get("PKT_TASK", "TARGET")
HPO_EPOCHS   = int(os.environ.get("PKT_HPO_EPOCHS", "200"))
HPO_PATIENCE = int(os.environ.get("PKT_HPO_PATIENCE", "50"))
HPO_RUNS     = int(os.environ.get("PKT_HPO_RUNS", "100"))
# Sweep-id persistence for crash-resume: created sweep ids are saved here; set
# PKT_HPO_RESUME=1 to re-attach an agent to the saved sweep instead of creating a new one.
HPO_RESUME   = os.environ.get("PKT_HPO_RESUME", "0") == "1"
SWEEP_DIR    = os.path.join("experiments", "hpo_sweeps")

# ---- Training protocol of the sweep (see docs/piano_consolidamento_v2.md) ----
#   v1 = legacy PathogenKG protocol used by the first PKT HPO: oversample x5, undersample 0.5,
#        early stopping on val M but test on the LAST model, opn in {sub, corr(=mult)}.
#   v2 = consolidated protocol: oversample 1 + train negatives as a hyperparameter, full
#        context graph, disjoint supervision edges, best-checkpoint (val M) restored before test,
#        opn in {sub, mult, ccorr}, no num_bases for CompGCN (it never used it).
HPO_PROTOCOL = os.environ.get("PKT_HPO_PROTOCOL", "v2")
if HPO_PROTOCOL not in ("v1", "v2"):
	raise ValueError("PKT_HPO_PROTOCOL must be v1 or v2")
_V2 = HPO_PROTOCOL == "v2"
HPO_OVERSAMPLE = int(os.environ.get("PKT_HPO_OVERSAMPLE", "1" if _V2 else "5"))
HPO_UNDERSAMPLE = float(os.environ.get("PKT_HPO_UNDERSAMPLE", "1.0" if _V2 else "0.5"))
HPO_DISJOINT = float(os.environ.get("PKT_HPO_DISJOINT", "0.3" if _V2 else "0.0"))
#%%
# Hyperparameter search space

# Attention this configuration requires 28-32 GB of GPU memory for CompGCN with 2 layers and 128 hidden units, so it may need to be adjusted based on available resources. The "narrowed" config is a more conservative search space that should be feasible on smaller GPUs while still exploring a range of values around the best found in the initial sweep.

# Discretized version: reduces search space for faster Bayesian convergence.
# Values chosen for biomedical KG drug repurposing (~50K nodes, ~2M edges).
SWEEP_CONFIG_DISCRETE = {
    'method': 'bayes',
    'metric': {
        'name': 'val_mixed_metric',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-4, 3e-4, 5e-4, 1e-3, 3e-3], #, 5e-3]
        },
        'regularization': {
            'values': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        },
        'grad_norm': {
            'values': [0.5, 1.0, 1.5, 2.0]
        },
        'dropout': {
            'values': [0.2, 0.3, 0.4, 0.5]
        },
        'weight_decay': {
            'values': [1e-3, 5e-3, 1e-2, 5e-2]  #  2e-2,
        },
        'scheduler_gamma': {
            'values': [0.99, 0.995, 0.997, 0.999]  # 0.993,
        },
        'conv_layer_num': {
            'values': [1, 2]
        },
        'mlp_out_layer': {
            'values': [64, 128, 200]
        },
        'layer_0': {
            'values': [64, 128, 200]
        },
        'layer_1': {
            'values': [64, 128, 200]
        },
        'layer_2': {
            'values': [64, 128, 200]
        },
        'num_bases': {
            'values': [10, 15, 20]
        },
        'opn': {
            'values': ['sub', 'corr']
        },
    }
}

# Continuous range version (wider exploration, slower convergence)
SWEEP_CONFIG_RANGE = {
    'method': 'bayes',
    'metric': {
        'name': 'val_mixed_metric',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 5e-4,
            'max': 5e-3
        },
        'regularization': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2
        },
        'grad_norm': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 2.0
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.2,
            'max': 0.5
        },

		'weight_decay': {
			'distribution': 'log_uniform_values',
			'min': 1e-3,
			'max': 5e-2
		},
		'scheduler_gamma': {
			'distribution': 'uniform',
			'min': 0.99,
			'max': 0.999
		},
        'conv_layer_num': {
            'values': [1, 2]
        },
        'mlp_out_layer': {
            'values': [64, 128, 200]
        },
        'layer_0': {
            'values': [64, 128, 200]
        },
        'layer_1': {
            'values': [64, 128, 200]
        },
        'layer_2': {
            'values': [64, 128, 200]
        },
        'num_bases': {
            'values': [10, 15, 20]
        },
        'opn': {
            'values': ['sub', 'corr']
        },
    }
}

SWEEP_CONFIG_RANGE_SMALL_GRAPHS = {
	'method': 'bayes',
	'metric': {
		'name': 'val_mixed_metric',
		'goal': 'maximize'
	},
	'parameters': {
		'learning_rate': {
			'distribution': 'log_uniform_values',
			'min': 1e-4,
			'max': 1e-2
		},
		'regularization': {
			'distribution': 'log_uniform_values',
			'min': 1e-6,
			'max': 1e-2
		},
		'grad_norm': {
			'distribution': 'uniform',
			'min': 0.5,
			'max': 5.0
		},
		'dropout': {
			'distribution': 'uniform',
			'min': 0.0,
			'max': 0.7
		},
		'conv_layer_num': {
			'values': [1, 2, 3]
		},
		'mlp_out_layer': {
			'values': [8, 16, 32, 64]
		},
		'layer_0': {
			'values': [8, 16, 32, 64]
		},
		'layer_1': {
			'values': [8, 16, 32, 64]
		},
		'layer_2': {
			'values': [8, 16, 32, 64]
		},
		'num_bases': {
			'values': [10, 20, 30, 40, 50]
		},
		'opn': {  # For CompGCN only
			'values': ['mult', 'sub', 'corr']
		},

	}
}

SWEEP_CONFIG = copy.deepcopy(SWEEP_CONFIG_DISCRETE)
if _V2:
	# select on the best validation M reached (restored checkpoint), not the last logged value
	SWEEP_CONFIG['metric'] = {'name': 'best_val_mixed_metric', 'goal': 'maximize'}
	# 'corr' was an element-wise product; test it under its real name plus the true circular correlation
	SWEEP_CONFIG['parameters']['opn'] = {'values': ['sub', 'mult', 'ccorr']}
	# negatives per positive in training (replaces the x5 oversampling, which only acted as 5 negatives)
	SWEEP_CONFIG['parameters']['train_negative_rate'] = {'values': [1, 5, 10]}
	# v1 winners sat on the upper bound (3e-3): full-batch training does one step per epoch
	SWEEP_CONFIG['parameters']['learning_rate'] = {'values': [3e-4, 1e-3, 3e-3, 1e-2]}

# Embedding-only DistMult baseline (no message passing): only optimisation params + embedding size.
DISTMULT_PARAMS = {
	'learning_rate': {'values': [3e-3, 1e-2, 3e-2, 1e-1]},
	'mlp_out_layer': {'values': [64, 128, 200, 400]},
}
DISTMULT_DROP = ['conv_layer_num', 'layer_0', 'layer_1', 'layer_2', 'num_bases', 'opn', 'dropout']

# AVAILABLE_MODELS = ['rgcn', 'rgat', 'compgcn']
# AVAILABLE_MODELS = ['rgat']
# v2 also tunes the DistMult baseline (a GNN must beat a FAIRLY tuned embedding-only model)
AVAILABLE_MODELS = os.environ.get("PKT_HPO_MODELS", "rgcn compgcn distmult" if _V2 else "rgcn compgcn").split()
# AVAILABLE_MODELS = ['compgcn']
BASE_SEED = 42
USE_ALTERNATIVE_NEG_SAMPLING = True
USE_FILTERED_EVAL = True   # True = filtered (standard KGE), False = legacy
negative_sampling = negative_sampling_filtered  # Use filtered negative sampling for better evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cleanup_cuda():
	"""Clean up CUDA memory"""
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
	gc.collect()

def create_model_from_config(model_name, config, in_channels_dict, num_nodes_per_type, num_entities, num_relations):
	"""Create model with hyperparameters from WandB config"""
	if model_name == 'distmult':
		return DistMultKGE(num_entities, num_relations + 1, config.mlp_out_layer, device=device)
	# Build conv_hidden_channels dict
	conv_hidden_channels = {}
	for i in range(config.conv_layer_num):
		layer_key = f'layer_{i}'
		if layer_key in config:
			conv_hidden_channels[layer_key] = getattr(config, layer_key)
		else:
			# Default fallback
			conv_hidden_channels[layer_key] = 128
	
	if model_name == 'rgcn':
		model = rgcn(
			in_channels_dict,
			None,
			config.mlp_out_layer,
			conv_hidden_channels,
			num_nodes_per_type,
			num_entities,
			num_relations + 1,
			config.conv_layer_num,
			config.num_bases,
			activation_function=F.relu,
			device=device
		)
	elif model_name == 'rgat':
		model = rgat(
			in_channels_dict,
			None,
			config.mlp_out_layer,
			conv_hidden_channels,
			num_nodes_per_type,
			num_entities,
			num_relations + 1,
			conv_num_layers=config.conv_layer_num,
			num_bases=config.num_bases,
			activation_function=F.relu,
			device=device
		)
	elif model_name == 'compgcn':
		model = compgcn(
			in_channels_dict,
			mlp_out_emb_size=config.mlp_out_layer,
			conv_hidden_channels=conv_hidden_channels,
			num_nodes_per_type=num_nodes_per_type,
			num_entities=num_entities,
			num_relations=num_relations,
			dropout=config.dropout,
			conv_num_layers=config.conv_layer_num,
			opn=config.opn,
			use_layer_norm=config.get('use_layer_norm', True),
			device=device,
		)

	return model

def train_model(config=None):
	"""Training function called by WandB sweep (pass `config` only for local testing)."""
	# Initialize WandB run
	wandb.init(config=config)
	config = wandb.config
	
	# Get model name from config (set by sweep)
	model_name = config.model_name
	
	# Fixed training parameters (dataset/task configurable via env — see module top)
	tsv_path = HPO_TSV
	task = HPO_TASK
	validation_size = 0.1
	test_size = 0.2
	epochs = HPO_EPOCHS  # Reduced for hyperopt
	patience = HPO_PATIENCE
	evaluate_every = 5
	negative_rate = 1
	alone = False

	# sampling protocol (v1 = legacy x5 / 0.5; v2 = see module top)
	oversample_rate = HPO_OVERSAMPLE
	undersample_rate = HPO_UNDERSAMPLE
	disjoint_supervision = HPO_DISJOINT
	train_negative_rate = int(config.get('train_negative_rate', negative_rate))
	alpha = 0.25
	gamma = 3.0
	alpha_adv = 2.0

	# Set seed for reproducibility (single run per trial -> split fixed at BASE_SEED)
	seed = BASE_SEED
	set_seed(seed)
	wandb.config.update({
		'protocol': HPO_PROTOCOL, 'split_seed': seed, 'oversample_rate': oversample_rate,
		'undersample_rate': undersample_rate, 'disjoint_supervision': disjoint_supervision,
		'train_negative_rate': train_negative_rate, 'negative_rate_eval': negative_rate,
		'epochs_max': epochs, 'patience_evals': patience, 'tsv': tsv_path, 'task': task,
	}, allow_val_change=True)

	
	try:
		# Load dataset
		# error missing oversample_rate, undersample_rate in get_dataset call
		(in_channels_dict, num_nodes_per_type, num_entities, num_relations,
		 train_triplets, train_index, flattened_features_per_type, val_triplets,
		 train_val_triplets, test_triplets, train_val_test_triplets,
		 edge_index, ent2id, relation2id) = get_dataset(
			tsv_path, task, validation_size, test_size, True, seed,
			# added new {2025-12-15}
			oversample_rate=oversample_rate,
			undersample_rate=undersample_rate
		)

			# preparazione parametri per neg sampling corretto
		all_entities_arr = np.arange(num_entities)
		all_true_arr = train_val_test_triplets.cpu().numpy()
		
		# Create model with hyperparameters from config
		model = create_model_from_config(
			model_name, config, in_channels_dict, 
			num_nodes_per_type, num_entities, num_relations
		)
		
		# Move to device
		model = model.to(device)
		train_index = train_index.to(device)
		flattened_features_per_type = {
			node_type: (features.to(device) if features is not None else None) 
			for node_type, features in flattened_features_per_type.items()
		}
		
		# Handle change points for RGAT
		change_points = None
		if model_name == 'rgat':
			sorted_indices = torch.argsort(train_index[:, 1])
			train_index = train_index[sorted_indices]
			
			rel_ids_sorted = train_index[:, 1]
			change_points = torch.cat([
				torch.tensor([0], device=device),
				(rel_ids_sorted[1:] != rel_ids_sorted[:-1]).nonzero(as_tuple=False).view(-1) + 1,
				torch.tensor([rel_ids_sorted.size(0)], device=device)
			])
		
		# Optimizer + Scheduler
		optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler_gamma)

		# Training loop
		best_mixed_metric = -float("inf")
		best_state, best_epoch = None, None
		patience_trigger = 0
		sup_sampler = None
		if disjoint_supervision > 0:
			sup_sampler = SupervisionSampler(train_index, train_triplets, num_relations, disjoint_supervision,
			                                 oversample_rate, seed)

		for epoch in range(1, epochs + 1):
			# Training
			if sup_sampler is not None:
				epoch_index, epoch_positives = sup_sampler.sample(epoch)
			else:
				epoch_index, epoch_positives = train_index, train_triplets
			train_triplets_np = epoch_positives.cpu().numpy() if torch.is_tensor(epoch_positives) else epoch_positives
			if not USE_ALTERNATIVE_NEG_SAMPLING:
				training_triplets, train_labels = negative_sampling(train_triplets_np, train_negative_rate)
			else:
				training_triplets, train_labels = negative_sampling(
    train_triplets_np, all_entities_arr, train_negative_rate, all_true_arr, seed=seed + epoch)


			training_triplets, train_labels = training_triplets.to(device), train_labels.to(device)

			train_metrics = train(
				model, optimizer, config.grad_norm, config.regularization,
				flattened_features_per_type, epoch_index,
				training_triplets, train_labels,
				alpha, gamma, alpha_adv, change_points
			)

			scheduler.step()

			# Validation
			if epoch % evaluate_every == 0:
				val_triplets_np = val_triplets.cpu().numpy() if torch.is_tensor(val_triplets) else val_triplets
				if not USE_ALTERNATIVE_NEG_SAMPLING:
					validation_triplets, val_labels = negative_sampling(val_triplets_np, int(negative_rate))
				else:
					validation_triplets, val_labels = negative_sampling(
    val_triplets_np, all_entities_arr, int(negative_rate), all_true_arr, seed=seed + 1000)

					
				validation_triplets, val_labels = validation_triplets.to(device), val_labels.to(device)
				
				val_metrics = test(
					model, config.regularization,
					flattened_features_per_type, train_index,
					validation_triplets, val_labels,
					train_val_triplets,
					alpha, gamma, alpha_adv, change_points,
					use_filtered_eval=USE_FILTERED_EVAL,
                    all_target_triplets=train_val_test_triplets,
                    num_entities=num_entities,
				)

				# Calculate mixed metric
				val_mixed = mixed_metric(val_metrics)

				# Log metrics to WandB
				wandb.log({
					'epoch': epoch,
					'train_loss': train_metrics["Loss"],
					'train_auroc': train_metrics["Auroc"],
					'train_auprc': train_metrics["Auprc"],
					'val_loss': val_metrics["Loss"],
					'val_auroc': val_metrics["Auroc"],
					'val_auprc': val_metrics["Auprc"],
					'val_mrr': val_metrics["MRR"],
					'val_mixed_metric': val_mixed
				})

				# Early stopping (+ best checkpoint kept in memory for v2)
				if val_mixed > best_mixed_metric:
					best_mixed_metric = val_mixed
					best_epoch = epoch
					patience_trigger = 0
					if _V2:
						best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
				else:
					patience_trigger += 1

				if patience_trigger > patience:
					break

		if _V2 and best_state is not None:
			# test the checkpoint selected on validation M (v1 tested the last model)
			model.load_state_dict(best_state)
		wandb.log({'best_val_mixed_metric': best_mixed_metric, 'best_epoch': best_epoch})

		# Final test evaluation
		test_triplets_np = test_triplets.cpu().numpy() if torch.is_tensor(test_triplets) else test_triplets
		if not USE_ALTERNATIVE_NEG_SAMPLING:
			testing_triplets, test_labels = negative_sampling(test_triplets_np, int(negative_rate))
		else:
			testing_triplets, test_labels = negative_sampling(
    test_triplets_np, all_entities_arr, int(negative_rate), all_true_arr, seed=seed + 2000)

		testing_triplets, test_labels = testing_triplets.to(device), test_labels.to(device)
		
		test_metrics = test(
			model, config.regularization,
			flattened_features_per_type, train_index,
			testing_triplets, test_labels,
			train_val_test_triplets,
			alpha, gamma, alpha_adv, change_points,
			use_filtered_eval=USE_FILTERED_EVAL,
                    all_target_triplets=train_val_test_triplets,
                    num_entities=num_entities,
		)
		
		# Log final test metrics
		wandb.log({
			'test_auroc': test_metrics["Auroc"],
			'test_auprc': test_metrics["Auprc"],
			'test_mrr': test_metrics["MRR"],
			'test_hits@1': test_metrics["Hits@"][1],
			'test_hits@3': test_metrics["Hits@"][3],
			'test_hits@10': test_metrics["Hits@"][10],
			'final_mixed_metric': 0.2 * test_metrics["Auroc"] + 0.4 * test_metrics["Auprc"] + 0.4 * test_metrics["MRR"]
		})
		"""
		final_mixed_metric
		AUROC (20%): misura separazione classi, robusta ma meno sensibile a imbalance.
		AUPRC (40%): cruciale per link prediction (pochi edge positivi), enfatizza precision/recall su positivi.
		MRR (40%): prioritizza top ranking (essenziale per raccomandazioni link).
		Pesi enfatizzano metriche ranking-specifiche vs AUROC generica.
		"""
		
		cleanup_cuda()
		
	except torch.cuda.OutOfMemoryError as e:
		print(f"🚨 CUDA OOM Error: {e}")
		print("💡 Trying to recover by cleaning up memory...")
		cleanup_cuda()
		
		# Log the error but don't fail the run
		wandb.log({
			'error': 'CUDA_OOM',
			'error_details': str(e),
			'status': 'skipped'
		})
		
		# Skip this configuration
		return
		
	except Exception as e:
		print(f"❌ Error in training: {e}")
		wandb.log({
			'error': 'training_error',
			'error_details': str(e),
			'status': 'failed'
		})
		cleanup_cuda()
		raise e
	
	finally:
		cleanup_cuda()
		wandb.finish()

def run_hyperparameter_optimization():
	"""Run hyperparameter optimization for all models"""
	
	for model_name in AVAILABLE_MODELS:
		print(f"Starting hyperparameter optimization for {model_name}")
		
		# Create model-specific sweep config
		# sweep_config = SWEEP_CONFIG.copy()
		import copy
		sweep_config = copy.deepcopy(SWEEP_CONFIG)

		sweep_config['name'] = f'{model_name}-hyperopt'
		
		# Add model_name as a fixed parameter
		sweep_config['parameters']['model_name'] = {'value': model_name}
		
		# Filter parameters based on model
		if model_name == 'distmult':
			for k in DISTMULT_DROP:
				sweep_config['parameters'].pop(k, None)
			sweep_config['parameters'].update(copy.deepcopy(DISTMULT_PARAMS))
		elif model_name != 'compgcn':
			# Remove CompGCN-specific parameters
			if 'opn' in sweep_config['parameters']:
				del sweep_config['parameters']['opn']
			if 'dropout' in sweep_config['parameters']:
				del sweep_config['parameters']['dropout']
		elif _V2 and 'num_bases' in sweep_config['parameters']:
			# HeterogeneousCompGCN has no basis decomposition: num_bases was a dead dimension
			del sweep_config['parameters']['num_bases']
		print(f"[HPO] protocol={HPO_PROTOCOL} oversample={HPO_OVERSAMPLE} undersample={HPO_UNDERSAMPLE} "
		      f"disjoint={HPO_DISJOINT} metric={sweep_config['metric']['name']}")

		# Create sweep with model-specific project name (or re-attach to a saved one)
		model_project = f"{PROJECT_NAME}-{model_name}"
		os.makedirs(SWEEP_DIR, exist_ok=True)
		id_path = os.path.join(SWEEP_DIR, f"{model_project}.txt")

		if HPO_RESUME and os.path.exists(id_path):
			sweep_id = open(id_path).read().strip()
			print(f"[resume] re-attaching to existing sweep {sweep_id} for {model_name}")
		else:
			sweep_id = wandb.sweep(
				sweep_config,
				project=model_project,
				entity=ENTITY
			)
			with open(id_path, "w") as f:
				f.write(sweep_id)
			print(f"Created sweep {sweep_id} for {model_name}  (saved -> {id_path})")

		print(f"Project: {model_project}")
		print(f"Resume later:  PKT_HPO_RESUME=1 python tuning_hyperparameter.py")
		print(f"   (or directly: wandb agent {ENTITY}/{model_project}/{sweep_id})")

		# Run the sweep (configurable via PKT_HPO_RUNS)
		number_of_runs = HPO_RUNS
		wandb.agent(
			sweep_id,
			train_model,
			count=number_of_runs,
			project=model_project,
			entity=ENTITY
		)

		print(f"Completed hyperparameter optimization for {model_name}")

if __name__ == "__main__":
	
	print("🔬 Starting hyperparameter optimization with WandB")
	print(f"🖥️ Device: {device}")
	print(f"📊 Total planned runs: {len(AVAILABLE_MODELS) * 100}")
	print("-" * 50)
	
	run_hyperparameter_optimization()
