# Site-specific environment for the block-1 Slurm jobs.
#
# Copy this file to experiments/block1/slurm/site_env.sh (git-ignored, stays on the cluster) and put
# there the two or three lines that activate the `gnn` environment on your cluster. Every job in
# experiments/block1/slurm/ sources site_env.sh if it exists, and falls back to `conda activate gnn`
# otherwise.
#
# Example for a cluster that provides its own activation script:
#   source /path/to/cluster-env.sh
#   cluster_activate gnn
#
# Example for plain conda:
#   source "$(conda info --base)/etc/profile.d/conda.sh"
#   conda activate gnn
