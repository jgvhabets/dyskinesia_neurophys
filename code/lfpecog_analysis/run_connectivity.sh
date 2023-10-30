#!/bin/sh
#SBATCH --ntasks=10
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=100G
#SBATCH --partition=medium
#SBATCH -o logs/slurm/slurm-%j-%a.out
#SBATCH -e logs/errors/error-%j-%a.out
#SBATCH -t 2-00:00:00

module load python
python get_connectivity_hpc.py $SLURM_ARRAY_TASK_ID $1 $2
