#!/usr/bin/bash
# Name the job
#SBATCH --job-name=parallel_allotment
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=medium
#SBATCH --mem=15G
#SBATCH --output=./logs/parallel_allotment/allotment_%A_%a.out
#SBATCH --array 0-546

#Load modules and environment

python -u -m experiments.run_ensembling --exp_id $SLURM_ARRAY_TASK_ID --exp_type parallel