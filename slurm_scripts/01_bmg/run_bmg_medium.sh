#!/usr/bin/bash
# Name the job
#SBATCH --job-name=bmg
#SBATCH --time=4-00:00:00
#SBATCH --partition=Kathleen
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=long
#SBATCH --mem=34G
#SBATCH --output=./logs/bmg_medium/%A_%a.out
#SBATCH --array 0-35
#SBATCH --exclusive=user

# Load modules and environment


methods=("so-smac")
num_methods=${#methods[@]}

task_index=$(( SLURM_ARRAY_TASK_ID / num_methods ))
method_index=$(( SLURM_ARRAY_TASK_ID % num_methods ))
method=${methods[$method_index]}

srun -c 2 --cpu-bind=cores python -u -m experiments.run_bmg --amlb_task_id_zero_based "${task_index}" --bmg_method "${method}" --n_cores_target_fun 2 --time_search_h 8