#!/usr/bin/bash
# Name the job
#SBATCH --job-name=create_preds
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --output=./logs/ens_preds_high/%A_%a.out
#SBATCH --array 0-10
#SBATCH --exclusive

# Check that exactly one argument is given
if [ "$#" -ne 1 ]; then
    echo "Usage: $(basename "$0") <seed>"
    exit 1
fi

# Read the parameter into a variable
seed=$1

# Load modules and environment


methods=("so-smac")
num_methods=${#methods[@]}

task_index=$(( SLURM_ARRAY_TASK_ID / num_methods ))
method_index=$(( SLURM_ARRAY_TASK_ID % num_methods ))
method=${methods[$method_index]}

for n in {1..8}; do
    echo "Running with $n cores..."
    srun -c $n --cpu-bind=cores python -u -m experiments.run_create_preds --amlb_task_id_zero_based "${task_index}" --bmg_method "${method}" --n_threads $n --seed $seed --n_cpu_filter 4
done
for n in {12..64..4}; do
    echo "Running with $n cores..."
    srun -c $n --cpu-bind=cores python -u -m experiments.run_create_preds --amlb_task_id_zero_based "${task_index}" --bmg_method "${method}" --n_threads $n --seed $seed --n_cpu_filter 4
done