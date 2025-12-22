#!/usr/bin/bash
# Name the job
#SBATCH --job-name=yourPartition.%j

### Start of Slurm SBATCH definitions
#SBATCH -a 1-6
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=Test
##SBATCH --gres=gpu:1
# Ask for the maximum memory per CPU
#SBATCH --mem=18000M
##SBATCH --exclusive

# Ask for up to 10 Minutes of runtime
#SBATCH --time=00:20:00



# Declare a file where the STDOUT/STDERR outputs will be written
#SBATCH --output=test_slurm.%J

### end of Slurm SBATCH definitions
hostname
module load Python/3.10.8-GCCcore-12.2.0-bare
### your program goes here (hostname is an example, can be any program)
# `srun` runs `ntasks` instances of your programm `hostname`
source ~/meta-energy/.venv/bin/activate
srun python test_cores.py
