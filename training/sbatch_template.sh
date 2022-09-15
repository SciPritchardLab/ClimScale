#!/bin/bash
#SBATCH --job-name="JOB_NAME_HERE"
#SBATCH --output="logs/srun-kerastuner-%j.%N.out"
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:4
#SBATCH --ntasks=5
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH -t CLOCK_TIME_HERE

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf
srun --mpi=pmi2 --wait=0 bash run-dynamic.shared.sh
