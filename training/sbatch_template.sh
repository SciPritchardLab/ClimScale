#!/bin/bash
#SBATCH --job-name="JOB_NAME_HERE"
#SBATCH --output="logs/srun-kerastuner-%j.%N.out"
#SBATCH --partition=PARTITION_HERE
#SBATCH --gpus=v100-32:NUM_GPUS_PER_NODE_HERE
#SBATCH --ntasks=NTASKS_HERE
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH -t CLOCK_TIME_HERE

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate tf2
srun --mpi=pmi2 --wait=0 bash run-dynamic.shared.sh
