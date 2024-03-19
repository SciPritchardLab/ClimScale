#!/bin/bash
#SBATCH --job-name="JOB_NAME_HERE"
#SBATCH --output=""logs/srun-wandb-%j.%N.out""
#SBATCH --partition=PARTITION_HERE
#SBATCH --gpus=v100-16:NUM_GPUS_PER_NODE_HERE
#SBATCH --ntasks=NUM_GPUS_PER_NODE_HERE
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH -t CLOCK_TIME_HERE

cp -v training_data/* /dev/shm
source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate wandbenv
srun --mpi=pmi2 --wait=0 bash run-dynamic.shared.sh
