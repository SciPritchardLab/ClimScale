#!/bin/bash
#SBATCH --job-name="get_offline_error_standard"
#SBATCH --output="get_offline_error.out"
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH -t 24:00:00

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate tf2
python get_offline_error.py
