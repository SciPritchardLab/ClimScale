#!/bin/bash
#SBATCH --job-name="RMSE_strato_specific"
#SBATCH --output="logs_strato.out"
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:1
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH -t 6:00:00

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate tf2
python test_set_evaluation_strato.py
