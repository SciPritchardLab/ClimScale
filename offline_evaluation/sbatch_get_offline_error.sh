#!/bin/bash
#SBATCH --job-name="get_offline_error"
#SBATCH --output="get_offline_error.out"
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH -t 24:00:00

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate tf2
python get_offline_error.py
