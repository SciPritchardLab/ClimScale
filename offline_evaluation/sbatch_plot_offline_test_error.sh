#!/bin/bash
#SBATCH --job-name="plot_offline_test_error_nodropout"
#SBATCH --output="plot_offline_test_error.out"
#SBATCH --partition=RM
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH -t 18:00

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate preprocessing
python plot_offline_test_error.py
