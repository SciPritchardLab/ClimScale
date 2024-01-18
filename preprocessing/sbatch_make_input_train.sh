#!/bin/bash
#SBATCH --job-name="make_input_train"
#SBATCH --output="make_input_train.out"
#SBATCH --partition=RM-512
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH -t 3:20:00

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate preprocessing
python make_input_train.py
