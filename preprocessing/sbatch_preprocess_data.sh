#!/bin/bash
#SBATCH --job-name="preprocess_data_specific"
#SBATCH --output="preprocess_data.out"
#SBATCH --partition=RM-512
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH -t 4:15:00

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate preprocessing
python preprocess_data.py
