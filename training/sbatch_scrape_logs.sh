#!/bin/bash
#SBATCH --job-name="scrape_logs"
#SBATCH --output="scrape_logs.out"
#SBATCH --partition=RM-shared
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH -t 15:00

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate preprocessing
python scrape_logs.py
