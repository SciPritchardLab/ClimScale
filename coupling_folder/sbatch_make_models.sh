#!/bin/bash
#SBATCH -J make_models 
#SBATCH -o sbatch_make_models.o%j
#SBATCH -e sbatch_make_models.e%j
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 1:10:00
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH --mail-type=ALL
#SBATCH -A atm200007p

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate tf2

python make_models.py
