#!/bin/bash
#SBATCH -J minirun 
#SBATCH -o sbatchminirun.o%j
#SBATCH -e sbatchminirun.e%j
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 40:00
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH --mail-type=ALL
#SBATCH -A atm200007p

module load intelmpi/20.4-intel20.4
export LD_LIBRARY_PATH=/ocean/projects/atm200007p/shared/netcdf/lib:$LD_LIBRARY_PATH
export I_MPI_COMPATIBILITY=4

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate preprocessing

echo "$runname MODEL_BEGINS (`date`)"
mpirun ./cam < atm_in > logfile
echo "$runname MODEL_END (`date`)"
python make_animations.py
