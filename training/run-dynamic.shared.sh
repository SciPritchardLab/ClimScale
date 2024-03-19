#!/bin/sh
echo "--- run-dynamic.sh ---"
echo SLURM_LOCALID $SLURM_LOCALID
echo SLURMD_NODENAME $SLURMD_NODENAME
#echo $LD_LIBRARY_PATH

# Just hardwire in .py file now. this way saves a munite.
# Everynode needs to copy data locally and set symlink once
# if [ "$SLURM_LOCALID" == "4" ] 
# then
# 	echo "Data copied"
#         mkdir /expanse/lustre/scratch/sungduk/temp_project/job_$SLURM_JOB_ID
# 	bash copy-data.sh
# else
# 	sleep 60 #Wait for other process to copy data
# fi

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate wandbenv
python tuning_script.py > logs/wandb-$SLURM_JOBID-$SLURMD_NODENAME-$SLURM_LOCALID.log 2>&1

