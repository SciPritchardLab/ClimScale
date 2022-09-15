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


source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf
python tuning.py > logs/keras-tuner-$SLURM_JOBID-$SLURMD_NODENAME-$SLURM_LOCALID.log 2>&1

