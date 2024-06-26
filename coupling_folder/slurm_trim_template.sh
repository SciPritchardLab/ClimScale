#!/bin/bash
#SBATCH -J slurm_VARIANT_START_STOP 
#SBATCH -o myjobTRIM.o%j
#SBATCH -e myjobTRIM.e%j
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 6:00:00
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH --mail-type=ALL
#SBATCH -A atm200007p

module load intelmpi/20.4-intel20.4
export LD_LIBRARY_PATH=/ocean/projects/atm200007p/shared/netcdf/lib:$LD_LIBRARY_PATH
export I_MPI_COMPATIBILITY=4

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate preprocessing

family=$1
ind1=$2
ind2=$3

main_folder=$(pwd)

for k in `seq -f "%03g" $ind1 $ind2`
do
	# run name
	runname=${family}_model_${k}

	# create run path
	rundir=../coupled_results/${runname}
	mkdir $rundir
	echo "$rundir"

	# move files for simulations
	# a. copy cam stuff
	cp -v cam_folder/run/cam $rundir
	cp -rv baseline/* $rundir
	cp -v atm_in.template $rundir/atm_in
	# b. copy keras stuff
	kerasdir=${rundir}/keras_matrices
	cp -v norm_files/inp_div.txt ${kerasdir}/inp_div.txt
	cp -v norm_files/inp_sub.txt ${kerasdir}/inp_sub.txt
	cp -v norm_files/out_scale.txt ${kerasdir}/out_scale.txt
	cp -v txt_models/${family}_model_${k}.txt ${kerasdir}/model.txt
	# c. copy trim script (trim_h1.sh)
	# cp -v <UPDATE_src_code_path> $rundir
	cd $rundir
	# update atm_in, submit.sh
	sed -i "s/FAMILY_MODELRANK/$runname/g" atm_in

	# run
	echo "$runname MODEL_BEGINS (`date`)"
	mpirun ./cam < atm_in > logfile
	echo "$runname MODEL_END (`date`)"
	
	# trim
	echo "$runname TRIM_BEGINS (`date`)"
	parallel bash trim_h1.sh ::: 0000 ::: `seq -w 1 12`
	echo "$runname TRIM_END (`date`)"

	cd $main_folder

	rundir=../coupled_results/${runname}/minirun
	mkdir $rundir
	echo "$rundir"
	cp -v cam_folder_debug/run/cam $rundir
	cp -rv baseline/* $rundir
	cp -v atm_in_debug.template $rundir/atm_in
	cp -v sbatch_minirun.sh $rundir/sbatch_minirun.sh
	cp -v brain_debug_check.ipynb $rundir/brain_debug_check.ipynb
	cp -v make_animations.py $rundir/make_animations.py
	kerasdir=${rundir}/keras_matrices
	cp -v norm_files/inp_div.txt ${kerasdir}/inp_div.txt
	cp -v norm_files/inp_sub.txt ${kerasdir}/inp_sub.txt
	cp -v norm_files/out_scale.txt ${kerasdir}/out_scale.txt
	cp -v txt_models/${family}_model_${k}.txt ${kerasdir}/model.txt
	cp -v h5_models/${family}_model_${k}.h5 $rundir/model.h5
	cd $rundir
	sed -i "s/FAMILY_MODELRANK/$runname/g" atm_in
	# echo "$runname MODEL_BEGINS (`date`)"
	# mpirun ./cam < atm_in > logfile
	# echo "$runname MODEL_END (`date`)"
	# python make_animations.py
	cd $main_folder
done
