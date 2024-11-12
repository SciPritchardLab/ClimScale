#! /usr/bin/csh -f
# make sure to load intelmpi before running this script
# e.g., module load intelmpi/20.4-intel20.4

module load intelmpi/20.4-intel20.4
set rpath="$HOME/repositories/spcam3.0-neural-net"
setenv camroot $rpath/models/atm/cam
setenv esmfroot $rpath/models/utils/esmf/build/linux_intel
echo $camroot
#exit(0)
cp $camroot/bld/Makefile.bridges2 $camroot/bld/Makefile
cp $rpath//models/utils/esmf/build/linux_intel/base_variables.bridges2 $rpath/models/utils/esmf/build/linux_intel/base_variables
# Note I had to install my own version of netcdf3.6.3 to be old enough to play nice with spcam3.
setenv INC_NETCDF   /ocean/projects/atm200007p/shared/netcdf/include
setenv LIB_NETCDF   /ocean/projects/atm200007p/shared/netcdf/lib

setenv MPICH_DIR $I_MPI_ROOT # on bridges-2 this is the location if intelmpi is module loaded
                             # Sungduk, Fri Mar  4 17:29:52 EST 2022
                             # e.g., module load intelmpi/20.4-intel20.4
                             # -mpi_lib for configure_mmf is modified (i.e., /release subdir is added)

# user override if desired. Expectation is you will use this scripts in a local dir, where obj created
set wrkdir       = /ocean/projects/atm200007p/jlin96/nnspreadtesting_good/nocoszrs/coupling_folder/cam_folder_debug
set blddir       = $wrkdir/obj
set rundir       = $wrkdir/run
set cfgdir       = $camroot/bld

## Ensure that run and build directories exist
mkdir -p $rundir                || echo "cannot create $rundir" && exit 1
mkdir -p $blddir                || echo "cannot create $blddir" && exit 1

## If an executable doesn't exist, build one.
if ( ! -x $blddir/cam ) then
    cd $blddir                  || echo "cd $blddir failed" && exit 1

# for SP control run activate this version:
#    $cfgdir/configure_mmf -fc mpiifort -cc cc -spmd -smp -dyn sld -res 64x128 -pcols 8 -nlev 30 -cam_exedir $rundir -mpi_inc $MPICH_DIR/intel64/include -mpi_lib $MPICH_DIR/intel64/lib/release || echo "configure failed" && exit 1
# for NN run activate this version:
    $cfgdir/configure_mmf -fflags "-DCLOUDBRAIN -DNEURALLIB -DBRAINDEBUG -DHDEBUG -DRHNN -DBETAFIX" -fc mpiifort -cc cc -spmd -smp -dyn sld -res 64x128 -pcols 8 -nlev 30 -cam_exedir $rundir -mpi_inc $MPICH_DIR/intel64/include -mpi_lib $MPICH_DIR/intel64/lib/release || echo "configure failed" && exit 1
# Use this non-SP build script when using CLOUDBRAIN to avoid stomping on state_save compiler messages:
    echo "building CAM in $blddir ..."
    rm -f Depends
    gmake -j 8 >&! Make.out #-j 8 #>&! MAKE.out      || echo "CAM build failed: see $blddir/MAKE.out" && exit 1
endif

exit 0
