#!/bin/bash
# 1. make sure that NCO is installed in your conda env
# 2. make sure that you are in the right dir.

# Usage
# parallel bash trim_h1.sh ::: 0000 ::: `seq -w 1 12`

yr=$1
mon=$2
fin=(`ls *.cam2.h1.${yr}-${mon}-*-*.nc`)

#fout
fout=(${fin[0]//./ })
fout=${fout[0]}.${fout[1]}.${fout[2]}.${yr}-${mon}.nc
#fout=AndKua_aqua_relativeModel0011.cam2.h1.${yr}-${mon}.nc

#echo ${fin[@]}
echo "< Generating $fout >"

# trim / concat
ncrcat -d lon,,,16 ${fin[@]} ${fout}

# delete original files

rm ${fin[@]}                      # remove h1 files
rm `ls *.cam2.h0.*` # remove h0 files
rm `ls *.cam2.r.*`                # remove restart files
rm logfile
