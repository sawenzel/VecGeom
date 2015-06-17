#!/bin/bash -x

export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8


THIS=$(dirname ${BASH_SOURCE[0]})

# first arguments is the source directory
if [ $# -ge 5 ]; then
  LABEL=$1 ; shift
  COMPILER=$1 ; shift
  BUILDTYPE=$1 ; shift
  EXTERNALS=$1 ; shift
  BACKEND=$1 ; shift
else
  echo "$0: expecting 4 arguments [LABEL]  [COMPILER] [BUILDTYPE] [EXTERNALS]"
  return
fi

if [ $LABEL == slc6 ] || [ $LABEL == cc7 ]
then
  export PATH=/afs/cern.ch/sw/lcg/contrib/CMake/3.0.0/Linux-i386/bin:${PATH}
else
  export EXTERNALDIR=$HOME/ROOT-externals/
fi

if [[ $COMPILER == *gcc* ]]
then
  gcc47version=4.7
  gcc48version=4.8
  gcc49version=4.9
  COMPILERversion=${COMPILER}version

  ARCH=$(uname -m)
  . /afs/cern.ch/sw/lcg/contrib/gcc/${!COMPILERversion}/${ARCH}-${LABEL}/setup.sh
  #export FC=gfortran
  #export CXX=`which g++`
  #export CC=`which gcc`

  export CMAKE_SOURCE_DIR=$WORKSPACE/VecGeom
  export CMAKE_BINARY_DIR=$WORKSPACE/VecGeom/builds
  export CMAKE_BUILD_TYPE=$BUILDTYPE

  export CMAKE_INSTALL_PREFIX=$WORKSPACE/VecGeom/installation
  export BACKEND=$BACKEND
  export CTEST_BUILD_OPTIONS="-DROOT=ON -DCTEST=ON -DBENCHMARK=ON ${ExtraCMakeOptions}"
#  export BACKEND=Vc
#  export CTEST_BUILD_OPTIONS="-DROOT=ON -DVc=ON -DCTEST=ON -DBENCHMARK=ON -DUSOLIDS=OFF ${ExtraCMakeOptions}"

fi

echo ${THIS}/setup.py -o ${LABEL} -c ${COMPILER} -b ${BUILDTYPE} -v ${EXTERNALS}
eval `${THIS}/setup.py -o ${LABEL} -c ${COMPILER} -b ${BUILDTYPE} -v ${EXTERNALS}`

##################### Eclair settings ##################
set -e

export PROJECT_ROOT=${WORKSPACE}
export ANALYSIS_DIR=${WORKSPACE}/VecGeom/jenkins
export OUTPUT_DIR=${WORKSPACE}/VecGeom-eclair
mkdir -p ${OUTPUT_DIR}
export PB_OUTPUT=${OUTPUT_DIR}/REPORT.@FRAME@.pb
export ECLAIR_DIAGNOSTICS_OUTPUT=${OUTPUT_DIR}/DIAGNOSTICS.txt
export ECL_CONFIG_FILE=${ANALYSIS_DIR}/VecGeom.ecl
rm -f ${ECLAIR_DIAGNOSTICS_OUTPUT}
rm -f ${OUTPUT_DIR}/REPORT.*.pb

export CC=/usr/bin/cc
export CXX=/usr/bin/c++
export AS=/usr/bin/as
export LD=/usr/bin/ld
export AR=/usr/bin/ar

eclair_env +begin +end -eval-file=${ECL_CONFIG_FILE} -- 'cmake ../ $CTEST_BUILD_OPTION && make -j 24'

cd ${OUTPUT_DIR}
rm -f REPORTS.db
eclair_report -create-db=REPORTS.db *.pb -load
rm -rf eclair_output
eclair_report -db=REPORTS.db -output=eclair_output/@TAG@.etr -reports
