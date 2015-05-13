#!/bin/bash -x

export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8


THIS=$(dirname ${BASH_SOURCE[0]})

# first arguments is the source directory
if [ $# -ge 4 ]; then
  LABEL=$1 ; shift
  COMPILER=$1 ; shift
  BUILDTYPE=$1 ; shift
  EXTERNALS=$1 ; shift
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
  export FC=gfortran
  export CXX=`which g++`
  export CC=`which gcc`

  export CMAKE_SOURCE_DIR=$WORKSPACE/VecGeom 
  export CMAKE_BINARY_DIR=$WORKSPACE/VecGeom/builds
  export CMAKE_BUILD_TYPE=$BUILDTYPE

  export CMAKE_INSTALL_PREFIX=$WORKSPACE/VecGeom/installation
  export BACKEND=Vc 
  export CTEST_BUILD_OPTIONS="-DROOT=ON -DVc=ON -DCTEST=ON -DBENCHMARK=ON -DUSOLIDS=OFF ${ExtraCMakeOptions}"

fi

echo ${THIS}/setup.py -o ${LABEL} -c ${COMPILER} -b ${BUILDTYPE} -v ${EXTERNALS}
eval `${THIS}/setup.py -o ${LABEL} -c ${COMPILER} -b ${BUILDTYPE} -v ${EXTERNALS}`