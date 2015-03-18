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

  export ExtraCMakeOptions="-DROOT=ON -DCTEST=ON -DBENCHMARK=ON -DUSOLIDS=ON ${ExtraCMakeOptions}"

elif [[ $COMPILER == *clang* ]]
then
  clang34version=3.4
  clang35version=3.5
  clang36version=3.6
  COMPILERversion=${COMPILER}version
  clang34gcc=48
  clang35gcc=49
  clang36gcc=49
  GCCversion=${COMPILER}gcc

  ARCH=$(uname -m)
  . /afs/cern.ch/sw/lcg/external/llvm/${!COMPILERversion}/${ARCH}-slc6/setup.sh
  export CC=`which clang`
  export CXX=`which clang++`

  export ExtraCMakeOptions="${ExtraCMakeOptions} -Dfortran=OFF -Dgcctoolchain=$(dirname $(dirname `which gcc`))"

elif [[ $COMPILER == *native* ]]
then
  export ExtraCMakeOptions="-Dfortran=OFF ${ExtraCMakeOptions}"

elif [[ $COMPILER == *icc* ]]
then
  . /afs/cern.ch/sw/IntelSoftware/linux/setup.sh
  . /afs/cern.ch/sw/IntelSoftware/linux/x86_64/xe2013/bin/ifortvars.sh intel64
  . /afs/cern.ch/sw/IntelSoftware/linux/x86_64/xe2013/bin/iccvars.sh intel64
  export CC=icc
  export CXX=icc
  export FC=ifort
fi

echo ${THIS}/setup.py -o ${LABEL} -c ${COMPILER} -b ${BUILDTYPE} -v ${EXTERNALS}
eval `${THIS}/setup.py -o ${LABEL} -c ${COMPILER} -b ${BUILDTYPE} -v ${EXTERNALS}`