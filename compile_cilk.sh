#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DCMAKE_CXX_COMPILER=icc -DBACKEND=Cilk -DVC_ACCELERATION=ON
make
cd ../