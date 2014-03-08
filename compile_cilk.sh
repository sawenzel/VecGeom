#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DCMAKE_CXX_COMPILER=icc -DCMAKE_C_COMPILER=icc -DBACKEND=Cilk
make
cd ../