#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DBACKEND=CUDA -DCMAKE_CXX_COMPILER=icc -DCMAKE_C_COMPILER=icc
make -j4
cd ../