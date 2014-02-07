#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DBACKEND=CUDA -DCMAKE_CXX_COMPILER=icc
make
cd ../