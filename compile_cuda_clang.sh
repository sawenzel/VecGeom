#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DBACKEND=CUDA -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
make -j4
cd ../