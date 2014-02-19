#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DBACKEND=CUDA -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
make
cd ../