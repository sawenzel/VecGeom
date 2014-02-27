#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Debug -DCUDA_ARCH=sm_35
make
cd ../