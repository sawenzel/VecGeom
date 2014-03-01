#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DBACKEND=CUDA -DCUDA_ARCH=sm_35
make
cd ../