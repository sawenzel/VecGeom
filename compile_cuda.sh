#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DBACKEND=CUDA
make -j4
cd ../