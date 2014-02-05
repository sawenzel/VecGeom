#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DCMAKE_CXX_COMPILER=g++ -DBACKEND=Vc -DVECTOR=avx
make
cd ../