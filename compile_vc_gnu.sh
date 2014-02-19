#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DBACKEND=Vc -DVC_ACCELERATION=ON
make
cd ../