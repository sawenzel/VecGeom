#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DBACKEND=Cilk -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icc -DROOT=ON -DVC_ACCELERATION=ON
make
cd ../