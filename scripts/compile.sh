#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../../ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release -DBACKEND=Vc -DBENCHMARK=ON -DROOT=ON -DUSOLIDS=ON -DVECTOR=avx
make -j 8
cd ../../
