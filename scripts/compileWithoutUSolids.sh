#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../../ -DGEANT4=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release -DBACKEND=Vc -DBENCHMARK=ON -DROOT=ON -DUSOLIDS=OFF -DVECGEOM_VECTOR=avx -DVECTOR=avx -DVc_DIR=/home/swenzel/local/vc0.8/lib/cmake/Vc
make -j 8
cd ../../

