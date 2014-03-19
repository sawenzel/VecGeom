#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DROOT=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBACKEND=Vc -DVc_DIR=/home/swenzel/repos/vc_build0.8/cmake
make
cd ../