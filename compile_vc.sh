#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DBACKEND=Vc -DVc_DIR=/home/swenzel/repos/vc_build0.8/cmake
make
cd ../