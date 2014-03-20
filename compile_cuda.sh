#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DCUDA=ON
make
cd ../