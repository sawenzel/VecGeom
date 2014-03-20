#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DCUDA=ON -DROOT=ON -DVC_ACCELERATION=ON
make
cd ../