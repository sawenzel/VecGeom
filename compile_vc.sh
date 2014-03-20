#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DBACKEND=Vc -DROOT=ON -DVC_ACCELERATION=ON
make
cd ../