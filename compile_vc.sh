#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DBACKEND=Vc
make -j4
cd ../