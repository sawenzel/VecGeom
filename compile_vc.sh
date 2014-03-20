#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DBACKEND=Vc
make
cd ../