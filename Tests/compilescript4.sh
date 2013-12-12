#!/bin/bash
g++ -fabi-version=6 -m64 -std=c++11 `root-config --cflags --libs` -I ../ ShapeTesterTester.cpp ../*.cpp -o bin/fastgeom -lGeom -lVc -ltbb -lrt -lusolids