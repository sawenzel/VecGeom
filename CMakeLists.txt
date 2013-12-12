cmake_minimum_required (VERSION 2.8)
project (fastgeom)

find_library(SOMELIB libVc.a)
find_library(SOMELIB libGeom.a)
add_library(libVc.a STATIC IMPORTED)
add_library(libGeom.a STATIC IMPORTED)

set(CMAKE_CXX_FLAGS "-fabi-version=6 -std=c++11 -m64 -lVc -lGeom")

option(VC_DIR "VC_DIR" "/usr/local/vc")

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("$ENV{ROOTSYS}/include")
include_directories("${VC_DIR}/include")
file(GLOB source_files ./ *.cpp Tests/ShapeTesterTester.cpp)

add_executable(fastgeom ${source_files})