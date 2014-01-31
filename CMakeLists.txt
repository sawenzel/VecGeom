cmake_minimum_required(VERSION 2.8)
project(vecgeom)

if (NOT backend)
  set(backend Vc)
endif()

if (backend STREQUAL "Vc")
  find_package(Vc REQUIRED)
  set(LIBS ${LIBS} ${Vc_LIBRARIES})
  get_filename_component(Vc_LIBPATH ${Vc_LIBRARIES} PATH)
  include_directories(${Vc_LIBPATH}/../include/)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -m64 -O2")
if (NOT (backend STREQUAL "CUDA"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()
if (CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=6 -ffast-math -finline-limit=10000000 -mavx -ftree-vectorize")
endif()

include_directories(${CMAKE_SOURCE_DIR})

add_executable(CompileTest ${CMAKE_SOURCE_DIR}/Test/CompileTest.cpp)