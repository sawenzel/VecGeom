cmake_minimum_required(VERSION 2.8)
project(vecgeom)

# Default to Vc
if (NOT BACKEND)
  set(BACKEND Vc)
endif ()

option(VC_ACCELERATION "Internal vectorization using Vc." ON)

if (BACKEND STREQUAL "Vc")
  set(Vc TRUE)
endif()
if (BACKEND STREQUAL "CUDA")
  set(CUDA TRUE)
endif()
if (BACKEND STREQUAL "Cilk")
  set(Cilk TRUE)
endif()

if (CUDA)
  set(VC_ACCELERATION OFF)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(Clang TRUE)
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(GNU TRUE)
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(Intel TRUE)
endif()

message(STATUS "Configuring with backend ${BACKEND}.")

if ((NOT Intel) AND VECTOR)
  message(STATUS "Compiling with vector instruction set ${VECTOR}")
  set(CMAKE_CXX_OPT "${CMAKE_CXX_OPT} -m${VECTOR}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -ftree-vectorize")
if (CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finline-limit=1000")
endif()
set(CMAKE_CXX_OPT "${CMAKE_CXX_OPT} -std=c++11")
if (Clang)
  set(CMAKE_CXX_OPT "${CMAKE_CXX_OPT} -Wgnu-static-float-init")
endif()
if (CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_CUDA")
endif()
if (Intel)
  set(CMAKE_CXX_OPT "${CMAKE_CXX_OPT} -xHost")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
endif()

if (VC_ACCELERATION OR Vc)

  find_package(Vc REQUIRED)
  set(LIBS ${LIBS} ${Vc_LIBRARIES})
  get_filename_component(Vc_LIBPATH ${Vc_LIBRARIES} PATH)
  include_directories(${Vc_LIBPATH}/../include/)

endif()

if (VC_ACCELERATION)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_VC_ACCELERATION")
endif()

if (Vc)

  if (GNU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=6")
  endif()
  set(SRC_CPP ${SRC_CPP} ${CMAKE_SOURCE_DIR}/source/vc_backend.cpp)
  set(SRC_COMPILETEST ${CMAKE_SOURCE_DIR}/test/compile_vc.cpp)

endif()

if (Cilk)

  if (NOT Intel)
    message(FATAL_ERROR "Must use Intel C++ compiler for Cilk backend.")
  endif()

  set(SRC_CPP ${SRC_CPP} ${CMAKE_SOURCE_DIR}/source/cilk_backend.cpp)
  set(SRC_COMPILETEST ${CMAKE_SOURCE_DIR}/test/compile_cilk.cpp)

endif()

if (CUDA)

  find_package(CUDA REQUIRED)
  include(FindCUDA)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_30 -O2 --use_fast_math -Xcudafe "--diag_suppress=code_is_unreachable")
  set(SRC_COMPILETEST ${CMAKE_SOURCE_DIR}/test/compile_cuda.cu)

endif()

include_directories(${CMAKE_SOURCE_DIR})

if (CUDA)
  message(STATUS "Compiling with NVCC flags: ${CUDA_NVCC_FLAGS}")
  message(STATUS "Compiling for CUDA with C++ flags: ${CMAKE_CXX_FLAGS}")
  cuda_compile(CUDA_COMPILETEST_O ${CMAKE_SOURCE_DIR}/test/compile_cuda.cu)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_OPT}")
message(STATUS "Compiling with C++ flags: ${CMAKE_CXX_FLAGS}")

set(SRC_CPP ${SRC_CPP} ${CMAKE_SOURCE_DIR}/source/logical_volume.cpp)

if (SRC_CPP)
  add_library(vecgeom_cpp STATIC ${SRC_CPP})
  set(LIBS ${LIBS} vecgeom_cpp)
endif()

if (NOT CUDA)
  add_library(vecgeom_compiletest STATIC ${SRC_COMPILETEST})
else()
  cuda_add_library(vecgeom_compiletest ${CUDA_COMPILETEST_O})
endif()

add_executable(compile_test ${CMAKE_SOURCE_DIR}/test/compile_main.cpp)
target_link_libraries(compile_test ${LIBS} vecgeom_compiletest)