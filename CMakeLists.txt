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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m${VECTOR}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -ftree-vectorize")
if (CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finline-limit=1000")
endif()
if (CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_CUDA")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()
if (Intel)
  set(CMAKE_CXX_FLAGS "-xHost ${CMAKE_CXX_FLAGS}")
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

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -rdc=true -arch=sm_30 -O2 --use_fast_math -Xcudafe "--diag_suppress=code_is_unreachable")
  set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(BUILD_SHARED_LIBS OFF)

  message(STATUS "Compiling with NVCC flags: ${CUDA_NVCC_FLAGS}")

  set(SRC_CPP ${SRC_CPP} ${CMAKE_SOURCE_DIR}/source/cuda_manager.cpp)
  set(SRC_COMPILETEST ${CMAKE_SOURCE_DIR}/test/compile_cuda.cu)

endif()

include_directories(${CMAKE_SOURCE_DIR})

set(SRC_CPP ${SRC_CPP} ${CMAKE_SOURCE_DIR}/source/transformation_matrix.cpp ${CMAKE_SOURCE_DIR}/source/logical_volume.cpp ${CMAKE_SOURCE_DIR}/source/placed_volume.cpp ${CMAKE_SOURCE_DIR}/source/unplaced_volume.cpp ${CMAKE_SOURCE_DIR}/source/scalar_backend.cpp)

message(STATUS "Compiling with C++ flags: ${CMAKE_CXX_FLAGS}")

if (NOT CUDA)
  add_library(vecgeom_cpp ${SRC_CPP})
else()
  cuda_add_library(vecgeom_cpp ${SRC_CPP})
endif()
set(LIBS ${LIBS} vecgeom_cpp)

if (NOT CUDA)
  add_executable(compile_test ${SRC_COMPILETEST} ${CMAKE_SOURCE_DIR}/test/compile_main.cpp)
else()
  cuda_add_executable(compile_test ${SRC_COMPILETEST} ${CMAKE_SOURCE_DIR}/test/compile_main.cpp)
endif()

target_link_libraries(compile_test ${LIBS})