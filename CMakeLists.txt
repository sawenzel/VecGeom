cmake_minimum_required(VERSION 2.8)
project(vecgeom)

# Default to Vc
if (NOT BACKEND)
  set(BACKEND Vc)
endif ()

if (BACKEND STREQUAL "Vc")
  set(Vc TRUE)
endif()
if (BACKEND STREQUAL "CUDA")
  set(CUDA TRUE)
endif()
if (BACKEND STREQUAL "Cilk")
  set(Cilk TRUE)
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

if ((NOT Cilk) AND VECTOR)
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

if (Vc)

  find_package(Vc REQUIRED)
  set(LIBS ${LIBS} ${Vc_LIBRARIES})
  get_filename_component(Vc_LIBPATH ${Vc_LIBRARIES} PATH)
  include_directories(${Vc_LIBPATH}/../include/)
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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost")

else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
endif()

if (CUDA)

  find_package(CUDA REQUIRED)
  include(FindCUDA)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_30 -O2 --use_fast_math)
  set(SRC_COMPILETEST ${CMAKE_SOURCE_DIR}/test/compile_cuda.cu)

endif()

include_directories(${CMAKE_SOURCE_DIR})

if (CUDA)
  cuda_compile(COMPILE_O ${CMAKE_SOURCE_DIR}/test/compile_cuda.cu)
  cuda_add_library(vecgeomcuda ${COMPILE_O})
  set(LIBS ${LIBS} vecgeomcuda)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_OPT}")

if (SRC_CPP)
  add_library(vecgeomcpp STATIC ${SRC_CPP})
  set(LIBS ${LIBS} vecgeomcpp)
endif()

add_executable(compile_main ${CMAKE_SOURCE_DIR}/test/compile_main.cpp)
target_link_libraries(compile_main ${LIBS})