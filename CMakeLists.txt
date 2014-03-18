cmake_minimum_required(VERSION 2.8.12)
project(vecgeom)

################################################################################

option(VC_ACCELERATION "Internal vectorization using Vc." ON)

option(ROOT "Include ROOT." OFF)

option(USOLIDS "Include USolids." OFF)

option(BENCHMARK "Enable performance comparisons, requiring ROOT and USolids." OFF)

if (BENCHMARK)
  set(ROOT ON)
  set(USolids ON)
endif()

if (USOLIDS)
  set(USolids ON)
endif()

if (NOT BACKEND)
  set(BACKEND "Vc")
endif()

if (NOT CUDA_ARCH)
  set(CUDA_ARCH sm_20)
endif()
set(CUDA_ARCH "-arch=${CUDA_ARCH}")

################################################################################

# Determine backend

if (BACKEND STREQUAL "Vc")
  set(Vc TRUE)
endif()
if (BACKEND STREQUAL "CUDA")
  set(CUDA TRUE)
endif()
if (BACKEND STREQUAL "Cilk")
  set(Cilk TRUE)
endif()

message(STATUS "Configuring with backend ${BACKEND}.")

# NVCC cannot compile Vc, which is C++11 exclusive
if (CUDA)
  set(VC_ACCELERATION OFF)
endif()


# Determine C++ compiler being used

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(Clang TRUE)
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(GNU TRUE)
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(Intel TRUE)
endif()


# Allow specifying vector instruction set

if ((NOT Intel) AND VECTOR)
  message(STATUS "Compiling with vector instruction set ${VECTOR}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m${VECTOR}")
endif()

################################################################################

# Set C++ compiler flags. Will also be forwarded by CUDA when compiling C++.

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -ftree-vectorize")
if (GNU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finline-limit=1000")
endif()
execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
if (GNU)
  if (GCC_VERSION VERSION_GREATER 4.9 OR GCC_VERSION VERSION_EQUAL 4.9)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=auto")
  endif()
endif()
if (CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_CUDA -Wno-vla")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()
if (Intel)
  set(CMAKE_CXX_FLAGS "-xHost ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
endif()
if (VC_ACCELERATION)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_VC_ACCELERATION")
endif()
if (BENCHMARK)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_BENCHMARK")
endif()

################################################################################

if (VC_ACCELERATION OR Vc)

  find_package(Vc REQUIRED)
  set(LIBS ${LIBS} ${Vc_LIBRARIES})
  # Dirty workaround because includes are not properly found by CMake
  get_filename_component(Vc_LIBPATH ${Vc_LIBRARIES} PATH)
  include_directories(${Vc_LIBPATH}/../include/)

endif()

if (Vc)

  if (GNU)
    # Fixes ABI issues with Vc using GNU compiler
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=6")
  endif()
  include_directories(${CMAKE_SOURCE_DIR}/backend/vc)
  file(GLOB SRC_BACKEND "source/backend/vc/*.cpp")
  set(SRC_COMPILETEST ${CMAKE_SOURCE_DIR}/test/compile_vc.cpp)

endif()

if (Cilk)

  if (NOT Intel)
    message(FATAL_ERROR "Must use Intel C++ compiler for Cilk backend.")
  endif()

  include_directories(${CMAKE_SOURCE_DIR}/backend/cilk)
  file(GLOB SRC_BACKEND "source/backend/cilk/*.cpp")
  set(SRC_COMPILETEST ${CMAKE_SOURCE_DIR}/test/compile_cilk.cpp)

endif()

if (CUDA)

  find_package(CUDA REQUIRED)
  include(FindCUDA)

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O2 --use_fast_math -Xcudafe "--diag_suppress=code_is_unreachable" -Xcudafe "--diag_suppress=initialization_not_reachable")
  IF(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G)
  endif()
  set(CUDA_SEPARABLE_COMPILATION ON)
  # set(CUDA_VERBOSE_BUILD ON)

  message(STATUS "Compiling with NVCC flags: ${CUDA_NVCC_FLAGS}")

  include_directories(${CMAKE_SOURCE_DIR}/backend/cuda)
  file(GLOB SRC_BACKEND source/backend/cuda/*.cpp source/backend/cuda/*.cu)
  set(SRC_COMPILETEST ${CMAKE_SOURCE_DIR}/test/compile_cuda.cu)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

endif()

if (ROOT)
  
  include(FindROOT.cmake)

  include_directories(${ROOT_INCLUDE_DIR})
  link_directories(${ROOT_LIBRARY_DIR})
  set(LIBS ${LIBS} ${ROOT_LIBRARIES} -lGeom)
  set(SRC_CPP ${SRC_CPP} ${CMAKE_SOURCE_DIR}/source/root_manager.cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_ROOT")

endif()

if (USolids)

  if (NOT USOLIDS_DIR AND (NOT USOLIDS_INCLUDE_DIR OR NOT USOLIDS_LIBRARY_DIR))
    message(FATAL_ERROR "Must specify USOLIDS_DIR or USOLIDS_INCLUDE_DIR and USOLIDS_LIBRARY_DIR to build benchmarking module.")
  endif()
  if (USOLIDS_INCLUDE_DIR)
    include_directories(${USOLIDS_INCLUDE_DIR})
  else()
    include_directories(${USOLIDS_DIR}/include)
  endif()
  if (USOLIDS_LIBRARY_DIR)
    set(USOLIDS_LIBRARY ${USOLIDS_LIBRARY_DIR}/libusolids.a)
    link_directories(${USOLIDS_LIBRARY_DIR})
  else()
    set(USOLIDS_LIBRARY ${USOLIDS_DIR}/lib/libusolids.a)
    link_directories(${USOLIDS_DIR}/lib)
  endif()
  set(LIBS ${LIBS} ${USOLIDS_LIBRARY})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_USOLIDS")

endif()

message(STATUS "Compiling with C++ flags: ${CMAKE_CXX_FLAGS}")

################################################################################

include_directories(${CMAKE_SOURCE_DIR})

set(SRC_CPP ${SRC_CPP}
  ${CMAKE_SOURCE_DIR}/source/logical_volume.cpp
  ${CMAKE_SOURCE_DIR}/source/placed_box.cpp
  ${CMAKE_SOURCE_DIR}/source/placed_volume.cpp
  ${CMAKE_SOURCE_DIR}/source/scalar_backend.cpp
  ${CMAKE_SOURCE_DIR}/source/transformation_matrix.cpp
  ${CMAKE_SOURCE_DIR}/source/unplaced_box.cpp
  ${CMAKE_SOURCE_DIR}/source/unplaced_volume.cpp
)
set(SRC_CPP ${SRC_CPP} ${SRC_BACKEND})

if (BENCHMARK)
  file(GLOB SRC_BENCHMARK "${CMAKE_SOURCE_DIR}/source/benchmarking/*.cpp")
  set(SRC_CPP ${SRC_CPP} ${SRC_BENCHMARK})
endif()

# Copy all source files to .cu-files in order for NVCC to compile them as CUDA
# code and not regular C++ files.

if (CUDA)

  file(GLOB EXE_CPP "test/*.cpp")

  foreach(SRC_FILE ${EXE_CPP})

    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME_WE)

    ADD_CUSTOM_COMMAND(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu
      COMMAND ${CMAKE_COMMAND} -E copy ${SRC_FILE} ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu
      DEPENDS ${SRC_FILE}
    )

  endforeach()

  foreach(SRC_FILE ${SRC_CPP})

    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME_WE)

    ADD_CUSTOM_COMMAND(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu
      COMMAND ${CMAKE_COMMAND} -E copy ${SRC_FILE} ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu
      DEPENDS ${SRC_FILE}
    )

    set(SRC_CUDA ${SRC_CUDA} ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu)

  endforeach()

endif()

################################################################################

# Compile and link

if (NOT CUDA)
  add_library(vecgeom_cpp ${SRC_CPP})
  target_link_libraries(vecgeom_cpp ${LIBS})
  set(LIBS ${LIBS} vecgeom_cpp)
  add_executable(create_geometry_test ${CMAKE_SOURCE_DIR}/test/create_geometry.cpp)
  target_link_libraries(create_geometry_test ${LIBS})
  if (ROOT)
    add_executable(root_geometry ${CMAKE_SOURCE_DIR}/test/root_geometry.cpp)
    target_link_libraries(root_geometry ${LIBS})
  endif()
  if (BENCHMARK)
    add_executable(benchmark ${CMAKE_SOURCE_DIR}/test/benchmark.cpp)
    target_link_libraries(benchmark ${LIBS})
    add_executable(complex_test1 ${CMAKE_SOURCE_DIR}/test/complex_test1.cpp)
    target_link_libraries(complex_test1 ${LIBS})
  endif()
else()
  cuda_add_executable(create_geometry_test ${SRC_CUDA} ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/create_geometry.cu OPTIONS ${CUDA_ARCH})
endif()
