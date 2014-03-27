cmake_minimum_required(VERSION 2.8.8)
project(vecgeom)

################################################################################

option(VC_ACCELERATION "Internal vectorization using Vc." ON)

option(CUDA "Enable compilation for CUDA." OFF)

option(CUDA_SPECIALIZATION "Use specialized volumes for CUDA." ON)

option(ROOT "Include ROOT." OFF)

option(USOLIDS "Include USolids." OFF)

option(BENCHMARK "Enable performance comparisons, requiring ROOT and USolids."
       OFF)

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
if (BACKEND STREQUAL "Cilk")
  set(Cilk TRUE)
endif()
if (NOT Vc AND NOT Cilk)
  message(FATAL_ERROR "Configuring with invalid backend \"${BACKEND}\".")
endif()

message(STATUS "Configuring with vector backend ${BACKEND}.")


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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ftree-vectorize")
if (GNU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finline-limit=1000")
  # Enable pretty diagnostics coloring if gcc 4.9+ is being used
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                  OUTPUT_VARIABLE GCC_VERSION)
  if (GCC_VERSION VERSION_GREATER 4.9 OR GCC_VERSION VERSION_EQUAL 4.9)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=auto")
  endif()
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

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_VC")

  if (GNU)
    # Fixes ABI issues with Vc using GNU compiler
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=6")
  endif()
  set(BACKEND_INCLUDE ${CMAKE_SOURCE_DIR}/backend/vc)
  set(
    SRC_EXTERNAL ${SRC_EXTERNAL}
    ${CMAKE_SOURCE_DIR}/source/backend/vc/backend.cpp
  )

endif()

if (Cilk)

  if (NOT Intel)
    message(FATAL_ERROR "Must use Intel C++ compiler (icc) for Cilk backend.")
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_CILK")

  set(BACKEND_INCLUDE ${CMAKE_SOURCE_DIR}/backend/cilk)
  set(
    SRC_EXTERNAL ${SRC_EXTERNAL}
    ${CMAKE_SOURCE_DIR}/source/backend/cilk/backend.cpp
  )

endif()

if (CUDA)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_CUDA")
  if (NOT CUDA_SPECIALIZATION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_CUDA_NO_SPECIALIZATION")
  endif()

  find_package(CUDA REQUIRED)
  include(FindCUDA)

  set(
    CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -O2 --use_fast_math -Xcudafe "--diag_suppress=code_is_unreachable"
    -Xcudafe "--diag_suppress=initialization_not_reachable"
  )
  IF(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G)
  endif()
  set(CUDA_SEPARABLE_COMPILATION ON)
  # set(CUDA_VERBOSE_BUILD ON)

  message(STATUS "Compiling with NVCC flags: ${CUDA_NVCC_FLAGS}")

  set(BACKEND_INCLUDE_CUDA ${CMAKE_SOURCE_DIR}/backend/cuda)
  set(SRC_EXTERNAL ${SRC_EXTERNAL} ${CMAKE_SOURCE_DIR}/source/cuda_manager.cpp)
  set(
    SRC_CUDA ${SRC_CUDA}
    ${CMAKE_SOURCE_DIR}/source/backend/cuda/interface.cu
  )

endif()

if (ROOT)
  
  include(FindROOT.cmake)

  include_directories(${ROOT_INCLUDE_DIR})
  link_directories(${ROOT_LIBRARY_DIR})
  set(LIBS ${LIBS} ${ROOT_LIBRARIES} -lGeom)
  set(SRC_EXTERNAL ${SRC_EXTERNAL} ${CMAKE_SOURCE_DIR}/source/root_manager.cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_ROOT")

endif()

if (USolids)

  if (NOT USOLIDS_DIR AND (NOT USOLIDS_INCLUDE_DIR OR NOT USOLIDS_LIBRARY_DIR))
    message(FATAL_ERROR "Must specify USOLIDS_DIR or USOLIDS_INCLUDE_DIR and
                         USOLIDS_LIBRARY_DIR to build benchmarking module.")
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
  ${CMAKE_SOURCE_DIR}/source/aos3d.cpp
  ${CMAKE_SOURCE_DIR}/source/container.cpp
  ${CMAKE_SOURCE_DIR}/source/geo_manager.cpp
  ${CMAKE_SOURCE_DIR}/source/logical_volume.cpp
  ${CMAKE_SOURCE_DIR}/source/placed_box.cpp
  ${CMAKE_SOURCE_DIR}/source/placed_volume.cpp
  ${CMAKE_SOURCE_DIR}/source/scalar_backend.cpp
  ${CMAKE_SOURCE_DIR}/source/soa3d.cpp
  ${CMAKE_SOURCE_DIR}/source/transformation_matrix.cpp
  ${CMAKE_SOURCE_DIR}/source/unplaced_box.cpp
  ${CMAKE_SOURCE_DIR}/source/unplaced_volume.cpp
)

# Copy all source files to .cu-files in order for NVCC to compile them as CUDA
# code and not regular C++ files.

if (CUDA)

  foreach(SRC_FILE ${SRC_CPP})

    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME_WE)

    ADD_CUSTOM_COMMAND(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu
      COMMAND ${CMAKE_COMMAND} -E copy ${SRC_FILE}
          ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu
      DEPENDS ${SRC_FILE}
    )

    set(
      SRC_CUDA ${SRC_CUDA}
      ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu
    )

  endforeach()

endif()

# Add files that won't be compiled for CUDA

if (BENCHMARK)
  file(GLOB SRC_BENCHMARK "${CMAKE_SOURCE_DIR}/source/benchmarking/*.cpp")
  set(SRC_CPP ${SRC_CPP} ${SRC_BENCHMARK})
endif()
set(SRC_CPP ${SRC_CPP} ${SRC_EXTERNAL})

# Compile cpp-files for C++11

set_source_files_properties(${SRC_CPP} PROPERTIES COMPILE_FLAGS -std=c++11)

################################################################################

# Compile and link

add_library(vecgeom_cpp ${SRC_CPP})
target_link_libraries(vecgeom_cpp ${LIBS})
set(LIBS ${LIBS} vecgeom_cpp)

set(
  SRC_EXECUTABLES
  ${CMAKE_SOURCE_DIR}/test/create_geometry.cpp
  ${CMAKE_SOURCE_DIR}/test/root_geometry.cpp
  ${CMAKE_SOURCE_DIR}/test/complex_test1.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/cpugpu.cpp
)
set_source_files_properties(
  ${SRC_EXECUTABLES}
  PROPERTIES COMPILE_FLAGS -std=c++11
)

if (NOT CUDA)
  add_executable(
    create_geometry_test
    ${CMAKE_SOURCE_DIR}/test/create_geometry.cpp
  )
  target_link_libraries(create_geometry_test ${LIBS})
  if (ROOT)
    add_executable(root_geometry ${CMAKE_SOURCE_DIR}/test/root_geometry.cpp)
    target_link_libraries(root_geometry ${LIBS})
    add_executable(complex_test1 ${CMAKE_SOURCE_DIR}/test/complex_test1.cpp)
    target_link_libraries(complex_test1 ${LIBS})
  endif()
  if (BENCHMARK)
    add_executable(benchmark ${CMAKE_SOURCE_DIR}/test/benchmark.cpp)
    target_link_libraries(benchmark ${LIBS})
  endif()
else()
  cuda_add_library(
    vecgeom_cuda
    ${SRC_CUDA}
    OPTIONS ${CUDA_ARCH}
  )
  target_link_libraries(vecgeom_cuda ${LIBS})
  set(LIBS ${LIBS} vecgeom_cuda)
  cuda_add_executable(
    cpugpu
    ${CMAKE_SOURCE_DIR}/test/cpugpu.cpp
    OPTIONS ${CUDA_ARCH}
  )
  target_link_libraries(cpugpu ${LIBS})
endif()