cmake_minimum_required(VERSION 2.8.8)
project(vecgeom)

################################################################################

option(VC_ACCELERATION "Internal vectorization using Vc." ON)

option(CUDA "Enable compilation for CUDA." OFF)

option(CUDA_SPECIALIZATION "Use specialized volumes for CUDA." ON)

option(ROOT "Include ROOT." OFF)

option(USOLIDS "Include USolids." OFF)

option(BENCHMARK "Enable performance comparisons." OFF)

option(NO_SPECIALIZATION "Disable specialization of volumes." OFF)

option(COVERAGE_TESTING "Enable coverage testing flags." OFF)

if (USOLIDS)
  set(USolids ON)
endif()

if (NOT BACKEND)
  message(STATUS "Backend not specified. Defaulting to Vc.")
  set(BACKEND "Vc")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/modules)

################################################################################

# Determine backend

set(Vc)
set(Cilk)
set(Scalar)

if (BACKEND STREQUAL "Vc")
  set(Vc TRUE)
endif()
if (BACKEND STREQUAL "Cilk")
  set(Cilk TRUE)
endif()
if (BACKEND STREQUAL "Scalar")
  set(Scalar TRUE)
endif()
if (NOT Vc AND NOT Cilk AND NOT Scalar)
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -DVECGEOM")
if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Weffc++ -ggdb")
  if (APPLE OR UNIX) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()
if (GNU)
  if (NOT ERROR_LIMIT AND NOT ERROR_LIMIT EQUAL 0)
    set(ERROR_LIMIT 20)
  endif()
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -fmax-errors=${ERROR_LIMIT} -finline-limit=10000000")
  # Enable pretty diagnostics coloring if gcc 4.9+ is being used
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                  OUTPUT_VARIABLE GCC_VERSION)
  if (GCC_VERSION VERSION_GREATER 4.9 OR GCC_VERSION VERSION_EQUAL 4.9)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=auto")
  endif()
  if (COVERAGE_TESTING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(LIBS ${LIBS} gcov)
  endif()
endif()
if (Intel)
  set(CMAKE_CXX_FLAGS "-xHost ${CMAKE_CXX_FLAGS}")
  if (COVERAGE_TESTING)
    message(FATAL_ERROR "Coverage testing not supported for icc.")
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -ftree-vectorize")
endif()
if (VC_ACCELERATION)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_VC_ACCELERATION")
endif()
if (BENCHMARK)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_BENCHMARK")
endif()
if (NO_SPECIALIZATION)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_NO_SPECIALIZATION")
endif()
if (Clang)
  if (COVERAGE_TESTING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(LIBS ${LIBS} profile_rt)
  endif()
endif()

################################################################################

if (Scalar)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_SCALAR")
endif()

if (VC_ACCELERATION OR Vc)

  find_package(Vc REQUIRED)
  set(LIBS ${LIBS} ${Vc_LIBRARIES})
 include_directories(AFTER SYSTEM ${Vc_INCLUDE_DIR})
endif()

if (Vc)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_VC")

  if (GNU)
    # Fixes ABI issues with Vc using GNU compiler
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=6")
  endif()
  set(BACKEND_INCLUDE backend/vc)
  set(SRC_EXTERNAL ${SRC_EXTERNAL} source/backend/vc/Backend.cpp)
  set(HEADERS_BACKEND ${HEADERS_BACKEND}
    ${CMAKE_SOURCE_DIR}/backend/vc/Backend.h
  )

endif()

if (Cilk)

  if (NOT Intel)
    message(FATAL_ERROR "Must use Intel C++ compiler (icc) for Cilk backend.")
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_CILK")

  set(BACKEND_INCLUDE backend/cilk)
  set(
    SRC_EXTERNAL ${SRC_EXTERNAL}
    source/backend/cilk/Backend.cpp
  )
  set(HEADERS_BACKEND ${HEADERS_BACKEND}
    ${CMAKE_SOURCE_DIR}/backend/cilk/Backend.h
  )

endif()

if (CUDA)

  find_package(CUDA REQUIRED)
  include(FindCUDA)

  if (NOT CUDA_ARCH)
    set(CUDA_ARCH 20)
  endif()
  set(CUDA_ARCH "-arch=sm_${CUDA_ARCH}")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_CUDA")
  if (NOT CUDA_SPECIALIZATION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_CUDA_NO_SPECIALIZATION")
  endif()

  set(
    CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -O2 --use_fast_math -Xcudafe "--diag_suppress=code_is_unreachable"
    -Xcudafe "--diag_suppress=initialization_not_reachable"
  )
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G)
  endif()
  set(CUDA_SEPARABLE_COMPILATION ON)
  # set(CUDA_VERBOSE_BUILD ON)

  message(STATUS "Compiling with NVCC flags: ${CUDA_NVCC_FLAGS}")

  set(BACKEND_INCLUDE_CUDA backend/cuda)
  set(SRC_EXTERNAL ${SRC_EXTERNAL} source/CudaManager.cpp)
  set(
    SRC_CUDA ${SRC_CUDA}
    source/CudaManager.cu
    source/backend/cuda/Interface.cu
  )

endif()

if (ROOT)
  
  find_package(ROOT REQUIRED)

  include_directories(AFTER SYSTEM ${ROOT_INCLUDE_DIR})
  link_directories(${ROOT_LIBRARY_DIR})
  link_directories(${ROOT_LIBRARY_DIR}/root) # For natively installed ROOT
  set(LIBS ${LIBS} "-lCint -lCore -lMathCore -lRIO -lThread -lGeom")
  set(SRC_EXTERNAL ${SRC_EXTERNAL}
    source/PlacedRootVolume.cpp
    source/RootGeoManager.cpp
    source/UnplacedRootVolume.cpp
  )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_ROOT")

endif()

if (USolids)

  if (NOT USOLIDS_DIR AND (NOT USOLIDS_INCLUDE_DIR OR NOT USOLIDS_LIBRARY_DIR))
    if (NOT ROOT)
      message(FATAL_ERROR "ROOT required to build USolids from source.")
    endif()
    message(STATUS "Building with shipped USolids.")
    add_subdirectory(USolids)
    include_directories(${CMAKE_SOURCE_DIR}/USolids/include)
    link_directories(${CMAKE_BINARY_DIR}/USolids)
  endif()
  if (USOLIDS_INCLUDE_DIR)
    include_directories(${USOLIDS_INCLUDE_DIR})
  elseif (USOLIDS_DIR)
    include_directories(${USOLIDS_DIR}/include)
  endif()
  if (USOLIDS_LIBRARY_DIR)
    link_directories(${USOLIDS_LIBRARY_DIR})
  elseif (USOLIDS_DIR)
    link_directories(${USOLIDS_DIR}/lib)
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_USOLIDS")
  
  set(LIBS ${LIBS} usolids)

endif()

message(STATUS "Compiling with C++ flags: ${CMAKE_CXX_FLAGS}")

################################################################################

include_directories(${CMAKE_SOURCE_DIR})

set(SRC_CPP_RELATIVE
  source/AOS3D.cpp
  source/LogicalVolume.cpp
  source/PlacedBox.cpp
  source/PlacedTube.cpp
  source/PlacedParallelepiped.cpp
  source/PlacedParaboloid.cpp
  source/PlacedTrapezoid.cpp
  source/PlacedVolume.cpp
  source/SOA3D.cpp
  source/SpecializedBox.cpp
  source/Transformation3D.cpp
  source/UnplacedBox.cpp
  source/UnplacedTube.cpp
  source/UnplacedParaboloid.cpp
  source/UnplacedParallelepiped.cpp
  source/UnplacedTrapezoid.cpp
  source/UnplacedVolume.cpp
  source/Vector.cpp
  source/Vector3D.cpp
  source/NavigationState.cpp	
  source/SimpleNavigator.cpp
)
foreach(SRC ${SRC_CPP_RELATIVE})
  set(SRC_CPP ${SRC_CPP} ${CMAKE_SOURCE_DIR}/${SRC})
endforeach()

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
  set(SRC_CPP ${SRC_CPP}
    source/benchmarking/BenchmarkResult.cpp
    source/benchmarking/Benchmarker.cpp
    source/benchmarking/VolumePointers.cpp
  )
  set(SRC_CUDA ${SRC_CUDA}
    source/benchmarking/Benchmarker.cu
  )
endif()
set(SRC_CPP ${SRC_CPP} ${SRC_EXTERNAL}
  source/GeoManager.cpp
)

# Compile cpp-files for C++11

set_source_files_properties(${SRC_CPP} PROPERTIES COMPILE_FLAGS -std=c++11)

################################################################################

# Build libraries

add_library(vecgeom_cpp ${SRC_CPP})
target_link_libraries(vecgeom_cpp ${LIBS})
set(LIBS ${LIBS} vecgeom_cpp)
if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(LIBS ${LIBS} -lrt)
endif()

if (CUDA)
  cuda_add_library(
    vecgeom_cuda
    ${SRC_CUDA}
    OPTIONS ${CUDA_ARCH}
  )
  target_link_libraries(vecgeom_cuda ${LIBS})
  set(LIBS ${LIBS} vecgeom_cuda)
  install(TARGETS vecgeom_cuda DESTINATION lib)
endif()

################################################################################

# Define executables

set(TEST_EXECUTABLES_CORE
  ${CMAKE_SOURCE_DIR}/test/core/create_geometry.cpp
  ${CMAKE_SOURCE_DIR}/test/core/testVectorSafety.cpp
  ${CMAKE_SOURCE_DIR}/test/core/ContainerTest.cpp
)
set(TEST_EXECUTABLES_ROOT
  ${CMAKE_SOURCE_DIR}/test/root/root_geometry.cpp
  ${CMAKE_SOURCE_DIR}/test/root/complex_test1.cpp
  ${CMAKE_SOURCE_DIR}/test/root/ImportFromRootFileTest.cpp
)
set(TEST_EXECUTABLES_USOLIDS
  ${CMAKE_SOURCE_DIR}/test/usolids/CompatibilityTest.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestBox.cpp
)
set(TEST_EXECUTABLES_BENCHMARK
  ${CMAKE_SOURCE_DIR}/test/benchmark/BoxBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/MultiBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/ParallelepipedBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/TubeBenchmark.cpp
)
set(TEST_EXECUTABLES_CUDA
  # Files go here
)
set(TEST_EXECUTABLES
  ${TEST_EXECUTABLES_CORE}
  ${TEST_EXECUTABLES_ROOT}
  ${TEST_EXECUTABLES_USOLIDS}
  ${TEST_EXECUTABLES_BENCHMARK}
  ${TEST_EXECUTABLES_CUDA}
)
set_source_files_properties(
  ${TEST_EXECUTABLES}
  PROPERTIES COMPILE_FLAGS -std=c++11
)

# Build executables

function(build_executables EXECUTABLES)
  foreach(EXECUTABLE ${EXECUTABLES})
    get_filename_component(TARGET_NAME ${EXECUTABLE} NAME_WE)
    add_executable(${TARGET_NAME} ${EXECUTABLE})
    target_link_libraries(${TARGET_NAME} ${LIBS})
  endforeach()
endfunction()

build_executables("${TEST_EXECUTABLES_CORE}")
if (BENCHMARK)
  build_executables("${TEST_EXECUTABLES_BENCHMARK}")
endif()
if (ROOT)
  build_executables("${TEST_EXECUTABLES_ROOT}")
endif()
if (USolids)
  build_executables("${TEST_EXECUTABLES_USOLIDS}")
endif()
if (CUDA)
  build_executables("${TEST_EXECUTABLES_CUDA}")
endif()

################################################################################

# Install headers and libraries

set(HEADER_FOLDERS
  base
  benchmarking
  management
  navigation
  volumes
)
set(BACKEND_FOLDERS
  backend/scalar
  backend/vc
  backend/cilk
  backend/cuda
)
foreach(FOLDER ${HEADER_FOLDERS})
  install(DIRECTORY ${FOLDER} DESTINATION include)
endforeach()
foreach(FOLDER ${BACKEND_FOLDERS})
  install(DIRECTORY ${FOLDER} DESTINATION include/backend)
endforeach()
install(FILES ${CMAKE_SOURCE_DIR}/backend/Backend.h DESTINATION include/backend)

foreach(HEADER ${HEADERS_BACKEND})
  get_filename_component(FILENAME ${HEADER} NAME)
  install(FILES ${HEADER} DESTINATION include/backend/vector)
endforeach()
install(TARGETS vecgeom_cpp DESTINATION lib)
