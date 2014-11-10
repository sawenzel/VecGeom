cmake_minimum_required(VERSION 2.8.8)
project(Vecgeom)

set(VECGEOM ON)

################################################################################

# Configuration options

option(VC_ACCELERATION "Internal vectorization using Vc." ON)

option(CUDA "Enable compilation for CUDA." OFF)

option(CUDA_SPECIALIZATION "Use specialized volumes for CUDA." ON)

option(ROOT "Include ROOT." OFF)

option(USOLIDS "Include USolids." OFF)

option(GEANT4 "Include Geant4." OFF)

option(BENCHMARK "Enable performance comparisons." OFF)

option(NO_SPECIALIZATION "Disable specialization of volumes." OFF)

option(COVERAGE_TESTING "Enable coverage testing flags." OFF)

option(CTEST "Enable CTest when building." ON)

option(PLANESHELL "Enable the use of PlaneShell class for the trapezoid." ON)

set(VECGEOM_VECTOR "" CACHE STRING
    "Vector instruction set to use (sse/avx/...).")

set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries.")
set(INSTALL_INCLUDE_DIR include CACHE PATH
    "Installation directory for header files.")
set(INSTALL_CMAKE_DIR lib/CMake/Vecgeom CACHE PATH
    "Installation directory for CMake files.")

# Make relative paths absolute
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

if (USOLIDS)
  set(USolids ON)
endif()

if (GEANT4)
  if (BENCHMARK)
    set(Geant4 ON)
  else()
    message(WARNING
      "Enabling Geant4 without benchmarks has no effect and will be disabled.")
    set(GEANT4 OFF)
    set(Geant4 OFF)
  endif()
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
  message(FATAL_ERROR "Invalid backend specified: \"${BACKEND}\"")
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
if (NOT GNU AND NOT Clang AND NOT Intel)
  message(WARNING "Unsupported compiler. Build will likely fail.")
endif()

if ((NOT PLANESHELL))
  message(STATUS "Disabling PlaneShell class")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVECGEOM_PLANESHELL_DISABLE")
endif()

################################################################################

# Set C++ compiler flags. Will also be forwarded by CUDA when compiling C++.

set(VECGEOM_COMPILATION_FLAGS)
set(VECGEOM_COMPILATION_FLAGS_GNU)
set(VECGEOM_COMPILATION_FLAGS_CLANG)
set(VECGEOM_COMPILATION_FLAGS_INTEL)
set(VECGEOM_NVCC_COMPILATION_FLAGS)

set(VECGEOM_COMPILATION_FLAGS
    "${VECGEOM_COMPILATION_FLAGS} -Wall -fPIC")
if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++ -ggdb")
  if (APPLE OR UNIX) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

# Compiler specific optimization flags
set(VECGEOM_COMPILATION_FLAGS_GNU
    "${VECGEOM_COMPILATION_FLAGS_GNU} -finline-limit=10000000")
set(VECGEOM_COMPILATION_FLAGS_GNU
    "${VECGEOM_COMPILATION_FLAGS_GNU} -ffast-math -ftree-vectorize")
set(VECGEOM_COMPILATION_FLAGS_CLANG
    "${VECGEOM_COMPILATION_FLAGS_CLANG} -ffast-math -ftree-vectorize")
set(VECGEOM_COMPILATION_FLAGS_INTEL
    "${VECGEOM_COMPILATION_FLAGS_INTEL} -xHost")
if (VECGEOM_VECTOR)
  message(STATUS "Compiling with vector instruction set ${VECGEOM_VECTOR}")
  set(VECGEOM_COMPILATION_FLAGS_GNU
      "${VECGEOM_COMPILATION_FLAGS_GNU} -m${VECGEOM_VECTOR}")
  set(VECGEOM_COMPILATION_FLAGS_CLANG
      "${VECGEOM_COMPILATION_FLAGS_CLANG} -m${VECGEOM_VECTOR}")
endif()

# Backend specific
if (Scalar)
  set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_SCALAR")
endif()
if (Vc)
  set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_VC")
  set(VECGEOM_COMPILATION_FLAGS_GNU
      "${VECGEOM_COMPILATION_FLAGS_GNU} -fabi-version=6")
endif()
if (Cilk)
  set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_CILK")
endif()
if (CUDA)
  set(CUDA_ARCH 20 CACHE STRING "CUDA device architecture.")
  set(CUDA_ARCH "-arch=sm_${CUDA_ARCH}")
  set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_CUDA")
  if (NOT CUDA_SPECIALIZATION)
    set(VECGEOM_COMPILATION_FLAGS
        "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_CUDA_NO_SPECIALIZATION")
  endif()
  set(VECGEOM_NVCC_COMPILATION_FLAGS ${VECGEOM_NVCC_COMPILATION_FLAGS}
      -O2 --use_fast_math -Xcudafe "--diag_suppress=code_is_unreachable"
      -Xcudafe "--diag_suppress=initialization_not_reachable")
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G)
  endif()
endif()

# Enable pretty diagnostics coloring if gcc 4.9+ is being used
if (GNU)
  set(VECGEOM_ERROR_LIMIT 20 CACHE STRING
      "Limit number of errors output by diagnostics.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmax-errors=${VECGEOM_ERROR_LIMIT}")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                  OUTPUT_VARIABLE GCC_VERSION)
  if (GCC_VERSION VERSION_GREATER 4.9 OR GCC_VERSION VERSION_EQUAL 4.9)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=auto")
  endif()
endif()

# Coverage testing support
if (COVERAGE_TESTING)
  if (GNU)
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} gcov)
  endif()
  if (Clang)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} profile_rt)
  endif()
  if (Intel)
    message(FATAL_ERROR "Coverage testing not supported for icc.")
  endif()
endif()

# Forward CMake options
if (VC_ACCELERATION)
  set(VECGEOM_COMPILATION_FLAGS
      "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_VC_ACCELERATION")
endif()
if (BENCHMARK)
  set(VECGEOM_COMPILATION_FLAGS
      "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_BENCHMARK")
endif()
if (NO_SPECIALIZATION)
  set(VECGEOM_COMPILATION_FLAGS
      "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_NO_SPECIALIZATION")
endif()
if (ROOT)
  set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_ROOT")
endif()
if (USolids)
  set(VECGEOM_COMPILATION_FLAGS
      "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_USOLIDS")
endif()
if (Geant4)
  set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_GEANT4")
endif()

# Pass flags to compilers
if (GNU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECGEOM_COMPILATION_FLAGS_GNU}")
endif()
if (Clang)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECGEOM_COMPILATION_FLAGS_CLANG}")
endif()
if (Intel)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECGEOM_COMPILATION_FLAGS_INTEL}")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECGEOM_COMPILATION_FLAGS}")
set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -O2")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${VECGEOM_NVCC_COMPILATION_FLAGS})

message(STATUS "Compiling with C++ flags: ${CMAKE_CXX_FLAGS}")
if (CUDA)
  message(STATUS "Compiling with NVCC flags: ${CUDA_NVCC_FLAGS}")
endif()

################################################################################

# Backend configuration

if (VC_ACCELERATION OR Vc)

  find_package(Vc REQUIRED)
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} ${Vc_LIBRARIES})
  include_directories(AFTER SYSTEM ${Vc_INCLUDE_DIR})

  if (Vc)
    set(SRC_EXTERNAL ${SRC_EXTERNAL} source/backend/vc/Backend.cpp)
  endif()

endif()

if (Cilk)

  if (NOT Intel)
    message(FATAL_ERROR "Must use Intel C++ compiler (icc) for Cilk backend.")
  endif()

  set(SRC_EXTERNAL ${SRC_EXTERNAL} source/backend/cilk/Backend.cpp)

endif()

if (CUDA)

  find_package(CUDA REQUIRED)
  include(FindCUDA)

  set(CUDA_SEPARABLE_COMPILATION ON)

  set(SRC_EXTERNAL ${SRC_EXTERNAL} source/CudaManager.cpp)
  set(SRC_CUDA ${SRC_CUDA}
      source/CudaManager.cu source/backend/cuda/Interface.cu)

endif()

if (ROOT)
  
  find_package(ROOT REQUIRED)

  include_directories(AFTER SYSTEM ${ROOT_INCLUDE_DIR})
  link_directories(${ROOT_LIBRARY_DIR})
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL}
      Cint Core MathCore RIO Thread Geom Gui Net Hist Graf Graf3d Gpad Tree
      Rint Postscript Matrix Physics m dl)
  set(SRC_EXTERNAL ${SRC_EXTERNAL}
      source/PlacedRootVolume.cpp
      source/RootGeoManager.cpp
      source/UnplacedRootVolume.cpp
      source/Visualizer.cpp)

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
  
  set(VECGEOM_LIBRARIES ${VECGEOM_LIBRARIES} usolids)

endif()

if (Geant4)

  find_package(Geant4 REQUIRED)
  include_directories(AFTER SYSTEM ${Geant4_INCLUDE_DIRS})
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL}
      ${Geant4_LIBRARIES})
  message(STATUS "Geant4 found in: ${Geant4_INCLUDE_DIRS}")

endif()

################################################################################

include_directories(${CMAKE_SOURCE_DIR})

set(SRC_CPP_RELATIVE
  source/AOS3D.cpp
  source/LogicalVolume.cpp
  source/PlacedCone.cpp
  source/PlacedBox.cpp
  source/PlacedTube.cpp
  source/PlacedTrd.cpp
  source/PlacedParallelepiped.cpp
  source/PlacedParaboloid.cpp
  source/PlacedTrapezoid.cpp
  source/PlacedVolume.cpp
  source/SOA3D.cpp
  source/SpecializedBox.cpp
  source/Transformation3D.cpp
  source/UnplacedCone.cpp
  source/UnplacedBox.cpp
  source/UnplacedTube.cpp
  source/UnplacedTrd.cpp
  source/UnplacedParaboloid.cpp
  source/UnplacedParallelepiped.cpp
  source/UnplacedTrapezoid.cpp
  source/UnplacedVolume.cpp
  source/Vector.cpp
  source/NavigationState.cpp
  source/SimpleNavigator.cpp

  source/UnplacedOrb.cpp
  source/PlacedOrb.cpp

  source/UnplacedSphere.cpp
  source/PlacedSphere.cpp
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

if (USOLIDS)
  set(SRC_CPP ${SRC_CPP}
    test/shape_tester/ShapeTester.cpp
  )
endif()
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
  source/ShapeDebugger.cpp
)

# Compile cpp-files for C++11

set_source_files_properties(${SRC_CPP} PROPERTIES COMPILE_FLAGS -std=c++11)

################################################################################

# Build libraries

add_library(vecgeom ${SRC_CPP})
target_link_libraries(vecgeom ${VECGEOM_LIBRARIES}
                      ${VECGEOM_LIBRARIES_EXTERNAL})
set(VECGEOM_LIBRARIES ${VECGEOM_LIBRARIES} vecgeom)
if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} rt)
endif()

set_source_files_properties(
  ${SRC_EXECUTABLES}
  PROPERTIES COMPILE_FLAGS -std=c++11
)

if (CUDA)
  cuda_add_library(
    vecgeomcuda
    ${SRC_CUDA}
    OPTIONS ${CUDA_ARCH}
  )
  target_link_libraries(vecgeomcuda ${VECGEOM_LIBRARIES}
                        ${VECGEOM_LIBRARIES_EXTERNAL})
  set(VECGEOM_LIBRARIES ${VECGEOM_LIBRARIES} vecgeomcuda)
endif()

################################################################################

# Define executables

set(TEST_EXECUTABLES_CORE
  ${CMAKE_SOURCE_DIR}/test/core/ContainerTest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/create_geometry.cpp
  ${CMAKE_SOURCE_DIR}/test/core/testVectorSafety.cpp
  ${CMAKE_SOURCE_DIR}/test/core/Transformation3DTest.cpp
)
set(TEST_EXECUTABLES_ROOT
  ${CMAKE_SOURCE_DIR}/test/root/root_geometry.cpp
  ${CMAKE_SOURCE_DIR}/test/root/complex_test1.cpp
  ${CMAKE_SOURCE_DIR}/test/root/ImportFromRootFileTest.cpp
  ${CMAKE_SOURCE_DIR}/test/root/TestExportToROOT.cpp
  # ${CMAKE_SOURCE_DIR}/test/root/ParaboloidTest.cpp
  ${CMAKE_SOURCE_DIR}/test/root/trd_validation.cpp
)
set(TEST_EXECUTABLES_VISUALIZATION
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeTrap.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeTube.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/DebugTube.cpp
)
set(TEST_EXECUTABLES_USOLIDS
  ${CMAKE_SOURCE_DIR}/test/usolids/CompatibilityTest.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestBox.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestCons.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTube.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTrd.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTrap.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestPolycone.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestPolyhedra.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestTet.cpp
  #${CMAKE_SOURCE_DIR}/test/unit_tests/TestOrb.cpp
  #${CMAKE_SOURCE_DIR}/test/unit_tests/TestSphere.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestBoxlikeTrapezoid.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/usolidTest.cpp
)
set(TEST_EXECUTABLES_BENCHMARK
  ${CMAKE_SOURCE_DIR}/test/benchmark/BoxBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/ParaboloidBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/ParaboloidScriptBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/MultiBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/ParallelepipedBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/TubeBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/TrapezoidBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/TrapezoidBenchmarkScript.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/OrbBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/SphereBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/TrdBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/ConeBenchmark.cpp
)
set(TEST_EXECUTABLES_CUDA
  # Files go here
)
set(TEST_EXECUTABLES_SHAPETESTER
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testBox.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testOrb.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testCone.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testTrapezoid.cpp
  # ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testParaboloid.cpp
)
set(TEST_EXECUTABLES
  ${TEST_EXECUTABLES_CORE}
  ${TEST_EXECUTABLES_ROOT}
  ${TEST_EXECUTABLES_VISUALIZATION}
  ${TEST_EXECUTABLES_USOLIDS}
  ${TEST_EXECUTABLES_BENCHMARK}
  ${TEST_EXECUTABLES_CUDA}
  ${TEST_EXECUTABLES_SHAPETESTER}
)
set_source_files_properties(
  ${TEST_EXECUTABLES}
  PROPERTIES COMPILE_FLAGS -std=c++11
)

# Build executables

if (CTEST)
  message(STATUS "Testing with CTest enabled.")
  enable_testing()
endif()

set(VECGEOM_EXECUTABLES)

macro(build_executables EXECUTABLES)
  foreach(EXECUTABLE ${EXECUTABLES})
    get_filename_component(TARGET_NAME ${EXECUTABLE} NAME_WE)
    add_executable(${TARGET_NAME} ${EXECUTABLE})
    target_link_libraries(${TARGET_NAME} ${VECGEOM_LIBRARIES}
                          ${VECGEOM_LIBRARIES_EXTERNAL})
    set(VECGEOM_EXECUTABLES ${VECGEOM_EXECUTABLES} ${TARGET_NAME})
  endforeach()
endmacro()

function(add_to_ctest EXECUTABLES)
  foreach(EXECUTABLE ${EXECUTABLES})
    if (CTEST)
      get_filename_component(TARGET_NAME ${EXECUTABLE} NAME_WE)
      add_test(${TARGET_NAME} ${TARGET_NAME})
    endif()
  endforeach()
endfunction()

build_executables("${TEST_EXECUTABLES_CORE}")
add_to_ctest("${TEST_EXECUTABLES_CORE}")
if (BENCHMARK)
  build_executables("${TEST_EXECUTABLES_BENCHMARK}")
endif()
if (ROOT)
  build_executables("${TEST_EXECUTABLES_ROOT}")
  add_to_ctest("${TEST_EXECUTABLES_ROOT}")
  build_executables("${TEST_EXECUTABLES_VISUALIZATION}")
endif()
if (USolids)
  build_executables("${TEST_EXECUTABLES_USOLIDS}")
  build_executables("${TEST_EXECUTABLES_SHAPETESTER}")
  add_to_ctest("${TEST_EXECUTABLES_USOLIDS}")
  add_to_ctest("${TEST_EXECUTABLES_SHAPETESTER}")
endif()
if (CUDA)
  build_executables("${TEST_EXECUTABLES_CUDA}")
  add_to_ctest("${TEST_EXECUTABLES_CUDA}")
endif()

################################################################################

# Installation

export(PACKAGE Vecgeom)

file(RELATIVE_PATH INSTALL_INCLUDE_DIR_RELATIVE
     "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}/.")
file(RELATIVE_PATH INSTALL_LIB_DIR_RELATIVE
     "${INSTALL_CMAKE_DIR}" "${INSTALL_LIB_DIR}/.")
# Build
set(CONF_TYPE "build")
set(CONF_INCLUDE_DIR "${CMAKE_SOURCE_DIR}")
set(CONF_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
set(CONF_LIBRARIES ${VECGEOM_LIBRARIES_EXTERNAL}
    "${CONF_LIBRARY_DIR}/libvecgeom.a")
if (USolids)
  set(CONF_LIBRARIES ${CONF_LIBRARIES}
      "${CONF_LIBRARY_DIR}/USolids/libusolids.a")
endif()
configure_file(VecgeomConfig.cmake.in
               "${PROJECT_BINARY_DIR}/VecgeomConfig.cmake" @ONLY)
# Installation
set(CONF_TYPE "install")
set(CONF_INCLUDE_DIR "\${THIS_DIR}/${INSTALL_INCLUDE_DIR_RELATIVE}")
set(CONF_LIBRARY_DIR "\${THIS_DIR}/${INSTALL_LIB_DIR_RELATIVE}")
set(CONF_LIBRARIES ${VECGEOM_LIBRARIES_EXTERNAL}
    "${CONF_LIBRARY_DIR}/libvecgeom.a")
if (USolids)
  set(CONF_LIBRARIES ${CONF_LIBRARIES} "${CONF_LIBRARY_DIR}/libusolids.a")
endif()
configure_file(VecgeomConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/VecgeomConfig.cmake" @ONLY)
 
# Install the VecgeomConfig.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/VecgeomConfig.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# Install headers and libraries
foreach(FOLDER base benchmarking management navigation volumes backend)
  install(DIRECTORY ${FOLDER} DESTINATION ${INSTALL_INCLUDE_DIR})
endforeach()
install(TARGETS vecgeom DESTINATION ${INSTALL_LIB_DIR})
if (CUDA)
  install(TARGETS vecgeomcuda DESTINATION ${INSTALL_LIB_DIR})
endif()
