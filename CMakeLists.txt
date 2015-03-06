
cmake_minimum_required(VERSION 2.8.8)
project(VecGeom)

set(VECGEOM ON)

################################################################################

# Configuration options

option(VC_ACCELERATION "Internal vectorization using Vc." ON)

option(CUDA "Enable compilation for CUDA." OFF)

option(CUDA_VOLUME_SPECIALIZATION "Use specialized volumes for CUDA." ON)

option(ROOT "Include ROOT." OFF)

option(USOLIDS "Include USolids." OFF)

option(GEANT4 "Include Geant4." OFF)

option(BENCHMARK "Enable performance comparisons." OFF)

option(NO_SPECIALIZATION "Disable specialization of volumes." OFF)

option(COVERAGE_TESTING "Enable coverage testing flags." OFF)

option(CTEST "Enable CTest when building." ON)

option(PLANESHELL "Enable the use of PlaneShell class for the trapezoid." ON)

option(DISTANCE_DEBUG "Enable comparision of calculcated distances againt ROOT/Geant4 behind the scenes" OFF)

option(DATA_DOWNLOAD "Enable downloading of data for tests" OFF)

option(OFFLOAD_MODE "Enable offload mode for Intel Xeon Phi." OFF) 

option(VTUNE "Enable the use of profiler Vtune." OFF)

set(VECGEOM_VECTOR "" CACHE STRING
    "Vector instruction set to use (sse/avx/...).")

set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries.")
set(INSTALL_INCLUDE_DIR include CACHE PATH
    "Installation directory for header files.")
set(INSTALL_CMAKE_DIR lib/CMake/VecGeom CACHE PATH
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

if (CTEST)
  set (DATA_DOWNLOAD ON)
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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

# Compiler specific optimization flags
if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
set(VECGEOM_COMPILATION_FLAGS_GNU
    "${VECGEOM_COMPILATION_FLAGS_GNU} -finline-limit=10000000")
set(VECGEOM_COMPILATION_FLAGS_GNU
    "${VECGEOM_COMPILATION_FLAGS_GNU} -ffast-math -ftree-vectorize")
set(VECGEOM_COMPILATION_FLAGS_CLANG
    "${VECGEOM_COMPILATION_FLAGS_CLANG} -ffast-math -ftree-vectorize")
endif()
set(VECGEOM_COMPILATION_FLAGS_INTEL
    "${VECGEOM_COMPILATION_FLAGS_INTEL} ")
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
  if (CUDA_VOLUME_SPECIALIZATION)
    set(VECGEOM_COMPILATION_FLAGS
        "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_CUDA_VOLUME_SPECIALIZATION")
  else()
    set(VECGEOM_COMPILATION_FLAGS
        "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_CUDA_NO_VOLUME_SPECIALIZATION")
  endif()
  set(VECGEOM_NVCC_COMPILATION_FLAGS ${VECGEOM_NVCC_COMPILATION_FLAGS}
      -std=c++11
      -Xcudafe "--diag_suppress=code_is_unreachable"
      -Xcudafe "--diag_suppress=initialization_not_reachable")
  set(VECGEOM_NVCC_CONFIGS
        DEBUG -g -G
        RELEASE  -O2 --use_fast_math
        RELWITHDEBINFO -g -G  -O2 --use_fast_math
        MINSIZEREL --use_fast_math )
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
if (DISTANCE_DEBUG)
  set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_DISTANCE_DEBUG")
endif()
if (OFFLOAD_MODE)
  if (NOT Intel)
    message(FATAL_ERROR "Must use Intel C++ compiler (icc) for Xeon Phi Offload Mode.")
  endif()
  set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -DOFFLOAD_MODE")
  message(STATUS "Preparing for a offload compiler mode.")
  set(VTUNE OFF)
endif()
if (VTUNE)
  set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -DVECGEOM_VTUNE")
  message(STATUS "Prepating for Intel VTune profiling.")
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
#set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -O2")


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

  set(SRC_EXTERNAL ${SRC_EXTERNAL} source/CudaManager.cpp source/backend/cuda/Interface.cpp)
  set(SRC_CUDA ${SRC_CUDA}
      source/CudaManager.cu)

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${VECGEOM_NVCC_COMPILATION_FLAGS})
endif()

if (ROOT)

  find_package(ROOT REQUIRED)

  include_directories(AFTER SYSTEM ${ROOT_INCLUDE_DIR})
  link_directories(${ROOT_LIBRARY_DIR})
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL}
      ${ROOT_LIBRARIES} Geom)
  set(SRC_EXTERNAL ${SRC_EXTERNAL}
      source/PlacedRootVolume.cpp
      source/RootGeoManager.cpp
      source/UnplacedRootVolume.cpp
      source/Visualizer.cpp)

endif()

if (USolids)

  if (NOT USOLIDS_DIR AND (NOT USOLIDS_INCLUDE_DIR OR NOT USOLIDS_LIBRARY_DIR))
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

if (VTUNE)
  find_package(VTUNE REQUIRED)
  include_directories(AFTER SYSTEM ${VTUNE_INCLUDE_DIR})
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} ${VTUNE_LIBRARIES} -lpthread -ldl)
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")

message(STATUS "Compiling with C++ flags: ${CMAKE_CXX_FLAGS}")
if (CUDA)
  message(STATUS "Compiling with NVCC flags: ${CUDA_NVCC_FLAGS}")
endif()


# until VecCore is separated from VecGeom (then it become something
# ${VecCore_DIR}/include
include_directories(${CMAKE_SOURCE_DIR}/VecCore/inc )
include_directories(${CMAKE_INSTALL_PREFIX}/VecCore)

################################################################################

include_directories(${CMAKE_SOURCE_DIR})

set(SRC_CPP_RELATIVE
  source/AOS3D.cpp
  source/LogicalVolume.cpp
  source/PlacedCone.cpp
  source/PlacedBox.cpp
  source/PlacedTube.cpp
  source/PlacedTorus.cpp
  source/PlacedTrd.cpp
  source/PlacedParallelepiped.cpp
  source/PlacedParaboloid.cpp
  source/PlacedPolyhedron.cpp
  source/PlacedTrapezoid.cpp
  source/PlacedVolume.cpp
  source/Planes.cpp
  source/Quadrilaterals.cpp
  source/Rectangles.cpp
  source/SOA3D.cpp
  source/Transformation3D.cpp
  source/UnplacedCone.cpp
  source/UnplacedBox.cpp
  source/UnplacedTube.cpp
  source/UnplacedTorus.cpp
  source/UnplacedTrd.cpp
  source/UnplacedParaboloid.cpp
  source/UnplacedParallelepiped.cpp
  source/UnplacedPolyhedron.cpp
  source/UnplacedTrapezoid.cpp
  source/UnplacedVolume.cpp
  source/Vector.cpp
  source/NavigationState.cpp
  source/SimpleNavigator.cpp

  source/UnplacedOrb.cpp
  source/PlacedOrb.cpp

  source/UnplacedSphere.cpp
  source/PlacedSphere.cpp

  source/UnplacedBooleanVolume.cpp
  source/PlacedBooleanVolume.cpp

  source/UnplacedPolycone.cpp
  source/PlacedPolycone.cpp
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

# some CUDA kernel code in "userspace"
if (CUDA)
    set(SRC_USER_CUDA_CPP
        userexamples/src/TestNavigationStatePool.cu
    )
endif()


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

# build the CUDA version of the library
if (CUDA)
  cuda_add_library(
    vecgeomcuda
    ${SRC_CUDA}
    OPTIONS ${CUDA_ARCH}
    ${VECGEOM_NVCC_CONFIGS}
  )
  target_link_libraries(vecgeomcuda ${VECGEOM_LIBRARIES}
                        ${VECGEOM_LIBRARIES_EXTERNAL})
  set(VECGEOM_LIBRARIES ${VECGEOM_LIBRARIES} vecgeomcuda)
endif()


# build the CUDA user-code library
if (CUDA)
  cuda_add_library(
    cudauserlib
    ${SRC_USER_CUDA_CPP}
    OPTIONS ${CUDA_ARCH}
  )
  target_link_libraries(cudauserlib ${VECGEOM_LIBRARIES} ${VECGEOM_LIBRARIES_EXTERNAL})
  set(USERKERNELLIB cudauserlib)
endif()
################################################################################

# Define executables

set(TEST_EXECUTABLES_CORE 
  ${CMAKE_SOURCE_DIR}/test/core/ContainerTest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/create_geometry.cpp
  ${CMAKE_SOURCE_DIR}/test/core/testVectorSafety.cpp
  ${CMAKE_SOURCE_DIR}/test/core/PlanesTest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/QuadrilateralTest.cpp
#  ${CMAKE_SOURCE_DIR}/test/core/SOATest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/Transformation3DTest.cpp
#  ${CMAKE_SOURCE_DIR}/test/core/boolminustest.cpp
#  ${CMAKE_SOURCE_DIR}/test/core/boolminustest2.cpp
  ${CMAKE_SOURCE_DIR}/test/core/PhiWedgeTest.cpp
  ${CMAKE_SOURCE_DIR}/test/core/BitSetTest.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestVecGeomPolycone.cpp
)


set(TEST_EXECUTABLES_ROOT
  ${CMAKE_SOURCE_DIR}/test/core/TestNavigationStatePool.cpp
  ${CMAKE_SOURCE_DIR}/test/root/root_geometry.cpp
  ${CMAKE_SOURCE_DIR}/test/root/complex_test1.cpp
  ${CMAKE_SOURCE_DIR}/test/root/E03Test.cpp
  ${CMAKE_SOURCE_DIR}/test/root/ImportFromRootFileTest.cpp
  ${CMAKE_SOURCE_DIR}/test/root/ImportTGeoPgon.cpp
  ${CMAKE_SOURCE_DIR}/test/root/TestExportToROOT.cpp
  # ${CMAKE_SOURCE_DIR}/test/root/ParaboloidTest.cpp
  ${CMAKE_SOURCE_DIR}/test/root/trd_validation.cpp

  # higher level benchmarks
  # ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/LocatePointsBenchmark.cpp
)
if(BENCHMARK)
  set(TEST_EXECUTABLES_ROOT
  ${TEST_EXECUTABLES_ROOT}
  ${CMAKE_SOURCE_DIR}/test/root/BenchmarkShapeFromROOTFile.cpp
  ${CMAKE_SOURCE_DIR}/test/root/BenchmarkShapeFromROOTFile_WithVisualization.cpp
)
endif()

set(TEST_EXECUTABLES_VISUALIZATION
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeTrap.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/VisualizeTube.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/DebugTube.cpp
  ${CMAKE_SOURCE_DIR}/test/visualization/DebugPolyhedron.cpp
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
  ${CMAKE_SOURCE_DIR}/test/benchmark/ParallelepipedBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/PolyhedronBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/TubeBenchmark.cpp
 # ${CMAKE_SOURCE_DIR}/test/benchmark/BoxMinusHollowTubeBenchmark.cpp
 # ${CMAKE_SOURCE_DIR}/test/benchmark/BoxMinusTubeSegBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/BoxMinusHollowTubeBenchmark_virtual.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/BoxUnionHollowTubeBenchmark_virtual.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/BoxIntersectHollowTubeBenchmark_virtual.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/BoxUnionBoxBenchmark_virtual.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/3LevelSubtractionBenchmark_virtual.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/TorusBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/TrapezoidBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/TrapezoidBenchmarkScript.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/OrbBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/SphereBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/TrdBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/benchmark/ConeBenchmark.cpp

  ${CMAKE_SOURCE_DIR}/test/benchmark/PolyconeBenchmark.cpp
)
set(TEST_EXECUTABLES_CUDA
  # Files go here
)
set(TEST_EXECUTABLES_SHAPETESTER
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testBox.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testOrb.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testCone.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testTube.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testTrd.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testTrapezoid.cpp
  # ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testParaboloid.cpp
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testPolycone.cpp 
  # ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testFromROOTFile.cpp 
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
                          ${VECGEOM_LIBRARIES_EXTERNAL}
                          ${USERKERNELLIB})
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
  add_to_ctest("${TEST_EXECUTABLES_BENCHMARK}")
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
#---------------------------------------------------------------------------
# Include creation and submission to dashboard cdash.cern.ch

include(CTest)

#---------------------------------------------------------------------------
# Set data to be dowloaded

message(STATUS "Downloading data files")
if(DATA_DOWNLOAD)
 set(DATA_URL "http://mnovak.web.cern.ch/mnovak/data/ExN03.root")
 set(EXPECTED_MD5SUM "b6b0cfdd5035117171bfe1b7f1f40c3f")

 function(DATA_DOWNLOAD)
    set(DATA_FILE ExN03.root)
    if(APPLE)
      execute_process(COMMAND curl -o ${DATA_FILE} ${DATA_URL})
    else()  
      execute_process(COMMAND wget -q ${DATA_URL} -O ${DATA_FILE})
    endif()
 endfunction(DATA_DOWNLOAD)

 message(STATUS "Searching for already downloaded data file(s)")
 find_file(DATA_FILE ExN03.root ${CMAKE_BINARY_DIR})
 if(DATA_FILE STREQUAL "DATA_FILE-NOTFOUND")
    message("File not found: ExN03.root")
    DATA_DOWNLOAD()
 else()
    message("File found:" ${DATA_FILE})
    if(APPLE)
      execute_process(COMMAND md5 ${DATA_FILE} OUTPUT_VARIABLE MD5SUM)
      string(LENGTH ${MD5SUM} MD5LENGTH)
      MATH(EXPR START "${MD5LENGTH} - 33")
      string(SUBSTRING ${MD5SUM} ${START} 32 MD5SUM)         
    else()
  execute_process(COMMAND md5sum ${DATA_FILE} OUTPUT_VARIABLE MD5SUM)
      string(SUBSTRING ${MD5SUM} 0 32 MD5SUM)
    endif()
    if(MD5SUM STREQUAL EXPECTED_MD5SUM)
        message("MD5sum of file is matching")
    else()
        message("MD5sum of file is not matching " ${EXPECTED_MD5SUM})
        DATA_DOWNLOAD()
    endif()
 endif()
endif()


################################################################################

# Installation

export(PACKAGE VecGeom)

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
configure_file(VecGeomConfig.cmake.in
               "${PROJECT_BINARY_DIR}/VecGeomConfig.cmake" @ONLY)
# Installation
set(CONF_TYPE "install")
set(CONF_INCLUDE_DIR "\${THIS_DIR}/${INSTALL_INCLUDE_DIR_RELATIVE}")
set(CONF_LIBRARY_DIR "\${THIS_DIR}/${INSTALL_LIB_DIR_RELATIVE}")
set(CONF_LIBRARIES ${VECGEOM_LIBRARIES_EXTERNAL}
    "${CONF_LIBRARY_DIR}/libvecgeom.a")
if (USolids)
  set(CONF_LIBRARIES ${CONF_LIBRARIES} "${CONF_LIBRARY_DIR}/libusolids.a")
endif()
configure_file(VecGeomConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/VecGeomConfig.cmake" @ONLY)

# Install the VecGeomConfig.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/VecGeomConfig.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# Install headers and libraries
foreach(FOLDER base benchmarking management navigation volumes backend)
  install(DIRECTORY ${FOLDER} DESTINATION ${INSTALL_INCLUDE_DIR})
endforeach()
install(TARGETS vecgeom DESTINATION ${INSTALL_LIB_DIR})
if (CUDA)
  install(TARGETS vecgeomcuda DESTINATION ${INSTALL_LIB_DIR})
endif()


# Not yet a proper project.
add_subdirectory (VecCore)
