
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

option(DISTANCE_DEBUG "Enable comparison of calculated distances againt ROOT/Geant4 behind the scenes" OFF)

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
#  if (BENCHMARK)
#    set(Geant4 ON)
#  else()
#    message(WARNING
#      "Enabling Geant4 without benchmarks has no effect and will be disabled.")
#    set(GEANT4 OFF)
#    set(Geant4 OFF)
#  endif()
   set(Geant4 ON)
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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
  if (APPLE OR UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
  endif()
  set(VECGEOM_NVCC_COMPILATION_FLAGS -g -G ${VECGEOM_NVCC_COMPILATION_FLAGS})
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  #set(VECGEOM_NVCC_COMPILATION_FLAGS -O3 ${VECGEOM_NVCC_COMPILATION_FLAGS})  # remove -use_fast_math
endif()

if (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(VECGEOM_NVCC_COMPILATION_FLAGS -g -G ${VECGEOM_NVCC_COMPILATION_FLAGS})
endif()
if (CMAKE_BUILD_TYPE MATCHES MinSizeRel)
  set(VECGEOM_NVCC_COMPILATION_FLAGS -use_fast_math ${VECGEOM_NVCC_COMPILATION_FLAGS})
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
    "${VECGEOM_COMPILATION_FLAGS_INTEL} -Wall -fPIC -diag-disable 3438 -O2 -fno-alias")
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
        RELEASE  -O3 -use_fast_math
        RELWITHDEBINFO -g -G  -O3 -use_fast_math
        MINSIZEREL -use_fast_math )
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
#set(VECGEOM_COMPILATION_FLAGS "${VECGEOM_COMPILATION_FLAGS} -O3")


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

if (OFFLOAD_MODE)
  set(SRC_EXTERNAL ${SRC_EXTERNAL} source/backend/mic/Backend.cpp)
  # This must be changed to an automatic way...
  set(MICVEC "/opt/intel/composer_xe_2015.2.164/compiler/include/mic")
  include_directories(${MICVEC})
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
  message(STATUS "Compiling with NVCC flags: ${CUDA_NVCC_FLAGS}")
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

  if (NOT USOLIDS_DIR AND (NOT USOLIDS_INCLUDE_DIRS OR NOT USOLIDS_LIBRARIES_DIR))
    message(STATUS "Building with shipped USolids.")
    add_subdirectory(USolids)
    include_directories(${CMAKE_SOURCE_DIR}/USolids/include)
    link_directories(${CMAKE_BINARY_DIR}/USolids)
  endif()
  if (USOLIDS_INCLUDE_DIRS)
    include_directories(${USOLIDS_INCLUDE_DIRS})
  elseif (USOLIDS_DIR)
    include_directories(${USOLIDS_DIR}/include)
  endif()
  if (USOLIDS_LIBRARIES_DIR)
    link_directories(${USOLIDS_LIBRARIES_DIR})
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
  source/RNG.cpp
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
  source/Wedge.cpp
  source/ABBoxManager.cpp

  source/USolidsInterfaceHelper.cpp
)
foreach(SRC ${SRC_CPP_RELATIVE})
  set(SRC_CPP ${SRC_CPP} ${CMAKE_SOURCE_DIR}/${SRC})
endforeach()

# Copy all source files to .cu-files in order for NVCC to compile them as CUDA
# code and not regular C++ files.

if (CUDA)

  foreach(SRC_FILE ${SRC_CPP})

    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME_WE)

    if (NOT "${SRC_FILENAME}" STREQUAL "ABBoxManager") 
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
    endif()
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
    source/benchmarking/NavigationBenchmarker.cu
  )
endif()
set(SRC_CPP ${SRC_CPP} ${SRC_EXTERNAL}
  source/GeoManager.cpp
  source/ShapeDebugger.cpp
  source/benchmarking/NavigationBenchmarker.cpp
)
if (NOT OFFLOAD_MODE)
  set(SRC_CPP ${SRC_CPP} ${SRC_EXTERNAL}
    source/CppExporter.cpp
  )
endif()

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
  ${CMAKE_SOURCE_DIR}/test/root/ExitingOrEntering.cpp
  # higher level benchmarks
  ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/LocatePointsBenchmark.cpp
  ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/XRayBenchmarkFromROOTFile.cpp
  ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/TraceTrack.cpp
  ${CMAKE_SOURCE_DIR}/test/globalbenchmarks/NavigationBenchmark.cpp
)
# separate list for ROOT UNIT tests
set(TEST_UNITTESTEXECUTABLES_ROOT
  ${CMAKE_SOURCE_DIR}/test/root/complex_test1.cpp
  ${CMAKE_SOURCE_DIR}/test/root/E03Test.cpp
  ${CMAKE_SOURCE_DIR}/test/root/TestExportToROOT.cpp
  ${CMAKE_SOURCE_DIR}/test/root/ImportTGeoPgon.cpp
  ${CMAKE_SOURCE_DIR}/test/core/TestNavigationStatePool.cpp
)


if(BENCHMARK)
  set(TEST_EXECUTABLES_ROOT
  ${TEST_EXECUTABLES_ROOT}
  ${CMAKE_SOURCE_DIR}/test/root/BenchmarkShapeFromROOTFile.cpp
  ${CMAKE_SOURCE_DIR}/test/root/BenchmarkShapeFromROOTFile_WithVisualization.cpp
  ${CMAKE_SOURCE_DIR}/test/root/CompareDistances.cpp
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
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestOrb.cpp
  ${CMAKE_SOURCE_DIR}/test/unit_tests/TestSphere.cpp
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
)

if(ROOT) # add shape testers that depend on ROOT
  set(TEST_EXECUTABLES_SHAPETESTER
  ${TEST_EXECUTABLES_SHAPETESTER}
  ${CMAKE_SOURCE_DIR}/test/shape_tester/shape_testFromROOTFile.cpp )
endif()

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
endif()
if (ROOT)
  build_executables("${TEST_EXECUTABLES_ROOT}")
  build_executables("${TEST_EXECUTABLES_VISUALIZATION}")
  add_to_ctest("${TEST_UNITTESTEXECUTABLES_ROOT}")
endif()
if (USolids)
  build_executables("${TEST_EXECUTABLES_USOLIDS}")
  build_executables("${TEST_EXECUTABLES_SHAPETESTER}")
  #add_to_ctest("${TEST_EXECUTABLES_USOLIDS}")
  #add_to_ctest("${TEST_EXECUTABLES_SHAPETESTER}")

  # add unit tests
  add_test(NAME USolidBox COMMAND TestBox --usolids)
  add_test(NAME VecGeomBox COMMAND TestBox --vecgeom)
  add_test(NAME USolidTube COMMAND TestTube --usolids)
  add_test(NAME VecTube COMMAND TestTube --vecgeom)

endif()
if (CUDA)
  build_executables("${TEST_EXECUTABLES_CUDA}")
#  add_to_ctest("${TEST_EXECUTABLES_CUDA}")
endif()

macro(add_cmsshapevalidation_test TESTNAME SHAPEFILE)
  add_test(NAME ${TESTNAME} COMMAND bash -c "${CMAKE_SOURCE_DIR}/test/scripts/RunRandomValidation ${CMAKE_BINARY_DIR}/BenchmarkShapeFromROOTFile  ${CMAKE_SOURCE_DIR}/test/cmstestdata/cms2015.root  ${CMAKE_SOURCE_DIR}/test/cmstestdata/${SHAPEFILE}" )
endmacro()

# Adding various shapes tests for nightlies
add_cmsshapevalidation_test( tubevalidation cmstubes.txt )
add_cmsshapevalidation_test( trapvalidation cmstraps.txt )
add_cmsshapevalidation_test( polyconevalidation cmspolycones.txt )
add_cmsshapevalidation_test( polyhedravalidation cmspolyhedra.txt )
add_cmsshapevalidation_test( conevalidation cmscones.txt )
add_cmsshapevalidation_test( boxvalidation cmsboxes.txt )
#taken out due to problems: add_cmsshapevalidation_test( booleanvalidation cmsbooleans.txt )
#taken out due to Issue-133: add_cmsshapevalidation_test( torusvalidation cmstori.txt )



#---------------------------------------------------------------------------
# Include creation and submission to dashboard cdash.cern.ch

include(CTest)

#---------------------------------------------------------------------------
# Set data to be dowloaded

if(DATA_DOWNLOAD)
  message(STATUS "Downloading data files")

  # define a function downloading from a URL into a local file LOCALFILE
  function(FILE_DOWNLOAD FILE_URL LOCALFILE ) 
   if(APPLE)
       execute_process(COMMAND curl -o ${LOCALFILE} ${FILE_URL})
   else()  
      execute_process(COMMAND wget -q ${FILE_URL} -O ${LOCALFILE})
   endif()
  endfunction(FILE_DOWNLOAD)
  # end of function FILE DOWNLOAD

  # define a function checking md5 hashes
  # result is stored in MD5MATCHES ( 1 == true, 0 == false )
  function(CHECKMD5 FILETOCHECK EXPECTEDMD5HASH MD5MATCHES)
      if(APPLE)
          execute_process(COMMAND md5 ${FILETOCHECK} OUTPUT_VARIABLE MD5SUM)
          string(LENGTH ${MD5SUM} MD5LENGTH)
          MATH(EXPR START "${MD5LENGTH} - 33")
          string(SUBSTRING ${MD5SUM} ${START} 32 MD5SUM)
      else()
          execute_process(COMMAND md5sum ${FILETOCHECK} OUTPUT_VARIABLE MD5SUM)
          string(SUBSTRING ${MD5SUM} 0 32 MD5SUM)
      endif()
      if(MD5SUM STREQUAL EXPECTEDMD5HASH)
        set(${MD5MATCHES} 1 PARENT_SCOPE)
      else()
        set(${MD5MATCHES} 0 PARENT_SCOPE)
      endif()
  endfunction(CHECKMD5)

  # actual function for managing the download
  function(DOWNLOAD_IF_NOT_INSTALLED FILE_URL LOCALFILE TARGETPATH MD5HASH )
    find_file(FOUNDFILE ${LOCALFILE} ${TARGETPATH} )
    if(FOUNDFILE STREQUAL "FOUNDFILE-NOTFOUND")
        # set need download
        message(STATUS "need download of ${LOCALFILE} since not found")
        set( NEEDTODOWNLOAD 1 )
    else()
        # check md5
        message(STATUS "found existing file ${LOCALFILE}")
        CHECKMD5( ${FOUNDFILE} ${MD5HASH} MD5CORRECT )
        if( ${MD5CORRECT} STREQUAL "1" )
            # do not set download flag   
            set( NEEDTODOWNLOAD 0 )
        else( )
            # set need download
            message(STATUS "hash ${MD5HASH} not correct for file ${FOUNDFILE} ${MD5CORRECT}" )
            set( NEEDTODOWNLOAD 1 )
        endif( )
    endif()
  
    if( ${NEEDTODOWNLOAD} STREQUAL 1 )
        message(STATUS " downloading ... ")
        set(DOWNLOADLOCATION "${TARGETPATH}/${LOCALFILE}")
        FILE_DOWNLOAD( ${FILE_URL} ${DOWNLOADLOCATION} )
    else()
        message(STATUS " doing nothing ... ")
    endif()
    # in principle have to check now if download succeeded and has right MD5
    # TOBEDONE
    
    # this is annoying but we have to clear FOUNDFILE SINCE THIS IS TREATED LIKE A STATIC VARIABLE
    unset(FOUNDFILE CACHE)
  endfunction(DOWNLOAD_IF_NOT_INSTALLED)
  
  DOWNLOAD_IF_NOT_INSTALLED( "http://mnovak.web.cern.ch/mnovak/data/cms2015.root" "cms2015.root" 
                             "${CMAKE_SOURCE_DIR}/test/cmstestdata/" "fb3ede867532b372c5e6f7138d00c07e" )
  DOWNLOAD_IF_NOT_INSTALLED( "http://mnovak.web.cern.ch/mnovak/data/ExN03.root" "ExN03.root"
                             "${CMAKE_BINARY_DIR}"                   "b6b0cfdd5035117171bfe1b7f1f40c3f" )
  # add more data to download here
  #  
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
set(CONF_INCLUDE_DIR "\${THIS_DIR}/${INSTALL_INCLUDE_DIR_RELATIVE}" "\${THIS_DIR}/${INSTALL_INCLUDE_DIR_RELATIVE}/VecCore/")
set(CONF_LIBRARY_DIR "\${THIS_DIR}/${INSTALL_LIB_DIR_RELATIVE}")
set(CONF_LIBRARIES ${VECGEOM_LIBRARIES_EXTERNAL}
    "${CONF_LIBRARY_DIR}/libvecgeom.a")
if (CUDA)
   set(CONF_LIBRARIES "${CONF_LIBRARY_DIR}/libvecgeomcuda.a" ${CONF_LIBRARIES}  "${CONF_LIBRARY_DIR}/libvecgeomcuda.a")
endif()
if (USolids)
  set(CONF_LIBRARIES ${CONF_LIBRARIES} "${CONF_LIBRARY_DIR}/libusolids.a")
  configure_file(VecGeomConfig.cmake.in
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/USolidsConfig.cmake" @ONLY)
endif()
configure_file(VecGeomConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/VecGeomConfig.cmake" @ONLY)

# Install the VecGeomConfig.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/VecGeomConfig.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
if (USolids)
  install(FILES
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/USolidsConfig.cmake"
    DESTINATION "${INSTALL_CMAKE_DIR}/../USolids" COMPONENT dev)
endif()

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
