# - Enforce an out-of-source builds before anything else
if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
  # The odd formatting of the string is needed to comply with CMake's display rules!
  message(FATAL_ERROR
"VecGeom in-source build detected
 Please remove these files from ${CMAKE_CURRENT_BINARY_DIR}:
  \n  CMakeCache.txt
  CMakeFiles
 \n Once these files are removed, create a separate directory and run CMake from there")
endif()

# - Setup the project
cmake_minimum_required(VERSION 3.16...3.26)

include(cmake/modules/RecordCmdLine.cmake)
include(ExternalProject)

# Set VecGeom_VERSION using git tags or release metadata
include("${CMAKE_CURRENT_LIST_DIR}/cmake/CgvFindVersion.cmake")
cgv_find_version(VecGeom)
message(STATUS "VecGeom version string: ${VecGeom_VERSION_STRING}")

project(VecGeom
  VERSION ${VecGeom_VERSION}
  DESCRIPTION "Vectorized geometry library for particle-detector simulation"
  HOMEPAGE_URL "https://gitlab.cern.ch/VecGeom/VecGeom"
  LANGUAGES C CXX)

################################################################################
# - Core CMake settings
# - Core/Custom modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
include(MacroUtilities)
include(CMakeDependentOption)
include(IntelCompileFeatures)
include(GNUInstallDirs)

# - Though we check for some absolute paths, ensure there are no others
vecgeom_set_default(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION ON)

# - Never export to or search in user/system package registry
vecgeom_set_default(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)
vecgeom_set_default(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON)
vecgeom_set_default(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY ON)

# - Force project directories to appear first in any list of includes
vecgeom_set_default(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)

vecgeom_set_default(BUILD_SHARED_LIBS OFF
  "Build shared libraries instead of static")
vecgeom_set_default(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON
  "Inform installed binaries of external library rpaths")
if(BUILD_SHARED_LIBS)
  # - Inform installed binaries of internal library rpaths
  vecgeom_set_default(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}")
  # - Only relink shared libs when interface changes
  vecgeom_set_default(CMAKE_LINK_DEPENDS_NO_SHARED ON)
endif()

# - Only report newly installed files
vecgeom_set_default(CMAKE_INSTALL_MESSAGE LAZY)

# - Basic library settings
vecgeom_set_default(CMAKE_POSITION_INDEPENDENT_CODE ON
  "Make sure build can be used with shared libs or runtime modules")

################################################################################
# - Core build settings
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
  message(STATUS "Set default CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif()

# - Store CompilerId as a boolean for convenience
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang|Intel)")
  message(WARNING "Unsupported C++ compiler '${CMAKE_CXX_COMPILER_ID}', build may fail.")
endif()

# - C++ standard and extensions
vecgeom_set_default(CMAKE_CXX_STANDARD 17 STRING "C++ standard")
vecgeom_set_default(CMAKE_CXX_EXTENSIONS OFF "Allow C++ compiler extensions")

set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
if(CMAKE_CXX_STANDARD LESS 17)
  message(FATAL_ERROR "VecGeom requires at least C++17")
endif()

# NOTE: to use ccache, user should specify -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

# Install paths
# Add a path for CMake config files (immutable)
set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake")
set(CMAKE_INSTALL_FULL_CMAKEDIR "${CMAKE_INSTALL_FULL_LIBDIR}/cmake")

# Add uninstall target if required
if(NOT TARGET uninstall)
  configure_file(cmake/cmake_uninstall.cmake.in
    "${PROJECT_BINARY_DIR}/cmake_uninstall.cmake"
    @ONLY)

  add_custom_target(uninstall
    COMMAND "${CMAKE_COMMAND}" -P "${PROJECT_BINARY_DIR}/cmake_uninstall.cmake"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")
endif()

################################################################################
# Configuration options
# - ISA
set(VECGEOM_ISAS empty)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(i686|x86_64)")
  set(VECGEOM_ISAS sse2 sse3 ssse3 sse4.1 sse4.2 avx avx2 native empty)
endif()

enum_option(VECGEOM_VECTOR DOC "Vector instruction set to be used"
  TYPE STRING
  VALUES ${VECGEOM_ISAS}
  CASE_INSENSITIVE)

# - ISA Backend
set(VECGEOM_BACKENDS scalar vc)
enum_option(VECGEOM_BACKEND DOC "Vector backend API to be used"
  TYPE STRING
  VALUES ${VECGEOM_BACKENDS}
  CASE_INSENSITIVE)

# Case adjustment because pass to VecCore components is
if(VECGEOM_BACKEND STREQUAL "scalar")
  set(VECGEOM_BACKEND "Scalar")
elseif(VECGEOM_BACKEND STREQUAL "vc")
  set(VECGEOM_BACKEND "Vc")
endif()

# Configure backend for export to VecGeom's Config.h and setting up VecCore
string(TOUPPER "${VECGEOM_BACKEND}" _BACKEND_UP)
set(VECGEOM_${_BACKEND_UP} True)
message(STATUS "Configuring with ${VECGEOM_BACKEND} backend")

if("${VECGEOM_BACKEND}" STREQUAL "Vc")
  set(VecCore_COMPONENTS ${VecCore_COMPONENTS} ${VECGEOM_BACKEND})
endif()

# - Core
option(VECGEOM_BUILTIN_VECCORE "Build VecCore and its dependencies from source" OFF)
# Check/Store an internal variable to check if we switch between builtin/external
# - If we switch from builtin -> external, then we must reset VecCore_DIR later.
# State might have changed...
if(NOT (__BUILTIN_VECCORE_LAST STREQUAL VECGEOM_BUILTIN_VECCORE))
  if(NOT VECGEOM_BUILTIN_VECCORE)
    set(__BUILTIN_VECCORE_DISABLED ON)
  endif()
endif()
set(__BUILTIN_VECCORE_LAST ${VECGEOM_BUILTIN_VECCORE} CACHE INTERNAL "Builtin VecCore marker")

# - Precision
option(VECGEOM_SINGLE_PRECISION "Use single precision floating point type throughout VecGeom" OFF)

option(VECGEOM_ENABLE_CUDA "Enable compilation for CUDA." OFF)
cmake_dependent_option(VECGEOM_CUDA_VOLUME_SPECIALIZATION "Use specialized volumes for CUDA." OFF "VECGEOM_ENABLE_CUDA" OFF)

if(VECGEOM_ENABLE_CUDA)
  enable_language(CUDA)
  # Keep these checks coherent with the C++ compiler/standard
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11)
      message(FATAL_ERROR "CUDA 11 or newer required, found '${CMAKE_CUDA_COMPILER_VERSION}'")
    endif()
  else()
    message(WARNING "Unsupported CUDA compiler '${CMAKE_CUDA_COMPILER_VERSION}', build will likely fail")
  endif()
  set(VecCore_COMPONENTS ${VecCore_COMPONENTS} "CUDA")
endif()

option(VECGEOM_NO_SPECIALIZATION "Disable specialization of volumes." ON)
if((NOT VECGEOM_NO_SPECIALIZATION) AND VECGEOM_ENABLE_CUDA)
  message(STATUS "Forcing VECGEOM_NO_SPECIALIZATION to ON (required by VECGEOM_ENABLE_CUDA=ON)")
  set(VECGEOM_NO_SPECIALIZATION ON CACHE BOOL "Disable specialization of volumes." FORCE)
endif()
option(VECGEOM_PLANESHELL "Enable the use of PlaneShell class for the trapezoid." ON)
option(VECGEOM_QUADRILATERAL_ACCELERATION "Enable SIMD vectorization when looping over quadrilaterals (in Polyhedron)." ON)
option(VECGEOM_DISTANCE_DEBUG "Enable comparison of calculated distances againt ROOT/Geant4 behind the scenes" OFF)
option(VECGEOM_INPLACE_TRANSFORMATIONS "Put transformation as members rather than pointers into PlacedVolume objects" ON)
option(VECGEOM_USE_CACHED_TRANSFORMATIONS "Use cached transformations in navigation states" OFF)
option(VECGEOM_USE_INDEXEDNAVSTATES "Use indices rather than volume pointers in NavigationState objects" ON)

option(VECGEOM_USE_NAVTUPLE "Use navigation index table with tuple states" OFF)
set(VECGEOM_NAVTUPLE_MAXDEPTH "4" CACHE STRING "Maximum depth of navigation tuple states")
option(VECGEOM_USE_SURF "Enable surface model navigation on GPU" ON)
option(VECGEOM_GDML "Enable GDML persistency. Requires Xerces-C" OFF)
option(VECGEOM_GDMLDEBUG "Enable additional debug information in GDML module" OFF)

option(VECGEOM_EMBREE "Enable Intel Embree" OFF)

option(VECGEOM_FAST_MATH "Enable the -ffast-math compiler option in Release builds" OFF)

# Navigation index is REQUIRED for CUDA or SURFACE models
cmake_dependent_option(VECGEOM_USE_NAVINDEX
  "Use navigation index table and states"
  OFF
  "NOT VECGEOM_ENABLE_CUDA;NOT VECGEOM_USE_SURF;NOT VECGEOM_USE_NAVTUPLE"
  ON
)

if(VECGEOM_USE_NAVTUPLE)
  message(STATUS "Enabled navigation state tuple navigation with maximum depth ${VECGEOM_NAVTUPLE_MAXDEPTH}")
  if (VECGEOM_USE_NAVINDEX)
    set(VECGEOM_USE_NAVINDEX OFF CACHE BOOL "Use navigation index table and states" FORCE)
  endif()
else()
  if (VECGEOM_USE_NAVINDEX)
    message(STATUS "Enabled navigation state index navigation")
  endif()
endif()

################################################################################
# - Testing, unit, coverage, benchmarking, validation etc
include(CTest)

# ROOT/Geant4 options are strictly testing only, but the APIs they expose are in use in the wild.
# We therefore retain names pending proper deprecation of the interfaces.
cmake_dependent_option(VECGEOM_ROOT "Enable testing using ROOT" OFF "BUILD_TESTING" OFF)
cmake_dependent_option(VECGEOM_GEANT4 "Enable testing using Geant4" OFF "BUILD_TESTING" OFF)

cmake_dependent_option(VECGEOM_TEST_BENCHMARK "Enable benchmark/performance tests" OFF "BUILD_TESTING" OFF)
cmake_dependent_option(VECGEOM_TEST_VALIDATION "Enable validation tests from CMS geometry" OFF "BUILD_TESTING" OFF)

# Static tests build/use clang-tidy, so require compile commands for VecGeom to be exported
cmake_dependent_option(VECGEOM_TEST_STATIC_ANALYSIS "Enable static analysis tests on VecGeom" OFF "BUILD_TESTING" OFF)
if(VECGEOM_TEST_STATIC_ANALYSIS)
  # static analysis test uses compile commands to run clang-tidy
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# Thought to now be obsolete
cmake_dependent_option(VECGEOM_TEST_VTUNE "Enable use of Intel Vtune for profiling tests" OFF "BUILD_TESTING" OFF)

# TODO: not completely set up. Will build VG+tests with correct flags/links, but no report gen.
cmake_dependent_option(VECGEOM_TEST_COVERAGE "Enable coverage testing flags" OFF "BUILD_TESTING" OFF)


################################################################################
# Minimum version of VecCore we need.
set(VecCore_VERSION "0.8.2")

# Enable/Disable build/use of builtin veccore
if(VECGEOM_BUILTIN_VECCORE)
  include(BuiltinVecCore)
elseif(__BUILTIN_VECCORE_DISABLED)
  # Unset VecCore_DIR if it points inside our build directory
  string(FIND "${VecCore_DIR}" "${PROJECT_BINARY_DIR}" __vcd_is_builtin)
  if(__vcd_is_builtin EQUAL 0)
    unset(VecCore_DIR CACHE)
  endif()
endif()

# Find VecCore with selected components turned on (CUDA and backend)
find_package(VecCore ${VecCore_VERSION} REQUIRED COMPONENTS ${VecCore_COMPONENTS})
set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} VecCore::VecCore)

################################################################################
# - C++ Compiler Flags/Settings for each build type
# Done after all options, as flags are dependent on chosen ISA
# Flags will also be forwarded by CUDA when compiling C++.
set(VECGEOM_ERROR_LIMIT 20 CACHE STRING "Limit number of errors output by compiler")
mark_as_advanced(VECGEOM_ERROR_LIMIT)

if(GNU)
  string(APPEND VECGEOM_CXX_FLAGS " -Wall -fmax-errors=${VECGEOM_ERROR_LIMIT}")

  if(CMAKE_BUILD_TYPE MATCHES Debug)
     string(APPEND VECGEOM_CXX_FLAGS " -ggdb -O0")
  else()
    string(APPEND VECGEOM_CXX_FLAGS " -ftree-vectorize -finline-limit=10000000")
    if(VECGEOM_FAST_MATH)
      string(APPEND VECGEOM_CXX_FLAGS " -ffast-math")
    endif()
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "7")
    string(APPEND VECGEOM_COMPILATION_FLAGS " -faligned-new")
  endif()

elseif(Intel)
  string(APPEND VECGEOM_CXX_FLAGS " -Wall -fmax-errors=${VECGEOM_ERROR_LIMIT}")

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    string(APPEND VECGEOM_CXX_FLAGS " -ggdb -O0")
  else()
    string(APPEND VECGEOM_CXX_FLAGS " -fno-alias")
  endif()

  if(NOT VECGEOM_FAST_MATH)
    string(VECGEOM_CXX_FLAGS " -fp-model precise")
  endif()

elseif(IntelLLVM)
  string(APPEND VECGEOM_CXX_FLAGS " -Wall -fmax-errors=${VECGEOM_ERROR_LIMIT}")

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    string(APPEND VECGEOM_CXX_FLAGS " -ggdb -O0")
  endif()

  if(VECGEOM_FAST_MATH)
    string(APPEND VECGEOM_CXX_FLAGS " -ffast-math")
    string(APPEND VECGEOM_CXX_FLAGS " -Wno-tautological-compare")
  else()
    string(APPEND VECGEOM_CXX_FLAGS " -fno-fast-math")
  endif()

elseif(Clang OR AppleClang)
  string(APPEND VECGEOM_CXX_FLAGS " -Wall -ferror-limit=${VECGEOM_ERROR_LIMIT}")

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    string(APPEND VECGEOM_CXX_FLAGS " -ggdb -O0")
  else()
    string(APPEND VECGEOM_CXX_FLAGS " -ftree-vectorize")
    if(VECGEOM_FAST_MATH)
      string(APPEND VECGEOM_CXX_FLAGS " -ffast-math")
    endif()
  endif()

  if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL "4")
    string(APPEND VECGEOM_COMPILATION_FLAGS " -faligned-new")
  endif()
endif()

# - Additional flags for coverage testing support
if(VECGEOM_TEST_COVERAGE)
  if(GNU)
    string(APPEND VECGEOM_CXX_FLAGS " -fprofile-arcs -ftest-coverage")
    set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} gcov)
  endif()

  if(Clang OR AppleClang)
    string(APPEND VECGEOM_CXX_FLAGS " --coverage")
    set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} profile_rt)
  endif()

  if(Intel OR IntelLLVM)
    message(FATAL_ERROR "Coverage testing not supported for icc.")
  endif()
endif()

# - ISA-specific flags
string(TOLOWER ${VECGEOM_VECTOR} _arch_lo)
string(TOUPPER ${VECGEOM_VECTOR} _arch_up)

if(${_arch_lo} MATCHES native)
  if(Intel OR IntelLLVM)
    string(APPEND VECGEOM_COMPILATION_FLAGS " -march=native")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc|ppc)64le")
    string(APPEND VECGEOM_COMPILATION_FLAGS " -mcpu=${_arch_lo} -mtune=${_arch_lo}")
  else()
    string(APPEND VECGEOM_COMPILATION_FLAGS " -march=${_arch_lo}")
  endif()
elseif(NOT ${_arch_lo} MATCHES empty)
  string(APPEND VECGEOM_COMPILATION_FLAGS " -m${_arch_lo}")
endif()
message(STATUS "Compiling for ${_arch_up} SIMD architecture")

# Combine/finalize flags.
# TODO: Rationalize this or use add_compile_options as appropriate
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECGEOM_CXX_FLAGS} ${VECGEOM_COMPILATION_FLAGS}")


################################################################################
# - CUDA Compiler Flags/Settings for each build type
#
if(VECGEOM_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    vecgeom_set_default(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  endif()

  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  vecgeom_set_default(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

  if(VECGEOM_FAST_MATH)
    string(APPEND CMAKE_CUDA_FLAGS " --use_fast_math")
  endif()

  if(NOT VECGEOM_NO_SPECIALIZATION)
    string(APPEND CMAKE_CUDA_FLAGS " -Xptxas --disable-optimizer-constants")
  endif()

  string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -G")
  string(APPEND CMAKE_CUDA_FLAGS_RELWITHDEBINFO " -line-info")

  # Pre CMake 3.18, we have to manually manage architecture flags, and can only
  # really support Nvidia compiler. Don't handle any other case
  if(CMAKE_VERSION VERSION_LESS "3.18" AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    # Take default using FindCUDA, likely a cleaner way with try_compile
    find_package(CUDA 11 REQUIRED)
    cuda_select_nvcc_arch_flags(_DEFAULT_CUDA_ARCH_FLAGS Auto)
    string(REGEX MATCH "[0-9][0-9]" _DEFAULT_CUDA_ARCH ${_DEFAULT_CUDA_ARCH_FLAGS})

    # Duplicate as far as possible CMake >= 3.18 behaviour
    # NB: does *not* support -real or -virtual extensions
    set(CMAKE_CUDA_ARCHITECTURES "${_DEFAULT_CUDA_ARCH}" CACHE STRING "List of CUDA Architectures")
    string(REPLACE " " ";" CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    list(FILTER CMAKE_CUDA_ARCHITECTURES EXCLUDE REGEX "^ *$")

    # Only apply flags in "NOT non-empty false" case
    if(CMAKE_CUDA_ARCHITECTURES)
      foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
        if(_arch MATCHES "(real|virtual)$")
          message(FATAL_ERROR "real/virtual keyword specifier in CMAKE_CUDA_ARCHITECTURES (argument '${_arch}') requires CMake 3.18 or newer")
        endif()
        # Should be equivalent to old VecGeom behaviour of "-arch=sm_${CUDA_ARCH}", but allowing multiple arches
        string(APPEND CMAKE_CUDA_FLAGS " --generate-code=code=[sm_${_arch},compute_${_arch}],arch=compute_${_arch}")
      endforeach()
    endif()
  endif()
endif()


################################################################################
# - External Packages
#
if(VECGEOM_ROOT)
  # ROOT install may be relying on ROOTSYS
  list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
  find_package(ROOT REQUIRED COMPONENTS Core Geom Graf3d Tree)
endif()

# Intel Embree
if(VECGEOM_EMBREE)
  find_package(embree 3.1 REQUIRED)
  include_directories(${EMBREE_INCLUDE_DIRS})
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} ${EMBREE_LIBRARIES})
endif()

if(VECGEOM_GEANT4)
  find_package(Geant4 REQUIRED)
endif()

if(VECGEOM_TEST_VTUNE)
  find_package(VTUNE REQUIRED)
  include_directories(AFTER SYSTEM ${VTUNE_INCLUDE_DIR})
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} ${VTUNE_LIBRARIES} -lpthread -ldl)
endif()

################################################################################
# - Configure source lists for library(ies)
# VECGEOM_COMMON_SRCS: C++ files compiled by both CXX and CUDA
# VECGEOM_CPPONLY_SRCS: C++ files only compiled for CXX
# VECGEOM_CUDA_SRCS: CUDA files (both from copies of common cpp->cu, and .cu)

# Sources common to C++ and CUDA
set(VECGEOM_COMMON_SRCS
  source/UnplacedPolycone.cpp
  source/UnplacedPolyhedron.cpp
  source/UnplacedTet.cpp
  source/UnplacedTorus2.cpp
  source/UnplacedTube.cpp
  source/UnplacedEllipticalTube.cpp
  source/UnplacedEllipticalCone.cpp
  source/UnplacedEllipsoid.cpp
  source/UnplacedCoaxialCones.cpp
  source/UnplacedGenericPolycone.cpp
  source/UnplacedCone.cpp
  source/UnplacedCutTube.cpp
  source/UnplacedGenTrap.cpp

  source/LogicalVolume.cpp
  source/PlacedPolyhedron.cpp
  source/PlacedPolycone.cpp
  source/PlacedCone.cpp
  source/PlacedAssembly.cpp
  source/PlacedBox.cpp
  source/PlacedSExtru.cpp
  source/PlacedTet.cpp
  source/PlacedHype.cpp
  source/PlacedTube.cpp
  source/PlacedEllipticalTube.cpp
  source/PlacedEllipticalCone.cpp
  source/PlacedEllipsoid.cpp
  source/PlacedCoaxialCones.cpp
  source/PlacedGenericPolycone.cpp
  source/PlacedCutTube.cpp
  source/PlacedTorus2.cpp
  source/PlacedTrd.cpp
  source/PlacedGenTrap.cpp
  source/PlacedParallelepiped.cpp
  source/PlacedParaboloid.cpp
  source/PlacedScaledShape.cpp
  source/PlacedTrapezoid.cpp
  source/PlacedTessellated.cpp
  source/PlacedMultiUnion.cpp
  source/PlacedExtruded.cpp
  source/PlacedVolume.cpp
  source/Planes.cpp
  source/Plane.cpp
  source/CutPlanes.cpp
  source/Quadrilaterals.cpp
  source/TessellatedHelpers.cpp
  source/Tile.cpp
  source/Scale3D.cpp
  source/Transformation3D.cpp
  source/UnplacedAssembly.cpp
  source/UnplacedBox.cpp
  source/UnplacedSExtruVolume.cpp
  source/UnplacedHype.cpp
  source/UnplacedTrd.cpp
  source/UnplacedParaboloid.cpp
  source/UnplacedParallelepiped.cpp
  source/UnplacedScaledShape.cpp
  source/UnplacedTrapezoid.cpp
  source/UnplacedTessellated.cpp
  source/UnplacedMultiUnion.cpp
  source/UnplacedExtruded.cpp
  source/UnplacedVolume.cpp
  source/NavStateIndex.cpp
  source/NavStatePath.cpp
  source/NavIndexTable.cpp

  source/UnplacedOrb.cpp
  source/PlacedOrb.cpp
  source/UnplacedSphere.cpp
  source/PlacedSphere.cpp
  source/UnplacedBooleanVolume.cpp
  source/PlacedBooleanVolume.cpp
  source/Wedge.cpp
  source/Wedge_Evolution.cpp
  source/ABBoxManager.cpp
  source/HybridManager2.cpp
  source/FlatVoxelManager.cpp
  source/BVH.cpp
  source/BVHManager.cpp

  source/NavigationSpecializer.cpp

  source/MarchingCubes.cpp
  source/ResultComparator.cpp
  source/ReducedPolycone.cpp
  source/Utils3D.cpp
  source/SolidMesh.cpp)

set(VECGEOM_CPPONLY_SRCS
  source/GeoManager.cpp
  source/CppExporter.cpp
  source/ReflFactory.cpp

  source/ColorUtils.cpp
  source/Environment.cpp
  source/Logger.cpp
  source/LoggerMessage.cpp
  source/LoggerTypes.cpp
)

if(VECGEOM_EMBREE)
  list(APPEND VECGEOM_COMMON_SRCS source/EmbreeManager.cpp)
endif()

# Copy all source files to .cu-files in order for NVCC to compile them as CUDA
# code and not regular C++ files.
if(VECGEOM_ENABLE_CUDA)
  list(APPEND VECGEOM_COMMON_SRCS
    source/RNG.cpp
    source/AOS3D.cpp
    source/SOA3D.cpp
    source/Vector.cpp)

  list(APPEND VECGEOM_CPPONLY_SRCS source/CudaManager.cpp)

  # Pure CUDA source files, which will be compiled together with the ones generated
  # from the dual-use .cpp files
  set(VECGEOM_CUDA_SRCS
    source/BVHManager.cu
    source/CudaManager.cu
    source/backend/cuda/Interface.cpp
    )

  # Filter file prefixes that won't be compiled for CUDA
  set(NOT_FOR_CUDA
    ABBoxManager
    HybridManager2
    FlatVoxelManager
    Medium
    RootGeoManager
    MarchingCubes
    NavigationSpecializer
    ResultComparator
    UnplacedTessellated
    UnplacedMultiUnion
    PlacedTessellated
    PlacedMultiUnion
    TessellatedCluster
    TessellatedHelpers
    Tile
    UnplacedExtruded
    PlacedExtruded
    SolidMesh)

  foreach(SRC_FILE ${VECGEOM_COMMON_SRCS})
    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME_WE)
    list(FIND NOT_FOR_CUDA ${SRC_FILENAME} _index)

    if(${_index} EQUAL -1)
      set(SRC_FILE_CPP ${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE})
      set(SRC_FILE_CU ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu)

      add_custom_command(
        OUTPUT ${SRC_FILE_CU}
        COMMAND ${CMAKE_COMMAND} -E copy ${SRC_FILE_CPP} ${SRC_FILE_CU}
        DEPENDS ${SRC_FILE_CPP})

      list(APPEND VECGEOM_CUDA_SRCS ${SRC_FILE_CU})
    endif()
  endforeach()
endif()

################################################################################
# Build libraries
if(NOT APPLE)
  set(VECGEOM_LIBRARIES_EXTERNAL ${VECGEOM_LIBRARIES_EXTERNAL} rt pthread)
endif()

# Set up hex version: add an extra 1 up front and strip it to zero-pad
math(EXPR _temp_version
  "((256 + ${PROJECT_VERSION_MAJOR}) * 256 + ${PROJECT_VERSION_MINOR}) * 256 + ${PROJECT_VERSION_PATCH}"
  OUTPUT_FORMAT HEXADECIMAL
)
string(SUBSTRING "${_temp_version}" 3 -1 _temp_version)
set(VECGEOM_HEX_VERSION "0x${_temp_version}")

# - Configure settings and version headers
configure_file(VecGeom/base/Config.h.in ${PROJECT_BINARY_DIR}/VecGeom/base/Config.h @ONLY)
configure_file(VecGeom/base/Version.h.in ${PROJECT_BINARY_DIR}/VecGeom/base/Version.h @ONLY)

add_library(vecgeom
  ${PROJECT_BINARY_DIR}/VecGeom/base/Config.h
  ${PROJECT_BINARY_DIR}/VecGeom/base/Version.h
  ${VECGEOM_COMMON_SRCS}
  ${VECGEOM_CPPONLY_SRCS})
target_compile_features(vecgeom PUBLIC cxx_std_${CMAKE_CXX_STANDARD})

# Add compile options that vecgeom clients must compile their code using Vecgeom
# with.
# 1. We've already added them to CMAKE_CXX_FLAGS above, as they have to be in
#    CMAKE_CXX_FLAGS for the current CUDA build... This means we get some duplication
#    of flags on the command line, but this should not have side effects for the
#    flags that can be used.
# 2. We have to use slightly awkward genexs when CUDA is enabled because VecGeom's
#    CUDA library links to the C++ version. Thus its compile options are used by
#    nvcc and may be unknown to it. CUDA 11 supports the -forward-unknown-to-host-compiler
#    argument which CMake will apply automatically from CMake 3.17. To support
#    CMake 3.16, explicitly forward flags using -Xcompiler=.
string(STRIP "${VECGEOM_COMPILATION_FLAGS}" __VECGEOM_COMPILE_OPTIONS)
separate_arguments(__VECGEOM_COMPILE_OPTIONS)
target_compile_options(vecgeom PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${__VECGEOM_COMPILE_OPTIONS}>)
if(VECGEOM_ENABLE_CUDA AND __VECGEOM_CUDA_COMPILE_OPTIONS)
  string(REPLACE ";" "," __VECGEOM_CUDA_COMPILE_OPTIONS "${__VECGEOM_COMPILE_OPTIONS}")
  target_compile_options(vecgeom PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${__VECGEOM_CUDA_COMPILE_OPTIONS}>)
endif()

target_include_directories(vecgeom PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(vecgeom
  PUBLIC ${VECGEOM_LIBRARIES_EXTERNAL}
  PRIVATE ${CMAKE_DL_LIBS})

if(VECGEOM_ROOT)
  target_link_libraries(vecgeom PUBLIC ROOT::Geom)
endif()

if(VECGEOM_GEANT4)
  target_link_libraries(vecgeom PUBLIC Geant4::G4geometry)
endif()

# build the CUDA version of the library
if(VECGEOM_ENABLE_CUDA)
  set(__vg_cuda_liblist "vecgeomcuda_obj;vecgeomcuda_static;vecgeomcuda")
  # 1. OBJECT library to perform the actual (separable) compilation once
  #    PIC so it can go into a shared lib
  add_library(vecgeomcuda_obj OBJECT ${VECGEOM_CUDA_SRCS})
  set_target_properties(vecgeomcuda_obj
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_SEPARABLE_COMPILATION ON)

  # 2. STATIC library composed of the object library objects
  #    This is the library that consumers should link to unless they
  #    have to use shared libs and know what they are doing
  add_library(vecgeomcuda_static STATIC $<TARGET_OBJECTS:vecgeomcuda_obj>)
  set_target_properties(vecgeomcuda_static
    PROPERTIES
      CUDA_RDC_LIBRARY_TYPE Static
  )

  # 3. SHARED library composed of the object library objects
  #    This is the same library as produced by the old, hacked, FindCUDA module
  #    It is:
  #    - Linked to the CUDA Shared runtime (so clients must also use this)
  #    - Internally device linked so that non-CUDA consumers can, in principle,
  #      link to this without needing to device-compile
  #    - CUDA consumers (i.e. use __device__ code from VecGeom) can still link
  #      to this library but must add `vecgeomcuda_static` to any device link step
  #      for example:
  #
  #      ```cmake
  #      add_library(MyLib MyLib.cu)
  #      target_link_libraries(MyLib vecgeomcuda)
  #      target_link_options(MyLib $<DEVICE_LINK:vecgeomcuda_static>)
  #      ```
  #
  #      It is the consumer's responsibility to hide or forward any device linking
  #      requirements. This is the case if `MyLib` exposes device code.

  # The CUDA_RDC properties are defines and used in the CudaRdcUtils cmake script.
  # See https://github.com/pcanal/CudaRDCExercise for a reference implementation.
  set(__vg_cuda_rdc_props
    CUDA_RDC_FINAL_LIBRARY vecgeomcuda
    CUDA_RDC_MIDDLE_LIBRARY vecgeomcuda_objects (or static)
    CUDA_RDC_STATIC_LIBRARY vecgeomcuda ${_staticsuf}
    CUDA_RDC_OBJECT_LIBRARY vecgeom_obj
  )

  add_library(vecgeomcuda $<TARGET_OBJECTS:vecgeomcuda_obj>)
  if (BUILD_SHARED_LIBS)
    set_target_properties(vecgeomcuda
      PROPERTIES
        CUDA_RDC_LIBRARY_TYPE Final
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Shared)
    # Library without the nvlink steps, this should be used for linking
    # whenever additional cuda might be added.
    list(APPEND __vg_cuda_liblist vecgeomcuda_objects)
    add_library(vecgeomcuda_objects $<TARGET_OBJECTS:vecgeomcuda_obj>)
    set_target_properties(vecgeomcuda_objects
      PROPERTIES
        CUDA_RDC_LIBRARY_TYPE Shared
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
        CUDA_RUNTIME_LIBRARY Shared)
    # Those are currently only used for exporting for use from
    # dependent products.
    set(__vg_cuda_rdc_props
      CUDA_RDC_FINAL_LIBRARY VecGeom::vecgeomcuda
      CUDA_RDC_MIDDLE_LIBRARY VecGeom::vecgeomcuda_objects
      CUDA_RDC_STATIC_LIBRARY VecGeom::vecgeomcuda_static
      CUDA_RDC_OBJECT_LIBRARY vecgeom_obj
    )
  else()
    set_target_properties(vecgeomcuda
      PROPERTIES
        CUDA_RDC_LIBRARY_TYPE Final
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Static)
    set(__vg_cuda_rdc_props
      CUDA_RDC_FINAL_LIBRARY VecGeom::vecgeomcuda
      CUDA_RDC_MIDDLE_LIBRARY VecGeom::vecgeomcuda_static
      CUDA_RDC_STATIC_LIBRARY VecGeom::vecgeomcuda_static
      CUDA_RDC_OBJECT_LIBRARY vecgeom_obj
    )
  endif()

  # Apply usage requirements to OBJECT, STATIC, SHARED targets
  # Required so that OBJECT library compiles correctly and that STATIC/SHARED
  # forward Usage Requirements to their consumers. Can be simplified from CMake 3.21
  # with support for linking OBJECT libraries.
  set_target_properties(vecgeom PROPERTIES ${__vg_cuda_rdc_props}
    EXPORT_PROPERTIES "CUDA_RDC_LIBRARY_TYPE;CUDA_RDC_FINAL_LIBRARY;CUDA_RDC_MIDDLE_LIBRARY;CUDA_RDC_STATIC_LIBRARY"
  )
  foreach(__vg_cuda_target ${__vg_cuda_liblist})
    target_include_directories(${__vg_cuda_target}
      PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_link_libraries(${__vg_cuda_target} PUBLIC VecCore::VecCore vecgeom)
    set_target_properties(${__vg_cuda_target} PROPERTIES ${__vg_cuda_rdc_props}
      EXPORT_PROPERTIES "CUDA_RDC_LIBRARY_TYPE;CUDA_RDC_FINAL_LIBRARY;CUDA_RDC_MIDDLE_LIBRARY;CUDA_RDC_STATIC_LIBRARY"
    )

    # - Suppress stack size warnings from a couple of functions
    # NB: currently identified as needed at device link step, but this only works on
    # CMake 3.18 and newer (for DEVICE_LINK)
    # TODO: find workaround for earlier versions
    # TODO: Check at CUDA compilation step
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18.0)
      target_link_options(${__vg_cuda_target} PUBLIC "$<DEVICE_LINK:SHELL: -Xnvlink --suppress-stack-size-warning>")
    endif()
  endforeach()
endif()

# build the libraries for GDML persistency
if(VECGEOM_GDML)
  add_subdirectory(persistency/gdml)
endif()

################################################################################
# TESTING
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

################################################################################
# Installation
# Install headers and libraries
install(DIRECTORY VecGeom DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERN "*.h.in" EXCLUDE)

install(FILES
    ${PROJECT_BINARY_DIR}/VecGeom/base/Config.h
    ${PROJECT_BINARY_DIR}/VecGeom/base/Version.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/VecGeom/base")

install(TARGETS vecgeom EXPORT VecGeomTargets DESTINATION "${CMAKE_INSTALL_LIBDIR}")

if(VECGEOM_ENABLE_CUDA)
  install(TARGETS vecgeomcuda vecgeomcuda_static EXPORT VecGeomTargets DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  if(BUILD_SHARED_LIBS)
    install(TARGETS vecgeomcuda_objects EXPORT VecGeomTargets DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endif()
endif()

# CMake support files
include(CMakePackageConfigHelpers)

file(RELATIVE_PATH INSTALL_INCLUDE_DIR_RELATIVE
  "${CMAKE_INSTALL_FULL_CMAKEDIR}/VecGeom" "${CMAKE_INSTALL_FULL_INCLUDEDIR}")
file(RELATIVE_PATH INSTALL_LIB_DIR_RELATIVE
  "${CMAKE_INSTALL_FULL_CMAKEDIR}/VecGeom" "${CMAKE_INSTALL_FULL_LIBDIR}")

# Common
write_basic_package_version_file(VecGeomConfigVersion.cmake COMPATIBILITY AnyNewerVersion)

# Build
set(CONF_INCLUDE_DIR "${PROJECT_SOURCE_DIR}")
set(PATH_VARS_ARG PATH_VARS CONF_INCLUDE_DIR)

if(VECGEOM_BUILTIN_VECCORE)
  set(VECCORE_PREFIX "${VecCore_ROOTDIR}")
  list(APPEND PATH_VARS_ARG VECCORE_PREFIX)
endif()

# Export all cache variables that start with VECGEOM_, renaming them to VecGeom_
set(VECGEOM_EXPORT_VARIABLES)
macro(vecgeom_export_var varext)
  list(APPEND VECGEOM_EXPORT_VARIABLES
    "set(VecGeom_${varext} \"${VECGEOM_${varext}}\")"
  )
endmacro()
list(APPEND VECGEOM_EXPORT_VARIABLES "# Configuration options")
get_directory_property(_cachevar_keys CACHE_VARIABLES)
foreach(_key IN LISTS _cachevar_keys)
  if(_key MATCHES "^VECGEOM_(.*)")
    vecgeom_export_var(${CMAKE_MATCH_1})
  endif()
endforeach()

# Add hints for direct dependencies and indirect geant dependencies
list(APPEND VECGEOM_EXPORT_VARIABLES "\n# Hints for upstream dependencies")
foreach(_key
  CUDAToolkit_BIN_DIR VecCore_DIR XercesC_DIR
)
  set(_val "${${_key}}")
  if(_val)
    list(APPEND VECGEOM_EXPORT_VARIABLES
      "if(NOT DEFINED ${_key})"
      "  set(${_key} \"${_val}\")"
      "endif()"
    )
  endif()
endforeach()
list(JOIN VECGEOM_EXPORT_VARIABLES "\n" VECGEOM_EXPORT_VARIABLES)

configure_package_config_file(VecGeomConfig.cmake.in
  "${PROJECT_BINARY_DIR}/VecGeomConfig.cmake"
  ${PATH_VARS_ARG}
  INSTALL_PREFIX "${PROJECT_BINARY_DIR}"
  INSTALL_DESTINATION "${PROJECT_BINARY_DIR}")
export(EXPORT VecGeomTargets NAMESPACE VecGeom::)

# Installation
set(CONF_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
set(PATH_VARS_ARG PATH_VARS CONF_INCLUDE_DIR)

if(VECGEOM_BUILTIN_VECCORE)
  set(VECCORE_PREFIX "${CMAKE_INSTALL_PREFIX}")
  list(APPEND PATH_VARS_ARG VECCORE_PREFIX)
endif()

configure_package_config_file(VecGeomConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/VecGeomConfig.cmake"
  ${PATH_VARS_ARG}
  INSTALL_DESTINATION "${CMAKE_INSTALL_CMAKEDIR}/VecGeom")

# Install the VecGeom{Config,Version,Targets}.cmake
install(
  FILES
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/VecGeomConfig.cmake"
    "${PROJECT_BINARY_DIR}/VecGeomConfigVersion.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/modules/CudaRdcUtils.cmake"
  DESTINATION "${CMAKE_INSTALL_CMAKEDIR}/VecGeom"
)

install(EXPORT VecGeomTargets
  NAMESPACE VecGeom::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VecGeom")


################################################################################
# Doxygen documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
  set(DOXYFILE_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/doxygen)
  foreach(d doc/doxygen VecGeom persistency scripts source)
    if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${d})
      message(SEND_ERROR "Doxygen configured wrongly: The path ${d} doesn't exist in ${PROJECT_SOURCE_DIR}.")
    endif()
    set(DOXYFILE_SOURCE_DIRS "${DOXYFILE_SOURCE_DIRS} ${PROJECT_SOURCE_DIR}/${d}")
  endforeach()

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doxygen
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    COMMENT "Writing documentation to ${DOXYFILE_OUTPUT_DIR}..."
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  add_custom_target(doxydir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DOXYFILE_OUTPUT_DIR}
    COMMENT "Creating doc directory")
  add_dependencies(doxygen doxydir)
endif()
