################################################################################
# Before run should be exported next variables:
# CMAKE_SOURCE_DIR        // CMake source directory
# CMAKE_BINARY_DIR        // CMake binary directory
# BUILDTYPE               // CMake build type: Debug, Release
# CMAKE_INSTALL_PREFIX    // Installation prefix for CMake (Jenkins trigger)
# LABEL                   // Name of node (Jenkins trigger)
# BACKEND                 // Backend for VecGeom (CUDA/Vc/Scalar/..)
#
# Optional variables:
# CTEST_SITE              // Name of the host computer
# ExtraCMakeOptions       // Additional options

cmake_minimum_required(VERSION 3.8)

################################################################################
# Build name settings
find_program(UNAME NAMES uname)
execute_process(COMMAND ${UNAME} -m OUTPUT_VARIABLE arch OUTPUT_STRIP_TRAILING_WHITESPACE)

if (DEFINED ENV{BACKEND})
  set(CTEST_BUILD_NAME "$ENV{OPTION}-${arch}+$ENV{BACKEND}-$ENV{LABEL}-$ENV{COMPILER}-$ENV{BUILDTYPE}")
else()
  set(CTEST_BUILD_NAME "$ENV{OPTION}-${arch}-$ENV{LABEL}-$ENV{COMPILER}-$ENV{BUILDTYPE}")
endif()

if(NOT "$ENV{gitlabMergedByUser}$ENV{gitlabMergeRequestIid}" STREQUAL "")
  set(CTEST_BUILD_NAME "$ENV{gitlabMergedByUser}#$ENV{gitlabMergeRequestIid}-${CTEST_BUILD_NAME}")
endif()

if(DEFINED ENV{CTEST_SITE})
  set(CTEST_SITE $ENV{CTEST_SITE})
elseif(DEFINED ENV{container} AND DEFINED ENV{NODE_NAME})
  set(CTEST_SITE "$ENV{NODE_NAME}-$ENV{container}")
else()
  find_program(HOSTNAME_CMD NAMES hostname)
  execute_process(COMMAND ${HOSTNAME_CMD} OUTPUT_VARIABLE CTEST_SITE OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

################################################################################
# Build dashboard model setup
if("$ENV{MODE}" STREQUAL "")
  set(CTEST_MODE Experimental)
else()
  set(CTEST_MODE "$ENV{MODE}")
endif()

if(${CTEST_MODE} MATCHES nightly)
  SET(MODEL Nightly)
elseif(${CTEST_MODE} MATCHES continuous)
  SET(MODEL Continuous)
elseif(${CTEST_MODE} MATCHES memory)
  SET(MODEL NightlyMemoryCheck)
else()
  SET(MODEL Experimental)
endif()

################################################################################
# Use multiple CPU cores to build

cmake_host_system_information(RESULT NCORES QUERY NUMBER_OF_PHYSICAL_CORES)

if(NOT DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
  set(ENV{CMAKE_BUILD_PARALLEL_LEVEL} ${NCORES})
endif()

if(NOT DEFINED ENV{CTEST_PARALLEL_LEVEL})
  set(ENV{CTEST_PARALLEL_LEVEL} ${NCORES})
endif()

################################################################################
# CTest/CMake settings
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS "1000")
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS "1000")
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE "50000")
set(CTEST_TEST_TIMEOUT 900)
set(CTEST_BUILD_CONFIGURATION "$ENV{BUILDTYPE}")
set(CMAKE_INSTALL_PREFIX "$ENV{CMAKE_INSTALL_PREFIX}")
set(CTEST_SOURCE_DIRECTORY "$ENV{CMAKE_SOURCE_DIR}")
set(CTEST_BINARY_DIRECTORY "$ENV{CMAKE_BINARY_DIR}")
set(CTEST_INSTALL_PREFIX "$ENV{CMAKE_INSTALL_PREFIX}")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
################################################################################
# Fixed set of CMake options
set(config_options -DCMAKE_INSTALL_PREFIX=${CTEST_INSTALL_PREFIX}
                   -DBUILDTYPE=${CTEST_BUILD_CONFIGURATION}
                   -DBUILD_TESTING=ON
                   -DVECGEOM_BUILTIN_VECCORE=OFF
                   -DVECGEOM_TEST_BENCHMARK=ON
                   -DVECGEOM_ROOT=ON
                   -DVECGEOM_CUDA_VOLUME_SPECIALIZATION=OFF
                   -DCMAKE_DISABLE_FIND_PACKAGE_GTest=ON # Ignore cvmfs gtest
                   $ENV{ExtraCMakeOptions})

################################################################################
# Options depending on compiler/label/etc.
if("$ENV{LABEL}" MATCHES cuda)
  list(APPEND config_options -DVECGEOM_ENABLE_CUDA=ON)
endif()

if (DEFINED ENV{BACKEND})
  list(APPEND config_options -DVECGEOM_BACKEND=$ENV{BACKEND})
endif()

if("$ENV{OPTION}" STREQUAL "SPEC")
  list(APPEND config_options -DVECGEOM_NO_SPECIALIZATION=OFF)
elseif("$ENV{OPTION}" STREQUAL "GDML")
  list(APPEND config_options -DVECGEOM_GDML=ON)
elseif("$ENV{OPTION}" STREQUAL "AVX")
  list(APPEND config_options -DVECGEOM_VECTOR=avx)
elseif("$ENV{OPTION}" STREQUAL "SSE3")
  list(APPEND config_options -DVECGEOM_VECTOR=sse3)
endif()

################################################################################
# git command configuration
find_program(CTEST_GIT_COMMAND NAMES git)
if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  set(CTEST_CHECKOUT_COMMAND "${CTEST_GIT_COMMAND} clone https://gitlab.cern.ch/VecGeom/VecGeom.git ${CTEST_SOURCE_DIRECTORY}")
endif()

set(CTEST_GIT_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")

function(vecgeom_set_checkout_if_available commit description)
  if("${commit}" STREQUAL "")
    return()
  endif()

  execute_process(
    COMMAND ${CTEST_GIT_COMMAND} -C ${CTEST_SOURCE_DIRECTORY} cat-file -e "${commit}^{commit}"
    RESULT_VARIABLE checkout_commit_status
    OUTPUT_QUIET
    ERROR_QUIET)

  if(checkout_commit_status EQUAL 0)
    set(CTEST_CHECKOUT_COMMAND "cmake -E chdir ${CTEST_SOURCE_DIRECTORY} ${CTEST_GIT_COMMAND} checkout -f ${commit}"
        PARENT_SCOPE)
  else()
    message(STATUS "Skipping CTest initial checkout of ${description} ${commit}: commit is not available locally")
  endif()
endfunction()

if(NOT "$ENV{GIT_COMMIT}" STREQUAL "")
  set(CTEST_GIT_UPDATE_CUSTOM ${CTEST_GIT_COMMAND} checkout -f $ENV{GIT_COMMIT})

  if("${MODEL}" MATCHES "Nightly|Experimental")
    vecgeom_set_checkout_if_available("$ENV{GIT_PREVIOUS_COMMIT}" "previous Jenkins commit")
  else()
    vecgeom_set_checkout_if_available("$ENV{GIT_COMMIT}" "Jenkins commit")
  endif()
endif()

################################################################################
## Output language
set($ENV{LC_MESSAGES}  "en_EN")

################################################################################
# Print summary information.
foreach(v
    CTEST_SITE
    CTEST_BUILD_NAME
    CTEST_SOURCE_DIRECTORY
    CTEST_BINARY_DIRECTORY
    CTEST_CMAKE_GENERATOR
    CTEST_BUILD_CONFIGURATION
    CTEST_GIT_COMMAND
    CTEST_CONFIGURE_COMMAND
    CTEST_SCRIPT_DIRECTORY
    CTEST_BUILD_FLAGS
  )
  set(vars "${vars}  ${v}=[${${v}}]\n")
endforeach(v)
message("Dashboard script configuration (check if everything is declared correctly):\n${vars}\n")
