#----------------------------------------------------------------------------
# Setup the project
#

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(USolids)

option(USOLIDS_USE_UVECTOR3
       "Use the UVector3 implementation instead of Vector3D." OFF)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

if (EXISTS ${PROJECT_SOURCE_DIR}/../base/Global.h)
  set(VECGEOM_DIR ${PROJECT_SOURCE_DIR}/..)
else()
  set(VECGEOM_DIR ${PROJECT_SOURCE_DIR})
endif()

include_directories(
	${PROJECT_SOURCE_DIR}/include
)
include_directories(
  ${VECGEOM_DIR}
)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs

file(GLOB sources_usolids ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers_usolids ${PROJECT_SOURCE_DIR}/include/*.hh ${PROJECT_SOURCE_DIR}/include/*.icc)

file(GLOB headers_vecgeom
  ${VECGEOM_DIR}/volumes/*Paraboloid*
  ${VECGEOM_DIR}/volumes/kernel/*Paraboloid*
  ${VECGEOM_DIR}/volumes/*Box* # Used for bounding box
  ${VECGEOM_DIR}/volumes/kernel/*Box*
  ${VECGEOM_DIR}/volumes/kernel/GenericKernels.h
)
set(headers_vecgeom ${headers_vecgeom}
  ${VECGEOM_DIR}/backend/Backend.h
  ${VECGEOM_DIR}/backend/scalar/Backend.h
  ${VECGEOM_DIR}/base/AlignedBase.h
  ${VECGEOM_DIR}/base/Array.h
  ${VECGEOM_DIR}/base/AOS3D.h
  ${VECGEOM_DIR}/base/Container3D.h
  ${VECGEOM_DIR}/base/Global.h
  ${VECGEOM_DIR}/base/RNG.h
  ${VECGEOM_DIR}/base/SOA3D.h
  ${VECGEOM_DIR}/base/SpecializedTransformation3D.h
  ${VECGEOM_DIR}/base/Transformation3D.h
  ${VECGEOM_DIR}/base/Vector.h # Used in LogicalVolume
  ${VECGEOM_DIR}/base/Vector3D.h
  ${VECGEOM_DIR}/management/GeoManager.h # Volumes register themselves
  ${VECGEOM_DIR}/management/VolumeFactory.h
  ${VECGEOM_DIR}/management/TransformationSpecializations.icc
  ${VECGEOM_DIR}/volumes/LogicalVolume.h
  ${VECGEOM_DIR}/volumes/PlacedVolume.h
  ${VECGEOM_DIR}/volumes/ShapeImplementationHelper.h
  ${VECGEOM_DIR}/volumes/UnplacedVolume.h
  ${VECGEOM_DIR}/volumes/USolidsInterfaceHelper.h
)

file(GLOB sources_vecgeom
  ${VECGEOM_DIR}/source/*Paraboloid*
  ${VECGEOM_DIR}/source/*Box*
)
set(
  sources_vecgeom ${sources_vecgeom}
  ${VECGEOM_DIR}/source/AOS3D.cpp
  ${VECGEOM_DIR}/source/GeoManager.cpp
  ${VECGEOM_DIR}/source/LogicalVolume.cpp
  ${VECGEOM_DIR}/source/PlacedVolume.cpp
  ${VECGEOM_DIR}/source/SOA3D.cpp
  ${VECGEOM_DIR}/source/Transformation3D.cpp
  ${VECGEOM_DIR}/source/UnplacedVolume.cpp
)

if (NOT VECGEOM)
  set(headers ${headers_usolids} ${headers_vecgeom})
  set(sources ${sources_usolids} ${sources_vecgeom})
else()
  set(headers ${headers_usolids})
  set(sources ${sources_usolids})
endif()

source_group("Header Files" FILES ${headers})
source_group("Source Files" FILES ${sources})

set_source_files_properties(
  ${sources}
  PROPERTIES COMPILE_FLAGS -std=c++11
)

# For installation of the USolids source. This means copying the files necessary
# to build USolids into the installation directory without moving all vecgeom
# files.
set(source_installation_command
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_INSTALL_PREFIX}/src
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_INSTALL_PREFIX}/include
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/bridges
    ${CMAKE_INSTALL_PREFIX}/bridges
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/G4TS
    ${CMAKE_INSTALL_PREFIX}/G4TS
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/doc
    ${CMAKE_INSTALL_PREFIX}/doc
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_LIST_FILE}
    ${CMAKE_INSTALL_PREFIX})
foreach(source ${sources_vecgeom})
  file(RELATIVE_PATH source_relative ${VECGEOM_DIR} ${source})
  get_filename_component(source_relative_path ${source_relative} PATH)
  set(source_installation_command ${source_installation_command}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${source}
      ${CMAKE_INSTALL_PREFIX}/${source_relative})
endforeach()

add_library(usolids ${sources} ${headers})
install(FILES ${headers_usolids} DESTINATION include)
install(TARGETS usolids DESTINATION lib)
if (NOT VECGEOM)
  foreach(header ${headers_vecgeom})
    file(RELATIVE_PATH header_relative ${VECGEOM_DIR} ${header})
    get_filename_component(header_relative_path ${header_relative} PATH)
    # Also add headers to the source installation list
    set(source_installation_command ${source_installation_command}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header}
        ${CMAKE_INSTALL_PREFIX}/${header_relative})
    install(FILES ${header_relative} DESTINATION
            include/${header_relative_path})
  endforeach()
endif()

macro(install_source_command commands)
  add_custom_target(install_source ${${commands}})
endmacro()

install_source_command(source_installation_command)
