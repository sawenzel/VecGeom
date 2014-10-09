#----------------------------------------------------------------------------
# Setup the project
#

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(USolids)

option(USOLIDS_USE_UVECTOR3
       "Use the UVector3 implementation instead of Vector3D." OFF)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

set(VECGEOM_DIR ${PROJECT_SOURCE_DIR}/..)

include_directories(
	${PROJECT_SOURCE_DIR}/include 
)
include_directories(
  ${VECGEOM_DIR}
)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

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
  ${VECGEOM_DIR}/base/Container3D.h
  ${VECGEOM_DIR}/base/Global.h
  ${VECGEOM_DIR}/base/AOS3D.h
  ${VECGEOM_DIR}/base/SOA3D.h
  ${VECGEOM_DIR}/base/Transformation3D.h
  ${VECGEOM_DIR}/base/Vector.h # Used in LogicalVolume
  ${VECGEOM_DIR}/base/Vector3D.h
  ${VECGEOM_DIR}/management/GeoManager.h # Volumes register themselves
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

source_group ("Header Files" FILES ${headers})
source_group ("Source Files" FILES ${sources})

if (NOT VECGEOM)
  set(sources ${sources} ${sources_vecgeom})
endif()

set_source_files_properties(
  ${sources}
  PROPERTIES COMPILE_FLAGS -std=c++11
)

add_library(usolids ${sources} ${headers})

install(FILES ${headers} DESTINATION include)
foreach(header ${headers_vecgeom})
  file(RELATIVE_PATH header_relative ${VECGEOM_DIR} ${header})
  get_filename_component(header_relative_folder ${header_relative} PATH)
  install(FILES ${header} DESTINATION include/${header_relative_folder})
endforeach()
install(TARGETS usolids DESTINATION lib)