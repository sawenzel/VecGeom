
#----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(USolids)

find_package(Geant4 REQUIRED)

set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

#----------------------------------------------------------------------------
# Setup Geant4 include directories and compile definitions
# Setup include directory for this project
#
include(${Geant4_USE_FILE})

add_subdirectory(${PROJECT_SOURCE_DIR}/G4TS G4TS/)
add_subdirectory(${PROJECT_SOURCE_DIR}/bridges/G4 G4Usolids/)
add_subdirectory(${PROJECT_SOURCE_DIR}/bridges/G4/OpticalEscape OpticalEscape/)
add_subdirectory(${PROJECT_SOURCE_DIR}/bridges/G4/SBT SBT/)
add_subdirectory(${PROJECT_SOURCE_DIR}/bridges/TGeo TGeo/)
# add_subdirectory(${PROJECT_SOURCE_DIR}/G4TS/testG4ExtrudedSolid testG4ExtrudedSolid/)
# add_subdirectory(${PROJECT_SOURCE_DIR}/G4TS/testG4Tess_Cad testG4Tess_Cad/)
# add_subdirectory(${PROJECT_SOURCE_DIR}/G4TS/testG4Tess_Mixed testG4Tess_Mixed/)
# add_subdirectory(${PROJECT_SOURCE_DIR}/G4TS/testG4Tess_Quadr testG4Tess_Quadr/)
# add_subdirectory(${PROJECT_SOURCE_DIR}/G4TS/testG4Tess_Tria testG4Tess_Triar/)
# add_subdirectory(${PROJECT_SOURCE_DIR}/G4TS/G01 G01/)

include_directories(
	${PROJECT_SOURCE_DIR}/include 
)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs
#

# ??? how to group header files, *.hh to "Header Files" group

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

source_group ("Header Files" FILES ${headers})
source_group ("Source Files" FILES ${sources})

add_library(usolids  ${sources} ${headers})

#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries
#

#link_directories (${LINK_DIRECTORIES})

target_link_libraries(usolids ${Geant4_LIBRARIES})

#----------------------------------------------------------------------------
# Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
#
###install(TARGETS exampleN01 DESTINATION bin)
