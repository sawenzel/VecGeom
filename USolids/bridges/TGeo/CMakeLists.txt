
#----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(UMultiUnion)
message("Setting up UMultiUnion")

#----------------------------------------------------------------------------
# Find Geant4 package, activating all available UI and Vis drivers by default
# You can set WITH_GEANT4_UIVIS to OFF via the command line or ccmake/cmake-gui
# to build a batch mode only executable
#
find_package(Geant4 REQUIRED)

set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}../../ ${CMAKE_MODULE_PATH})

find_package (ROOT REQUIRED Core RIO Matrix GenVector MathCore Geom Net Thread Graf Graf3d Hist Gpad)


#----------------------------------------------------------------------------
# Setup Geant4 include directories and compile definitions
# Setup include directory for this project
#
include(${Geant4_USE_FILE})

include_directories(
	../../include
	../../
	${ROOT_INCLUDE_DIR}
)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs
#

file(GLOB sources ${PROJECT_SOURCE_DIR}/*.cxx)
file(GLOB headers ${PROJECT_SOURCE_DIR}/*.h)

#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries
#

add_executable(umultiunion ${sources} ${headers})

target_link_libraries(umultiunion usolids ${Geant4_LIBRARIES} ${ROOT_LIBRARIES})
