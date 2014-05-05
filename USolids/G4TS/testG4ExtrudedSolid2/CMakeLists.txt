
#----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(SBT)
message("Setting up testG4ExtrudedSolid")

#----------------------------------------------------------------------------
# Find Geant4 package, activating all available UI and Vis drivers by default
# You can set WITH_GEANT4_UIVIS to OFF via the command line or ccmake/cmake-gui
# to build a batch mode only executable
#
find_package(Geant4 REQUIRED)


#----------------------------------------------------------------------------
# Setup Geant4 include directories and compile definitions
# Setup include directory for this project
#
include(${Geant4_USE_FILE})

include_directories(
	../../../..include
	../../include
)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs
#

#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries
#

add_executable(testG4ExtrudedSolid2 testG4ExtrudedSolid2.cc)

target_link_libraries(testG4ExtrudedSolid2 ${Geant4_LIBRARIES})
