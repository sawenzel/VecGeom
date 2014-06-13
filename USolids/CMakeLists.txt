#----------------------------------------------------------------------------
# Setup the project
#

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(USolids)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

include_directories(
	${PROJECT_SOURCE_DIR}/include 
)
include_directories(
  ${PROJECT_SOURCE_DIR}/../ 
)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

set_source_files_properties(
  ${sources}
  PROPERTIES COMPILE_FLAGS -std=c++11
)

source_group ("Header Files" FILES ${headers})
source_group ("Source Files" FILES ${sources})

add_library(usolids ${sources} ${headers})

install(FILES ${headers} DESTINATION include)
install(TARGETS usolids DESTINATION lib)