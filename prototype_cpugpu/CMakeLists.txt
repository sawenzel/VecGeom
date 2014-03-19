cmake_minimum_required(VERSION 2.8.12)
project(vecgeom)

include_directories(${CMAKE_SOURCE_DIR})
set(SRC_CPP ${CMAKE_SOURCE_DIR}/interface.cpp ${CMAKE_SOURCE_DIR}/testclass.cpp)

find_package(CUDA REQUIRED)
include(FindCUDA)
set(CUDA_SEPARABLE_COMPILATION ON)

foreach(SRC_FILE ${SRC_CPP})

  get_filename_component(SRC_FILENAME ${SRC_FILE} NAME_WE)

  ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu
    COMMAND ${CMAKE_COMMAND} -E copy ${SRC_FILE} ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu
    DEPENDS ${SRC_FILE}
  )

  set(SRC_CUDA ${SRC_CUDA} ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu)

endforeach()

add_library(cpugpu_cpp ${SRC_CPP})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DCPUGPU_CUDA)
cuda_add_executable(cpugpu_final ${CMAKE_SOURCE_DIR}/main.cpp ${SRC_CUDA} OPTIONS -arch=sm_20)
target_link_libraries(cpugpu_final cpugpu_cpp)