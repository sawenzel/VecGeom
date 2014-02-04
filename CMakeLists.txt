cmake_minimum_required(VERSION 2.8)
project(vecgeom)

# Default to Vc
if (NOT BACKEND)
  set(BACKEND Vc)
endif ()

if (BACKEND STREQUAL "Vc")
  set(Vc TRUE)
endif()
if (BACKEND STREQUAL "CUDA")
  set(CUDA TRUE)
endif()
if (BACKEND STREQUAL "Cilk")
  set(Cilk TRUE)
endif()

message(STATUS "Configuring with backend ${BACKEND}.")

if ((NOT Cilk AND NOT CUDA) AND VECTOR)
  message(STATUS "Compiling with vector instruction set ${VECTOR}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m${VECTOR}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -Wall -ftree-vectorize")
if (CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finline-limit=1000")
endif()

if (Vc)

  find_package(Vc REQUIRED)
  set(LIBS ${LIBS} ${Vc_LIBRARIES})
  get_filename_component(Vc_LIBPATH ${Vc_LIBRARIES} PATH)
  include_directories(${Vc_LIBPATH}/../include/)
  if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=6")
  endif()
  set(SRC_CPP ${SRC_CPP} ${CMAKE_SOURCE_DIR}/source/vc_backend.cpp)
  set(SRC_COMPILETEST ${CMAKE_SOURCE_DIR}/test/compile_vc.cpp)

endif()

if (Cilk)

  set(SRC_CPP ${SRC_CPP} ${CMAKE_SOURCE_DIR}/source/cilk_backend.cpp)
  set(SRC_COMPILETEST ${CMAKE_SOURCE_DIR}/test/compile_cilk.cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost")

else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
endif()

if (CUDA)

  find_package(CUDA REQUIRED)
  include(FindCUDA)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_30 -O2 --use_fast_math)
  set(SRC_COMPILETEST ${CMAKE_SOURCE_DIR}/test/compile_cuda.cu)

endif()

include_directories(${CMAKE_SOURCE_DIR})

if (SRC_CPP)
  add_library(vecgeomcpp STATIC ${SRC_CPP})
  set(LIBS ${LIBS} vecgeomcpp)
endif()

if (NOT CUDA)
  add_executable(compile_test ${CMAKE_SOURCE_DIR}/test/compile_test.cpp)
else()
  cuda_add_library(cudamain ${CMAKE_SOURCE_DIR}/test/compile_test.cpp)
  set(LIBS ${LIBS} cudamain)
  add_executable(compile_test ${CMAKE_SOURCE_DIR}/test/compile_cuda.cu)
endif()

target_link_libraries(compile_test ${LIBS})