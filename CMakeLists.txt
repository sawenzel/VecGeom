cmake_minimum_required(VERSION 2.8)
project(vecgeom)

if (NOT BACKEND)
  set(BACKEND Vc)
endif ()

message(STATUS "Configuring with backend ${BACKEND}.")

if (CMAKE_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}-Wall -m64 -O2")
if (NOT (BACKEND STREQUAL "CUDA"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

if (BACKEND STREQUAL Cilk)

  set(COMPILETEST_SRC ${COMPILETEST_SRC} ${CMAKE_SOURCE_DIR}/test/compile_cilk.cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost")

else()

  if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -finline-limit=10000000 -ftree-vectorize")
    if (BACKEND STREQUAL "Vc")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=6")
    endif()
  endif()

  if (VECTOR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m${VECTOR}")
  endif()

endif()

if (BACKEND STREQUAL "Vc")
  find_package(Vc REQUIRED)
  set(LIBS ${LIBS} ${Vc_LIBRARIES})
  get_filename_component(Vc_LIBPATH ${Vc_LIBRARIES} PATH)
  include_directories(${Vc_LIBPATH}/../include/)
  set(COMPILETEST_SRC ${CMAKE_SOURCE_DIR}/test/compile_vc.cpp)
endif()

include_directories(${CMAKE_SOURCE_DIR})

message(STATUS "Will compile with flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Libraries included: ${LIBS}")

add_executable(compile_test ${CMAKE_SOURCE_DIR}/test/compile_test.cpp ${COMPILETEST_SRC})
target_link_libraries(compile_test ${LIBS})