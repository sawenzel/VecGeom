cmake_minimum_required(VERSION 2.8)
project(vector_comparison)

include_directories(${PROJECT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})

if (NOT UNIX)
  message(SEND_ERROR "Non-Unix platforms not supported.")
endif()

find_package(Vc REQUIRED)
include_directories(${Vc_LIBRARIES}/../include/)
set(LIBS ${LIBS} ${Vc_LIBRARIES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -m64 -std=c++11 -O2")
set(SCALAR_OPT -no-vec)
set(VECTOR_OPT -xHost)

set(SCALAR_SRC Scalar.cpp VectorTester.cpp)
set(VECTOR_SRC Cilk.cpp Vc.cpp)
foreach (f ${SCALAR_SRC})
  set_source_files_properties(${f} PROPERTIES LANGUAGE CXX)
  set_source_files_properties(${f} PROPERTIES COMPILE_FLAGS ${SCALAR_OPT})
endforeach(f)
foreach (f ${VECTOR_SRC})
  set_source_files_properties(${f} PROPERTIES LANGUAGE CXX)
  set_source_files_properties(${f} PROPERTIES COMPILE_FLAGS ${VECTOR_OPT})
endforeach(f)

add_executable(Main ${SCALAR_SRC} ${VECTOR_SRC} Main.cpp)
target_link_libraries(Main ${LIBS})