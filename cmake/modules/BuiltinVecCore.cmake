#  This function is used to force a build on a dependant project at cmake configuration phase.
#
function (build_external_project target globpattern ) #FOLLOWING ARGUMENTS are the ARGS of ExternalProject_Add
  set(trigger_build_topdir ${PROJECT_BINARY_DIR}/buildExternals/${target})
  set(trigger_src_dir ${PROJECT_BINARY_DIR}/buildExternals/${target}/trigger_src)
  set(trigger_build_dir ${PROJECT_BINARY_DIR}/buildExternals/${target}/build)

  #mktemp dir in build tree
  file(MAKE_DIRECTORY ${trigger_build_dir} ${trigger_src_dir} ${trigger_build_dir})

  #generate false dependency project
  set(CMAKE_LIST_CONTENT "
      cmake_minimum_required(VERSION 3.16...3.27)
      project(BuiltinVecCore)
      include(ExternalProject)
      ExternalProject_add(${target}
              ${ARGN}
              )

      add_custom_target(trigger_${target})
      add_dependencies(trigger_${target} ${target})
  ")

  file(WRITE ${trigger_src_dir}/CMakeLists.txt "${CMAKE_LIST_CONTENT}")

  execute_process(COMMAND ${CMAKE_COMMAND} ${trigger_src_dir} -G ${CMAKE_GENERATOR}
    WORKING_DIRECTORY ${trigger_build_dir}
  )
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    WORKING_DIRECTORY ${trigger_build_dir}
  )

  FILE(GLOB VecCoreHeaders string(${globpattern}))

  add_custom_target(${target})
  add_custom_command(
          OUTPUT ${VecCoreHeaders}
          COMMAND ${CMAKE_COMMAND} --build .
          WORKING_DIRECTORY ${trigger_build_dir}
  )
endfunction()


set(VecCore_PROJECT "VecCore-${VecCore_VERSION}")
set(VecCore_SRC_URI "http://lcgpackages.web.cern.ch/lcgpackages/tarFiles/sources")
set(VecCore_SRC_MD5 "1d0a4f4f64989b2277905c00ce88f120")  # for VecCore@0.8.2
set(VecCore_DESTDIR "${PROJECT_BINARY_DIR}/installExternals/${VecCore_PROJECT}")
set(VecCore_ROOTDIR "${VecCore_DESTDIR}/${CMAKE_INSTALL_PREFIX}")
set(VecCore_SRC_TAG "v${VecCore_VERSION}")

if(VECGEOM_VC)
  set(Vc_LIBNAME ${CMAKE_STATIC_LIBRARY_PREFIX}Vc${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(Vc_LIBRARY ${VecCore_ROOTDIR}/lib${LIB_SUFFIX}/${Vc_LIBNAME})
endif()

build_external_project(${VecCore_PROJECT}
  "${VecCore_ROOTDIR}/include/VecCore/*.h;${VecCore_ROOTDIR}/include/VecCore/VecCore;${VecCore_ROOTDIR}/include/VecCore/Backend/*.h"
  "URL \"${VecCore_SRC_URI}/VecCore-${VecCore_VERSION}.tar.gz\"
  URL_MD5 ${VecCore_SRC_MD5}
  #GIT_REPOSITORY \"${VecCore_SRC_URI}\"
  #GIT_TAG \"${VecCore_SRC_TAG}\"

  PREFIX external
  STAMP_DIR external/stamp
  BINARY_DIR external/build
  BUILD_IN_SOURCE 0
  LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
  CMAKE_ARGS -G \"${CMAKE_GENERATOR}\"
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_TESTING=OFF
             -DCUDA=${VECGEOM_ENABLE_CUDA} -DVC=${VECGEOM_VC}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
           \"-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}\"
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
           \"-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}\"
             -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
           \"-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}\"
           \"-DVc_DIR=${Vc_DIR}\"
  INSTALL_COMMAND env DESTDIR=${VecCore_DESTDIR} ${CMAKE_COMMAND} --build . --target install"
)

add_custom_target(VecCore)
add_dependencies(VecCore ${VecCore_PROJECT})

install(DIRECTORY ${VecCore_ROOTDIR}/ DESTINATION "." )

# Find VecCore with selected components turned on (CUDA and backend)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${VecCore_DESTDIR}/${CMAKE_INSTALL_PREFIX})

# Make sure to look again for a version of VecCore so that we can make sure to find
# the one we just build.
unset(VecCore_DIR CACHE)
