#    \file CMakeLists.txt
#    \brief builds a library for interoperation of VecGeom with GDML
#
#    \authors Author:  Dmitry Savin
#
#    \license Distributed under the Apache license 2.0

add_library(vgdml
  include/Auxiliary.h
  include/Backend.h
  include/Frontend.h
  include/Helper.h
  include/MaterialInfo.h
  include/RegionInfo.h
  include/Middleware.h
  include/ReflFactory.h
  src/Backend.cpp
  src/Frontend.cpp
  src/Helper.cpp
  src/Middleware.cpp
)

target_include_directories(vgdml PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
cuda_rdc_target_link_libraries(vgdml PUBLIC XercesC::XercesC vecgeom)

if(VECGEOM_GDMLDEBUG)
  # Only used in Middleware.cpp, and to set a constant
  target_compile_definitions(vgdml PRIVATE GDMLDEBUG)
endif()

install(TARGETS vgdml EXPORT VecGeomTargets DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/VecGeom/gdml"
  FILES_MATCHING PATTERN "*.h"
)
