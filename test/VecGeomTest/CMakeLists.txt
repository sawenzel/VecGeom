set(VecGeomTest_sources
  # Shape tester
  ShapeTester.h
  ShapeTester.cpp #includes ConventionChecker.cpp ...
  # Benchmark
  Benchmarker.h
  Benchmarker.cpp
  BenchmarkResult.h
  BenchmarkResult.cpp
  NavigationBenchmarker.h
  NavigationBenchmarker.cpp
  VolumePointers.h
  VolumePointers.cpp)
if(VECGEOM_ENABLE_CUDA)
  list(APPEND VecGeomTest_sources
    Benchmarker.cu
    NavigationBenchmarker.cu
  )
endif()
if(VECGEOM_ROOT)
  list(APPEND VecGeomTest_sources
    # ROOT
    NavStateConverter.h
    PlacedRootVolume.cpp
    PlacedRootVolume.h
    RootGeoManager.cpp
    RootGeoManager.h
    UnplacedRootVolume.cpp
    UnplacedRootVolume.h
    ShapeDebugger.cpp
    ShapeDebugger.h
    Visualizer.cpp
    Visualizer.h
  )
endif()
if(VECGEOM_GEANT4)
  list(APPEND VecGeomTest_sources
    # Geant4
    G4GeoManager.h
    G4GeoManager.cpp
  )
endif()
cuda_rdc_add_library(VecGeomTest ${VecGeomTest_sources})

# Eventually Re-expose compile definitions for when we finally sever VecGeom's ROOT/Geant4 dependencies
# Currently done by vecgeom config header
#target_compile_definitions(VecGeomTest PUBLIC
#$<$<BOOL:VECGEOM_GEANT4>:VECGEOM_GEANT4>
#  $<$<BOOL:VECGEOM_ROOT>:VECGEOM_ROOT>
#  )

cuda_rdc_target_include_directories(VecGeomTest PUBLIC ${PROJECT_SOURCE_DIR}/test)
if(VECGEOM_ROOT)
  list(APPEND _extra_libs ROOT::Graf3d)
  list(APPEND _extra_priv_libs ROOT::Core ROOT::Geom ROOT::Gpad)
endif()
if(VECGEOM_GEANT4)
  list(APPEND _extra_libs Geant4::G4geometry)
  # G4persistency public because there is direct use of it in tests
  if(TARGET Geant4::G4persistency)
    list(APPEND _extra_libs Geant4::G4persistency)
  else()
    list(APPEND _extra_libs Geant4::G4gdml)
  endif()
endif()
cuda_rdc_target_link_libraries(VecGeomTest
  PUBLIC
    vecgeom
    ${_extra_libs}
  PRIVATE
    ${_extra_priv_libs}
)

if(VECGEOM_GEANT4 AND VECGEOM_ROOT)
  # Install prefix for VGM built via ExternalProject
  set(VGM_INSTALL "${PROJECT_BINARY_DIR}/vgminstall")

  ExternalProject_Add(VGM
    GIT_REPOSITORY "https://github.com/vmc-project/vgm.git"
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${VGM_INSTALL}
      -DWITH_EXAMPLES=OFF
      -DGeant4_DIR=${Geant4_DIR}
      -DROOT_DIR=${ROOT_DIR}
      -DCMAKE_INSTALL_LIBDIR=${VGM_INSTALL}/lib
      -DCMAKE_BUILD_RPATH=\$ORIGIN
      -DCMAKE_INSTALL_RPATH=\$ORIGIN
      -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
  )

  # Interface target for VGM
  add_library(VecGeomVGM INTERFACE)
  add_library(VecGeom::VGM ALIAS VecGeomVGM)

  target_include_directories(VecGeomVGM INTERFACE "${VGM_INSTALL}/include")
  target_link_directories(   VecGeomVGM INTERFACE "${VGM_INSTALL}/lib")   # <— key

  # Link by library names; resolver will find them in ${VGM_INSTALL}/lib at build time
  cuda_rdc_target_link_libraries(VecGeomVGM INTERFACE
    BaseVGM ClhepVGM Geant4GM RootGM
  )

  # Ensure build/link happen after VGM is built/installed
  add_dependencies(VecGeomTest_objects VGM)
  add_dependencies(VecGeomTest         VGM)
  add_dependencies(VecGeomVGM          VGM)   # optional, helps if something links VecGeom::VGM directly

  # Run-from-build-tree rpath for anything linking VecGeomTest
  set_property(TARGET VecGeomTest APPEND PROPERTY BUILD_RPATH   "${VGM_INSTALL}/lib")
  set_property(TARGET VecGeomTest APPEND PROPERTY INSTALL_RPATH "${VGM_INSTALL}/lib")
  cuda_rdc_target_link_libraries(VecGeomTest PUBLIC VecGeom::VGM)
endif()
