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
  # We use VGM to convert (test) geometries from ROOT input to G4
  # without needing an intermediate gdml file
  # for user friendliness .. added as external project for now
  set(VGM_INSTALL "${PROJECT_BINARY_DIR}/vgminstall")
  set(VGM_LIBRARIES
    "${VGM_INSTALL}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}BaseVGM${CMAKE_SHARED_LIBRARY_SUFFIX}"
    "${VGM_INSTALL}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}Geant4GM${CMAKE_SHARED_LIBRARY_SUFFIX}"
    "${VGM_INSTALL}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}RootGM${CMAKE_SHARED_LIBRARY_SUFFIX}")

  ExternalProject_Add(VGM
    GIT_REPOSITORY "https://github.com/vmc-project/vgm.git"
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${VGM_INSTALL}
      -DWITH_EXAMPLES=OFF
      -DGeant4_DIR=${Geant4_DIR}
      -DROOT_DIR=${ROOT_DIR}
      -DCMAKE_INSTALL_LIBDIR=${VGM_INSTALL}/lib
    BUILD_BYPRODUCTS ${VGM_LIBRARIES})

  # This is hacky, but avoids chicken-egg of ExternalPackage vs FindPackage
  # NB: Realistically, either need to vendor/FetchContent VGM, or use external package...
  if(NOT EXISTS "${VGM_INSTALL}/include")
    file(MAKE_DIRECTORY "${VGM_INSTALL}/include")
  endif()
  add_library(VecGeom::VGM INTERFACE IMPORTED)
  target_include_directories(VecGeom::VGM INTERFACE "${VGM_INSTALL}/include")
  cuda_rdc_target_link_libraries(VecGeom::VGM INTERFACE ${VGM_LIBRARIES})

  cuda_rdc_target_link_libraries(VecGeomTest PRIVATE VecGeom::VGM)
  add_dependencies(VecGeomTest VGM)
endif()
