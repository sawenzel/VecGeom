add_library(VecGeomTest STATIC
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
  VolumePointers.cpp
  $<$<BOOL:${VECGEOM_ENABLE_CUDA}>:
    Benchmarker.cu;
    NavigationBenchmarker.cu>
  # ROOT
  $<$<BOOL:${VECGEOM_ROOT}>:
    NavStateConverter.h;
    PlacedRootVolume.cpp;
    PlacedRootVolume.h;
    RootGeoManager.cpp;
    RootGeoManager.h;
    UnplacedRootVolume.cpp;
    UnplacedRootVolume.h;
    ShapeDebugger.cpp;
    ShapeDebugger.h;
    Visualizer.cpp;
    Visualizer.h>
  # Geant4
  $<$<BOOL:${VECGEOM_GEANT4}>:
    G4GeoManager.h;
    G4GeoManager.cpp>
  )

# Eventually Re-expose compile definitions for when we finally sever VecGeom's ROOT/Geant4 dependencies
# Currently done by vecgeom config header
#target_compile_definitions(VecGeomTest PUBLIC
#$<$<BOOL:VECGEOM_GEANT4>:VECGEOM_GEANT4>
#  $<$<BOOL:VECGEOM_ROOT>:VECGEOM_ROOT>
#  )

target_include_directories(VecGeomTest PUBLIC ${PROJECT_SOURCE_DIR}/test)
target_link_libraries(VecGeomTest
  PUBLIC
    vecgeom
    "$<$<BOOL:${VECGEOM_ROOT}>:ROOT::Graf3d>"
    # G4persistency public because there is direct use of it in tests
    "$<$<BOOL:${VECGEOM_GEANT4}>:Geant4::G4geometry;$<IF:$<TARGET_EXISTS:Geant4::G4persistency>,Geant4::G4persistency,Geant4::G4gdml>>"
  PRIVATE
    "$<$<BOOL:${VECGEOM_ROOT}>:ROOT::Core;ROOT::Geom;ROOT::Gpad>"
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
  target_link_libraries(VecGeom::VGM INTERFACE ${VGM_LIBRARIES})

  target_link_libraries(VecGeomTest PRIVATE VecGeom::VGM)
  add_dependencies(VecGeomTest VGM)
endif()
