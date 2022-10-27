################################################################################
## TESTING
################################################################################
# Helper Macros/Functions

add_definitions(-UNDEBUG) # ensure asserts are enabled in all tests

macro(build_executables EXECUTABLES)
  foreach(EXECUTABLE ${EXECUTABLES})
    get_filename_component(TARGET_NAME ${EXECUTABLE} NAME_WE)
    add_executable(${TARGET_NAME} ${EXECUTABLE})
    target_link_libraries(${TARGET_NAME} PUBLIC
      $<IF:$<BOOL:${VECGEOM_ENABLE_CUDA}>,vecgeomcuda_static,>
      VecGeomTest)
  endforeach()
endmacro()

function(add_to_ctest EXECUTABLES)
  foreach(EXECUTABLE ${EXECUTABLES})
    get_filename_component(TARGET_NAME ${EXECUTABLE} NAME_WE)
    add_test(${TARGET_NAME} ${TARGET_NAME})
  endforeach()
endfunction()

# actual function for managing the download
function(DOWNLOAD_IF_NOT_INSTALLED FILE_URL LOCALFILE TARGETPATH MD5HASH )
  set(FILENAME "${TARGETPATH}/${LOCALFILE}")
  if(EXISTS "${FILENAME}")
    file(MD5 "${FILENAME}" MD5ACTUAL)
    if(MD5ACTUAL STREQUAL MD5HASH)
      return()
    else()
      message(WARNING "Hash not correct for data file ${FILENAME}:\n"
        "expected: ${MD5HASH}\n"
        "actual  : ${MD5ACTUAL}\n"
        "Trying to re-download")
    endif()
  endif()

  message(STATUS "Downloading ${LOCALFILE}")
  file(DOWNLOAD ${FILE_URL} ${FILENAME} SHOW_PROGRESS EXPECTED_HASH MD5=${MD5HASH})
endfunction()
################################################################################

################################################################################
# Download data if required
message(STATUS "Checking data files required for tests")
if(VECGEOM_ROOT)
  download_if_not_installed("https://geant4-data.web.cern.ch/geometry/detectors/cms2015.root"
    "cms2015.root"
    "${CMAKE_CURRENT_BINARY_DIR}"
    "fb3ede867532b372c5e6f7138d00c07e")
  download_if_not_installed("https://geant4-data.web.cern.ch/geometry/detectors/ExN03.root"
    "ExN03.root"
    "${CMAKE_CURRENT_BINARY_DIR}"
    "b6b0cfdd5035117171bfe1b7f1f40c3f")
endif()

################################################################################
# Define Testing libraries
################################################################################
add_subdirectory(VecGeomTest)


################################################################################
# Define executables
################################################################################
# - Dedicated Benchmark tests
if(VECGEOM_TEST_BENCHMARK)
  add_subdirectory(benchmark)
endif()

################################################################################
# - Static Analysis tests
if(VECGEOM_TEST_STATIC_ANALYSIS)
  add_subdirectory(static_analysis)
endif()

################################################################################
# - Core Tests
set(TEST_EXECUTABLES_CORE
  core/TestVoxelHashMap.cpp
  core/SafetyEstimatorTest.cpp
  core/ContainerTest.cpp
  core/create_geometry.cpp
  core/BitSetTest.cpp
  core/PlanesTest.cpp
  core/QuadrilateralTest.cpp
  core/Transformation3DTest.cpp
  # core/boolminustest.cpp
  # core/boolminustest2.cpp
  core/PhiWedgeTest.cpp
  core/ThetaConeTest.cpp
  core/TestConvexity.cpp
  core/BooleanConvexityTest.cpp
  unit_tests/TestEstimateSurfaceArea.cpp
  unit_tests/TestVecGeomPolycone.cpp
  unit_tests/TestNavStateIndex.cpp
  core/TestSExtru.cpp
  # to be enabled when running cleanly
  unit_tests/TestBooleans.cpp
  core/AssemblyTest.cpp
  core/TestMakeInstance.cpp
  #  services/CompNavStatePools.cpp
  core/TestMaskedAssign.cpp
  core/TestVector.cpp
  core/TestHybridBVH.cpp
  core/TestEarlyReturns.cpp
  core/CreateTessels.cpp
  core/CreateExtruded.cpp
  core/TestGeoManager.cpp
  )

if(VECGEOM_USE_INDEXEDNAVSTATES)
  list(APPEND TEST_EXECUTABLES_CORE core/TestNavigationState.cpp)
endif()

if (VECGEOM_USE_NAVINDEX)
  list(APPEND TEST_EXECUTABLES_CORE surfaces/testEM3.cpp)
  list(APPEND TEST_EXECUTABLES_CORE surfaces/testTubeSurfaces.cpp)
  list(APPEND TEST_EXECUTABLES_CORE surfaces/testTrdSurfaces.cpp)
  if (VECGEOM_ENABLE_CUDA)
    add_executable(testCUDA surfaces/testCUDA.cpp surfaces/testCUDA.cu)
    target_link_libraries(testCUDA PUBLIC vecgeomcuda_static)
    add_test(NAME testCUDA COMMAND testCUDA)
  endif()
endif()

if(NOT VECGEOM_ENABLE_CUDA)
  list(APPEND TEST_EXECUTABLES_CORE core/testVectorSafety.cpp)
endif()

build_executables("${TEST_EXECUTABLES_CORE}")
add_to_ctest("${TEST_EXECUTABLES_CORE}")

################################################################################
# - ROOT Tests
if(VECGEOM_ROOT)
  set(TEST_EXECUTABLES_ROOT
    core/BoxBoxIntersectionTest.cpp
    core/SplittedABBox.cpp
    root/root_geometry.cpp
    root/complex_test1.cpp
    root/E03Test.cpp
    root/ImportFromRootFileTest.cpp
    root/ImportTGeoPgon.cpp
    root/TestExportToROOT.cpp
    root/ExitingOrEntering.cpp
    root/AssemblyExample.cpp
    # higher level benchmarks or executables
    globalbenchmarks/LocatePointsBenchmark.cpp
    globalbenchmarks/SafetyKernelBenchmarker.cpp
    globalbenchmarks/NavigationKernelBenchmarker.cpp
    globalbenchmarks/TraceTrack.cpp
    globalbenchmarks/XRayBenchmarkFromROOTFile.cpp
    # ???
    services/NavigationSpecializerTest.cpp
    services/LibraryGenerator.cpp)

  if(VECGEOM_TEST_BENCHMARK)
    list(APPEND TEST_EXECUTABLES_ROOT
      root/BenchmarkShapeFromROOTFile.cpp
      root/BenchmarkShapeFromROOTFile_WithVisualization.cpp
      root/CompareDistances.cpp
      root/GenerateSurfacePoints.cpp
      shape_tester/shapeDebug.cpp)
  endif()

  if(VECGEOM_EMBREE)
    list(APPEND TEST_EXECUTABLES_ROOT core/EmbreeManagerTest.cpp)
  endif()

  add_executable(TestNavigationStatePool
    core/TestNavigationStatePool.cpp
    $<$<BOOL:VECGEOM_ENABLE_CUDA>:core/TestNavigationStatePool.cu>)
  target_link_libraries(TestNavigationStatePool PRIVATE
    $<IF:$<BOOL:${VECGEOM_ENABLE_CUDA}>,vecgeomcuda_static,vecgeom>
    $<$<BOOL:${VECGEOM_ROOT}>:VecGeomTest>)
  add_test(NAME TestNavigationStatePool COMMAND TestNavigationStatePool)

  # - separate list for ROOT UNIT tests
  set(TEST_UNITTESTEXECUTABLES_ROOT
    root/complex_test1.cpp
    root/E03Test.cpp
    root/TestExportToROOT.cpp
    root/ImportTGeoPgon.cpp)

  build_executables("${TEST_EXECUTABLES_ROOT}")
  add_to_ctest("${TEST_UNITTESTEXECUTABLES_ROOT}")
endif()

################################################################################
# - Visualization tests
if(VECGEOM_ROOT)
  set(TEST_EXECUTABLES_VISUALIZATION
    visualization/VisualizeTrap.cpp
    visualization/VisualizePolycone.cpp
    visualization/VisualizePolyhedron.cpp
    visualization/VisualizeParboloid.cpp
    visualization/VisualizeCone.cpp
    # visualization/VisualizeTorus.cpp
    visualization/VisualizeTube.cpp
    visualization/VisualizeScaled.cpp
    visualization/DebugTube.cpp
    visualization/DebugPolyhedron.cpp
    visualization/VisualizeSphere.cpp
    visualization/VisualizeGenTrap.cpp
    visualization/VisualizeHype.cpp
    visualization/VisualizeParallelepiped.cpp
    visualization/VisualizeCutTube.cpp
    # visualization/VisualizeExtruded.cpp
    )

  if(NOT VECGEOM_ENABLE_CUDA)
    list(APPEND TEST_EXECUTABLES_VISUALIZATION visualization/VisualizeMultiUnion.cpp)
  endif()

  build_executables("${TEST_EXECUTABLES_VISUALIZATION}")
endif()

################################################################################
# - Shape Tests
set(TEST_EXECUTABLES_SHAPES
  unit_tests/TestBox.cpp
  unit_tests/TestCons.cpp
  unit_tests/TestGenTrap.cpp
  unit_tests/TestTube.cpp
  unit_tests/TestEllipticalTube.cpp
  unit_tests/TestEllipticalCone.cpp
  unit_tests/TestEllipsoid.cpp
  unit_tests/TestCoaxialCones.cpp
  unit_tests/TestGenericPolycone.cpp
  unit_tests/TestLegendPolycone.cpp
  unit_tests/TestHype.cpp
  unit_tests/TestTrd.cpp
  unit_tests/TestTrap.cpp
  unit_tests/TestParallelepiped.cpp
  unit_tests/TestPolycone.cpp
  unit_tests/TestPolyhedra.cpp
  unit_tests/TestTet.cpp
  unit_tests/TestOrb.cpp
  unit_tests/TestSphere.cpp
  unit_tests/TestTorus2.cpp
  unit_tests/TestParaboloid.cpp
  unit_tests/TestReducedPolycone.cpp
  unit_tests/TestUtils3D.cpp
  unit_tests/TestMarchingCubes.cpp
  unit_tests/TestVECGEOM-601.cpp)

if(NOT VECGEOM_ENABLE_CUDA)
  list(APPEND TEST_EXECUTABLES_SHAPES
    unit_tests/TestTessellated.cpp
    unit_tests/TestMesh.cpp)
endif()

build_executables("${TEST_EXECUTABLES_SHAPES}")
# add unit tests
add_test(NAME TestBox COMMAND TestBox)
add_test(NAME TestCone COMMAND TestCons)
add_test(NAME TestGenTrap COMMAND TestGenTrap)
add_test(NAME TestHype COMMAND TestHype)
add_test(NAME TestLegendPolycone COMMAND TestLegendPolycone)
add_test(NAME TestOrb COMMAND TestOrb)
add_test(NAME TestPolycone COMMAND TestPolycone)
add_test(NAME TestPolyhedra COMMAND TestPolyhedra)
add_test(NAME TestParallelepiped COMMAND TestParallelepiped)
add_test(NAME TestParaboloid COMMAND TestParaboloid)
add_test(NAME TestSphere COMMAND TestSphere)
add_test(NAME TestTet COMMAND TestTet)
add_test(NAME TestTrap COMMAND TestTrap)
add_test(NAME TestTrd COMMAND TestTrd)
add_test(NAME TestTube COMMAND TestTube)
add_test(NAME TestVECGEOM-601 COMMAND TestVECGEOM-601)
if(NOT VECGEOM_ENABLE_CUDA)
  add_test(NAME TestTessellated COMMAND TestTessellated)
endif()

################################################################################
# - CUDA tests
if(VECGEOM_ENABLE_CUDA)
  if(VECGEOM_GDML)
    add_executable(BVHTest cuda/BVHTest.cu cuda/BVHTest.cpp )
    add_executable(GeometryTest cuda/GeometryTest.cpp cuda/GeometryTest.cu)

    target_link_libraries(BVHTest      PUBLIC vecgeomcuda_static vgdml)
    target_link_libraries(GeometryTest PUBLIC vecgeomcuda_static vgdml)

    add_test(NAME BVHTest      COMMAND BVHTest ${PROJECT_SOURCE_DIR}/test/gdml/gdmls/trackML.gdml)
    add_test(NAME GeometryTest-Device-trackML
      COMMAND GeometryTest ${PROJECT_SOURCE_DIR}/test/gdml/gdmls/trackML.gdml -validate false)
    add_test(NAME GeometryTest-Device-cms2018
      COMMAND GeometryTest ${PROJECT_SOURCE_DIR}/test/gdml/gdmls/cms2018.gdml -validate false -stacksize 32768)
  endif()
endif()

################################################################################
# - Shape Tester tests
set(TEST_EXECUTABLES_SHAPETESTER
  shape_tester/shape_testBox.cpp
  shape_tester/shape_testSExtru.cpp
  shape_tester/shape_testOrb.cpp
  shape_tester/shape_testSphere.cpp
  shape_tester/shape_testCone.cpp
  shape_tester/shape_testEllipticalCone.cpp
  shape_tester/shape_testTube.cpp
  shape_tester/shape_testEllipticalTube.cpp
  shape_tester/shape_testEllipsoid.cpp
  shape_tester/shape_testHype.cpp
  shape_tester/shape_testTrd.cpp
  shape_tester/shape_testTrapezoid.cpp
  shape_tester/shape_testTet.cpp
  shape_tester/shape_testParaboloid.cpp
  shape_tester/shape_testPolycone.cpp
  shape_tester/shape_testGenericPolycone.cpp
  shape_tester/shape_testGenTrap.cpp
  shape_tester/shape_testParallelepiped.cpp
  shape_tester/convention_testTube.cpp
  shape_tester/shape_testPolyhedron.cpp
  shape_tester/shape_testTorus2.cpp
  shape_tester/shape_testCutTube.cpp
  shape_tester/shape_testExtruded.cpp
  shape_tester/shape_testMultiUnion.cpp)

if(VECGEOM_ROOT)
  list(APPEND TEST_EXECUTABLES_SHAPETESTER
    shape_tester/shape_testFromROOTFile.cpp
    shape_tester/shape_debugFromROOTFile.cpp)
endif()

if(NOT VECGEOM_ENABLE_CUDA)
  list(APPEND TEST_EXECUTABLES_SHAPETESTER shape_tester/shape_testTessellated.cpp)
endif()

if(VECGEOM_TEST_BENCHMARK)
  build_executables("${TEST_EXECUTABLES_SHAPETESTER}")
endif()


if(VECGEOM_TEST_VALIDATION AND VECGEOM_ROOT)
  macro(add_cmsshapevalidation_test TESTNAME SHAPEFILE)
    add_test(NAME ${TESTNAME} COMMAND bash -c "${PROJECT_SOURCE_DIR}/test/scripts/RunRandomValidation ${PROJECT_BINARY_DIR}/BenchmarkShapeFromROOTFile  ${CMAKE_CURRENT_BINARY_DIR}/cms2015.root  ${PROJECT_SOURCE_DIR}/test/cmstestdata/${SHAPEFILE}" )
  endmacro()

  # Adding various shapes tests for nightlies
  add_cmsshapevalidation_test( tubevalidation cmstubes.txt )
  add_cmsshapevalidation_test( trapvalidation cmstraps.txt )
  add_cmsshapevalidation_test( polyconevalidation cmspolycones.txt )
  add_cmsshapevalidation_test( polyhedravalidation cmspolyhedra.txt )
  add_cmsshapevalidation_test( conevalidation cmscones.txt )
  add_cmsshapevalidation_test( boxvalidation cmsboxes.txt )
  #taken out due to problems: add_cmsshapevalidation_test( booleanvalidation cmsbooleans.txt )
  #taken out due to Issue-133: add_cmsshapevalidation_test( torusvalidation cmstori.txt )
endif()

################################################################################
# - GDML tests
if(VECGEOM_GDML)
  add_subdirectory(gdml)
endif()
