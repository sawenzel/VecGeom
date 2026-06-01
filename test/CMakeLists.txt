################################################################################
## TESTING
################################################################################
# Helper Macros/Functions
################################################################################
# - Build and CTest a GTest-based VecGeom test
function(vecgeom_add_test EXECUTABLE)
  get_filename_component(TARGET_NAME ${EXECUTABLE} NAME_WE)
  add_executable(${TARGET_NAME} ${EXECUTABLE})
  cuda_rdc_target_link_libraries(${TARGET_NAME} PRIVATE vecgeom GTest::gtest_main)
  add_test(${TARGET_NAME} ${TARGET_NAME})
endfunction()
################################################################################

add_definitions(-UNDEBUG) # ensure asserts are enabled in all tests

macro(build_executables EXECUTABLES)
  foreach(EXECUTABLE ${EXECUTABLES})
    get_filename_component(TARGET_NAME ${EXECUTABLE} NAME_WE)
    add_executable(${TARGET_NAME} ${EXECUTABLE})
    cuda_rdc_target_link_libraries(${TARGET_NAME} PUBLIC VecGeomTest)
  endforeach()
endmacro()

function(add_to_ctest EXECUTABLES)
  foreach(EXECUTABLE ${EXECUTABLES})
    get_filename_component(TARGET_NAME ${EXECUTABLE} NAME_WE)
    add_test(${TARGET_NAME} ${TARGET_NAME})
  endforeach()
endfunction()

function(add_shape_contract_case_tests TARGET_NAME TIER LABELS)
  foreach(SOLID_CASE ${ARGN})
    set(ENABLED_FAMILIES contracts normals surface distance_to_out distance_to_in safeties hit_consistency)
    foreach(FAMILY ${ENABLED_FAMILIES})
      list(FIND SHAPE_CONTRACT_DISABLED_CTEST_CASE_FAMILIES ${SOLID_CASE}:${FAMILY} DISABLED_FAMILY_INDEX)
      list(FIND SHAPE_CONTRACT_DISABLED_CTEST_TIER_FAMILIES ${TIER}:${SOLID_CASE}:${FAMILY} DISABLED_TIER_FAMILY_INDEX)
      if(DISABLED_FAMILY_INDEX GREATER -1 OR DISABLED_TIER_FAMILY_INDEX GREATER -1)
        list(REMOVE_ITEM ENABLED_FAMILIES ${FAMILY})
      endif()
    endforeach()

    list(LENGTH ENABLED_FAMILIES ENABLED_FAMILY_COUNT)
    if(ENABLED_FAMILY_COUNT EQUAL 0)
      continue()
    endif()

    if(ENABLED_FAMILY_COUNT EQUAL 7)
      add_test(NAME ${TARGET_NAME}:${TIER}:${SOLID_CASE}
        COMMAND ${TARGET_NAME} -tier ${TIER} -test_family all -case_name ${SOLID_CASE})
      set_tests_properties(${TARGET_NAME}:${TIER}:${SOLID_CASE} PROPERTIES LABELS "${LABELS}")
    else()
      foreach(FAMILY ${ENABLED_FAMILIES})
        add_test(NAME ${TARGET_NAME}:${TIER}:${FAMILY}:${SOLID_CASE}
          COMMAND ${TARGET_NAME} -tier ${TIER} -test_family ${FAMILY} -case_name ${SOLID_CASE})
        set_tests_properties(${TARGET_NAME}:${TIER}:${FAMILY}:${SOLID_CASE} PROPERTIES LABELS "${LABELS};${FAMILY}")
      endforeach()
    endif()
  endforeach()
endfunction()

function(add_shape_manual_edge_case_tests TARGET_NAME TIER LABELS)
  foreach(SOLID_CASE ${ARGN})
    list(FIND SHAPE_CONTRACT_DISABLED_CTEST_CASE_FAMILIES ${SOLID_CASE}:manual_edge_cases DISABLED_FAMILY_INDEX)
    list(FIND SHAPE_CONTRACT_DISABLED_CTEST_TIER_FAMILIES ${TIER}:${SOLID_CASE}:manual_edge_cases DISABLED_TIER_FAMILY_INDEX)
    if(DISABLED_FAMILY_INDEX GREATER -1 OR DISABLED_TIER_FAMILY_INDEX GREATER -1)
      continue()
    endif()

    add_test(NAME ${TARGET_NAME}:${TIER}:manual_edge_cases:${SOLID_CASE}
      COMMAND ${TARGET_NAME} -tier ${TIER} -test_family manual_edge_cases -case_name ${SOLID_CASE})
    set_tests_properties(${TARGET_NAME}:${TIER}:manual_edge_cases:${SOLID_CASE}
      PROPERTIES LABELS "${LABELS};manual_edge_cases")
  endforeach()
endfunction()

function(add_tests_for_files TEST_NAME)
  set(COMMAND_ARGS)
  set(GEOMS_ARGS)
  set(COLLECTING_COMMAND_ARGS OFF)
  set(COLLECTING_GEOMS_ARGS OFF)
  # Loop through the arguments
  foreach(ARG IN LISTS ARGN)
    if(ARG STREQUAL "COMMAND")
      set(COLLECTING_COMMAND_ARGS ON)
      set(COLLECTING_GEOMS_ARGS OFF)
    elseif(ARG STREQUAL "GEOMS")
      set(COLLECTING_GEOMS_ARGS ON)
      set(COLLECTING_COMMAND_ARGS OFF)
    elseif(COLLECTING_COMMAND_ARGS)
      list(APPEND COMMAND_ARGS "${ARG}")
    elseif(COLLECTING_GEOMS_ARGS)
      list(APPEND GEOMS_ARGS "${ARG}")
    endif()
  endforeach()
  # Iterate over the file list
  foreach(FILE ${GEOMS_ARGS})
    # Get the file name without the path
    get_filename_component(FILE_NAME ${FILE} NAME)
    # Create a test with the command and the file name
    add_test(NAME ${TEST_NAME}:${FILE_NAME} COMMAND ${COMMAND_ARGS} ${FILE})
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

# Get all files in a directory
function(get_files_in_directory DIR OUTPUT_VAR)
    if(NOT IS_DIRECTORY ${DIR})
        message(FATAL_ERROR "Directory does not exist: ${DIR}")
    endif()
    file(GLOB FILES_IN_DIR "${DIR}/*")
    set(${OUTPUT_VAR} ${FILES_IN_DIR} PARENT_SCOPE)
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
vecgeom_add_test(core/TestVector.cpp)
vecgeom_add_test(core/TestVoxelHashMap.cpp)

set(TEST_EXECUTABLES_CORE
  core/RNGTest.cpp
  core/SafetyEstimatorTest.cpp
  core/ShapeCheckResultTest.cpp
  core/ShapeContractDeterministicTest.cpp
  core/ShapeSampleSetTest.cpp
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
  core/TestExtrudedMesh.cpp
  core/TestTessellatedMesh.cpp
  # to be enabled when running cleanly
  unit_tests/TestBooleans.cpp
  core/AssemblyTest.cpp
  core/TestMakeInstance.cpp
  #  services/CompNavStatePools.cpp
  core/TestMaskedAssign.cpp
  core/TestOvershoot.cpp
  core/TestHybridBVH.cpp
  core/TestBVH.cpp
  core/TestEarlyReturns.cpp
  # core/CreateTessels.cpp //--> To be adapted to new struct
  core/CreateExtruded.cpp
  core/TestGeoManager.cpp
  core/TestTriangularFacet.cpp
  core/TestNavigationState.cpp
  )

if (VECGEOM_USE_SURF)
  list(APPEND TEST_EXECUTABLES_CORE surfaces/testLogic.cpp)
  list(APPEND TEST_EXECUTABLES_CORE surfaces/testEM3.cpp)
  list(APPEND TEST_EXECUTABLES_CORE surfaces/testTubeSurfaces.cpp)
  list(APPEND TEST_EXECUTABLES_CORE surfaces/testTrdSurfaces.cpp)
  list(APPEND TEST_EXECUTABLES_CORE surfaces/testEmbedding.cpp)

  if (VECGEOM_GDML)
    get_files_in_directory("${CMAKE_SOURCE_DIR}/test/gdml/gdmls/unit_tests" SURFACE_MODEL_TESTS)
    if (VECGEOM_ENABLE_CUDA)
      add_executable(testRaytracing surfaces/testRaytracing.cpp surfaces/testRaytracing.cu)
      cuda_rdc_target_link_libraries(testRaytracing PUBLIC vgdml)
      add_test(NAME testRaytracing COMMAND testRaytracing -gdml_name ${PROJECT_SOURCE_DIR}/test/gdml/gdmls/trackML.gdml)
      add_tests_for_files(testRaytracing COMMAND testRaytracing -nrays 100000 -mmunit 0.1 -debug 1 -ongpu 1 -accept_zeros 1 -test_bvh 1 -gdml_name GEOMS ${SURFACE_MODEL_TESTS})
    else()
      add_executable(testRaytracing surfaces/testRaytracing.cpp)
      cuda_rdc_target_link_libraries(testRaytracing PUBLIC vgdml)
      add_test(NAME testRaytracing COMMAND testRaytracing -ongpu 0 -gdml_name ${PROJECT_SOURCE_DIR}/test/gdml/gdmls/trackML.gdml)
      add_tests_for_files(testRaytracing COMMAND testRaytracing -nrays 100000 -mmunit 0.1 -debug 1 -ongpu 0 -accept_zeros 1 -test_bvh 1 -gdml_name GEOMS ${SURFACE_MODEL_TESTS})
    endif()
    add_executable(testBVH surfaces/testBVH.cpp)
    cuda_rdc_target_link_libraries(testBVH PUBLIC vgdml)
  endif()

  if (VECGEOM_ENABLE_CUDA AND VECGEOM_GDML)
    add_executable(testCUDA surfaces/testCUDA.cpp surfaces/testCUDA.cu)
    cuda_rdc_target_link_libraries(testCUDA PUBLIC vecgeom vgdml)
    add_test(NAME testCUDA COMMAND testCUDA)
  endif()
endif()

build_executables("${TEST_EXECUTABLES_CORE}")
add_to_ctest("${TEST_EXECUTABLES_CORE}")

add_executable(ShapeContractTest core/ShapeContractTest.cpp)
cuda_rdc_target_link_libraries(ShapeContractTest PUBLIC VecGeomTest)

set(VECGEOM_SHAPE_CONTRACT_CTEST_TIER "medium" CACHE STRING
  "ShapeContractTest tier registered in CTest (fast, medium, or slow)")
set_property(CACHE VECGEOM_SHAPE_CONTRACT_CTEST_TIER PROPERTY STRINGS fast medium slow)
set(_VECGEOM_SHAPE_CONTRACT_ALLOWED_CTEST_TIERS fast medium slow)
if(NOT VECGEOM_SHAPE_CONTRACT_CTEST_TIER IN_LIST _VECGEOM_SHAPE_CONTRACT_ALLOWED_CTEST_TIERS)
  message(FATAL_ERROR
    "Unsupported VECGEOM_SHAPE_CONTRACT_CTEST_TIER='${VECGEOM_SHAPE_CONTRACT_CTEST_TIER}'. "
    "Use fast, medium, or slow.")
endif()

set(SHAPE_CONTRACT_SOLID_CASES
  # Box
  box

  # Tube
  tube_fullphi
  tube_section
  tube_thin_wall_long
  tube_narrow_phi
  tube_almost_full_phi
  tube_short_disk

  # Cone
  cone_fullphi
  cone_section
  cone_thin_shell
  cone_almost_cylinder
  cone_narrow_phi
  cone_almost_full_phi

  # Ellipsoid
  ellipsoid

  # EllipticalCone
  elliptical_cone

  # EllipticalTube
  elliptical_tube
  elliptical_tube_thin
  elliptical_tube_long

  # Parallelepiped
  parallelepiped_general
  parallelepiped_high_shear

  # Tet
  tet
  tet_sliver_like

  # Trd
  trd_boxlike
  trd_extreme_aspect
  trd_increasing_xy

  # Trapezoid
  trapezoid_corners
  trapezoid_extreme_skew
  trapezoid_near_para

  polycone_cms_like
  generic_polycone_irregular
  polycone_two_section_sharp_jump
  polycone_many_section_alternating
  polycone_nearly_repeated_z
  generic_polycone_zigzag_profile
  paraboloid

  # CutTube
  cuttube_section_inner

  # Hype
  hype

  # Orb
  orb

  # Sphere
  sphere_thin_shell
  sphere_narrow_phi
  sphere_narrow_theta
  sphere_almost_full_phi
  sphere_section

  # Torus2
  torus2_general

  # Polyhedron
  polyhedron_phi_section

  # GenTrap
  gentrap_twisted

  # SExtru
  sextru_circular

  # Extruded
  extruded_multilayer

  # Tessellated
  tessellated_orb

  # MultiUnion
  multiunion_boxes

  # Boolean
  boolean_union_offset_boxes
  boolean_union_rotated_touching_boxes
  boolean_union_overlapping_tube_cone
  boolean_union_disjoint_tubes
  boolean_intersection_box_tube
  boolean_intersection_rotated_boxes
  boolean_intersection_rotated_tube_cone
  boolean_intersection_nested_union
  boolean_subtraction_box_tube
  boolean_subtraction_tube_from_cone
  boolean_subtraction_union_from_box
  boolean_subtraction_exact_phi_wedge
  boolean_union_of_subtractions
  boolean_intersection_of_subtractions
  boolean_nested_transformed_subtraction
  boolean_subtraction_exact_polycone_sections
  boolean_subtraction_polyhedron_exact_z_slab
  boolean_subtraction_polyhedron_exact_radial_shell
  boolean_subtraction_polyhedron_side_cut_tube
  boolean_subtraction_polyhedron_phi_seam_rotated
  boolean_subtraction_polyhedron_near_coincident
  )

# this list should be resolved and go away
set(SHAPE_CONTRACT_DISABLED_CTEST_CASE_FAMILIES
  elliptical_cone:surface
  polycone_cms_like:contracts
  polycone_cms_like:normals
  polycone_cms_like:surface
  paraboloid:contracts
  paraboloid:normals
  paraboloid:surface
  paraboloid:distance_to_in
  paraboloid:hit_consistency
  multiunion_boxes:normals
  multiunion_boxes:distance_to_in
  multiunion_boxes:hit_consistency
  hype:distance_to_in
  hype:hit_consistency
  hype:surface
  orb:surface
  sphere_thin_shell:surface
  sphere_narrow_phi:surface
  sphere_narrow_theta:normals
  sphere_narrow_theta:surface
  sphere_almost_full_phi:surface
  sphere_section:distance_to_in
  sphere_section:hit_consistency
  sphere_section:surface
  torus2_general:hit_consistency
  torus2_general:surface
  extruded_multilayer:surface
  )

if(VECGEOM_TEST_STATIC_ANALYSIS)
  list(APPEND SHAPE_CONTRACT_DISABLED_CTEST_CASE_FAMILIES
    extruded_multilayer:contracts
    extruded_multilayer:normals
    extruded_multilayer:distance_to_out
    extruded_multilayer:distance_to_in
    extruded_multilayer:safeties
    extruded_multilayer:hit_consistency
    )
endif()

# Keep the tier-specific hook for future plumbing, but medium findings are
# intentionally promoted to the common family-disable list so fast/medium stay
# aligned on the same enabled coverage.
set(SHAPE_CONTRACT_DISABLED_CTEST_TIER_FAMILIES
  )

set(SHAPE_CONTRACT_MANUAL_EDGE_CASE_SOLID_CASES
  box
  boolean_intersection_rotated_boxes
  boolean_nested_transformed_subtraction
  boolean_subtraction_polyhedron_exact_z_slab
  boolean_subtraction_polyhedron_exact_radial_shell
  boolean_subtraction_polyhedron_side_cut_tube
  boolean_subtraction_polyhedron_phi_seam_rotated
  boolean_subtraction_polyhedron_near_coincident
  cuttube_section_inner
  cone_fullphi
  cone_narrow_phi
  generic_polycone_irregular
  generic_polycone_zigzag_profile
  polycone_two_section_sharp_jump
  tube_fullphi
  )

# Register one unit test per configured solid in the selected CTest tier while
# keeping disabled coverage in family-scoped lists only. Disabled families stay
# available through the ShapeContractTest executable for local debugging
# without recompilation.
set(SHAPE_CONTRACT_CTEST_TIERS ${VECGEOM_SHAPE_CONTRACT_CTEST_TIER})

foreach(SHAPE_CONTRACT_CTEST_TIER_NAME ${SHAPE_CONTRACT_CTEST_TIERS})
  if(SHAPE_CONTRACT_CTEST_TIER_NAME STREQUAL "fast")
    set(SHAPE_CONTRACT_CASE_LABELS "fast;contracts;normals;surface;distance_to_out;distance_to_in;safeties;hit_consistency")
    set(SHAPE_CONTRACT_MANUAL_LABELS "fast;manual_edge_cases")
  elseif(SHAPE_CONTRACT_CTEST_TIER_NAME STREQUAL "medium")
    set(SHAPE_CONTRACT_CASE_LABELS "medium;contracts;normals;surface;distance_to_out;distance_to_in;safeties;hit_consistency;seeded")
    set(SHAPE_CONTRACT_MANUAL_LABELS "medium;manual_edge_cases")
  elseif(SHAPE_CONTRACT_CTEST_TIER_NAME STREQUAL "slow")
    set(SHAPE_CONTRACT_CASE_LABELS "slow;contracts;normals;surface;distance_to_out;distance_to_in;safeties;hit_consistency;seeded")
    set(SHAPE_CONTRACT_MANUAL_LABELS "slow;manual_edge_cases")
  endif()

  add_shape_contract_case_tests(ShapeContractTest ${SHAPE_CONTRACT_CTEST_TIER_NAME} "${SHAPE_CONTRACT_CASE_LABELS}"
    ${SHAPE_CONTRACT_SOLID_CASES})
  add_shape_manual_edge_case_tests(ShapeContractTest ${SHAPE_CONTRACT_CTEST_TIER_NAME} "${SHAPE_CONTRACT_MANUAL_LABELS}"
    ${SHAPE_CONTRACT_MANUAL_EDGE_CASE_SOLID_CASES})
endforeach()

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


  if(VECGEOM_ENABLE_CUDA)
    list(APPEND _test_nav_src core/TestNavigationStatePool.cu)
  endif()
  if(VECGEOM_ROOT)
    list(APPEND _test_nav_libs VecGeomTest)
  endif()
  add_executable(TestNavigationStatePool
    core/TestNavigationStatePool.cpp
    ${_test_nav_src})
  cuda_rdc_target_link_libraries(TestNavigationStatePool PRIVATE
    vecgeom ${_test_nav_libs})
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
  unit_tests/TestTessellated.cpp
  unit_tests/TestVECGEOM-601.cpp)

if(NOT VECGEOM_ENABLE_CUDA)
  list(APPEND TEST_EXECUTABLES_SHAPES
    unit_tests/TestMesh.cpp)
endif()

build_executables("${TEST_EXECUTABLES_SHAPES}")
# add unit tests
add_test(NAME TestBox COMMAND TestBox)
add_test(NAME TestCone COMMAND TestCons)
add_test(NAME TestGenTrap COMMAND TestGenTrap)
add_test(NAME TestHype COMMAND TestHype)
#add_test(NAME TestLegendPolycone COMMAND TestLegendPolycone)
add_test(NAME TestOrb COMMAND TestOrb)
add_test(NAME TestGenericPolycone COMMAND TestGenericPolycone)
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
add_test(NAME TestTessellated COMMAND TestTessellated)

################################################################################
# - CUDA tests
if(VECGEOM_ENABLE_CUDA)
  if(VECGEOM_GDML)
    add_executable(BVHTest cuda/BVHTest.cu cuda/BVHTest.cpp )
    add_executable(GeometryTest cuda/GeometryTest.cpp cuda/GeometryTest.cu)

    cuda_rdc_target_link_libraries(BVHTest      PUBLIC vecgeom vgdml)
    cuda_rdc_target_link_libraries(GeometryTest PUBLIC vecgeom vgdml)

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
  shape_tester/shape_testMultiUnion.cpp
  shape_tester/shape_testTessellated.cpp
)

if(VECGEOM_ROOT)
  list(APPEND TEST_EXECUTABLES_SHAPETESTER
    shape_tester/shape_testFromROOTFile.cpp
    shape_tester/shape_debugFromROOTFile.cpp)
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
