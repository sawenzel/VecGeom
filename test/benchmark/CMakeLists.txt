#define some benchmark executables
set(TEST_EXECUTABLES_BENCHMARK
  ${PROJECT_SOURCE_DIR}/test/benchmark/BoxBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/SExtruBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/ConcaveSExtruBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/ParaboloidBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/ParaboloidScriptBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/ParallelepipedBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/PolyhedronBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/TubeBenchmark.cpp
#  ${PROJECT_SOURCE_DIR}/test/benchmark/BoxMinusHollowTubeBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/BoxMinusHollowTubeBenchmark_virtual.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/BoxUnionHollowTubeBenchmark_virtual.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/BoxIntersectHollowTubeBenchmark_virtual.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/BoxUnionBoxBenchmark_virtual.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/BoxMinusHalfSpaceUnionBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/3LevelSubtractionBenchmark_virtual.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/TorusBenchmark2.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/TrapezoidBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/TrapezoidBenchmarkScript.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/OrbBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/SphereBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/HypeBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/TetBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/TrdBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/ConeBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/EllipticalTubeBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/EllipticalConeBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/EllipsoidBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/PolyconeBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/GenericPolyconeBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/ScaledBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/BoxScaledBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/GenTrapBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/CutTubeBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/TessellatedBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/ExtrudedBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/MultiUnionBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/UnionFlatteningBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/globalbenchmarks/NavigationBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/BenchmarkVGDMLShape.cpp
)

build_executables("${TEST_EXECUTABLES_BENCHMARK}")

if(VECGEOM_GDML)
  build_executables(${PROJECT_SOURCE_DIR}/test/benchmark/TestNavIndex.cpp)
  if(VECGEOM_ENABLE_CUDA)
    cuda_rdc_target_sources(TestNavIndex PRIVATE TestNavIndex.cu)
  endif()
  cuda_rdc_target_link_libraries(TestNavIndex PUBLIC vgdml)
  cuda_rdc_target_link_libraries(BenchmarkVGDMLShape PUBLIC vgdml)
endif()

#define which ones should be run under ctest
set(THISCTESTS
  # TESTS MARKED # FAIL AND NEED ATTENTION
  ${PROJECT_SOURCE_DIR}/test/benchmark/BoxBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/SExtruBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/ConcaveSExtruBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/ParaboloidBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/ParaboloidScriptBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/ParallelepipedBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/PolyhedronBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/TubeBenchmark.cpp
# ${PROJECT_SOURCE_DIR}/test/benchmark/BoxMinusHollowTubeBenchmark_virtual.cpp
# ${PROJECT_SOURCE_DIR}/test/benchmark/BoxUnionHollowTubeBenchmark_virtual.cpp
# ${PROJECT_SOURCE_DIR}/test/benchmark/BoxIntersectHollowTubeBenchmark_virtual.cpp
# ${PROJECT_SOURCE_DIR}/test/benchmark/BoxUnionBoxBenchmark_virtual.cpp
# ${PROJECT_SOURCE_DIR}/test/benchmark/3LevelSubtractionBenchmark_virtual.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/TorusBenchmark2.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/TrapezoidBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/TrapezoidBenchmarkScript.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/OrbBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/SphereBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/HypeBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/TrdBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/ConeBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/GenTrapBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/ScaledBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/BoxScaledBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/CutTubeBenchmark.cpp
  ${PROJECT_SOURCE_DIR}/test/benchmark/PolyconeBenchmark.cpp
)

add_to_ctest("${CTESTS_BENCHMARK}")
