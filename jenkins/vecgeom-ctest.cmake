include(${CMAKE_CURRENT_LIST_DIR}/vecgeom-configure.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/vecgeom-submit.cmake)

################################################################################
# Test custom update with a dashboard script.
message("Running CTest Dashboard Script (custom update)...")
include("${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake")

ctest_start(${MODEL} TRACK ${MODEL} APPEND)

ctest_test(BUILD ${CTEST_BINARY_DIRECTORY}
          APPEND)
vecgeom_ctest_submit(PARTS Test)

if(${MODEL} MATCHES NightlyMemoryCheck)
  vecgeom_ctest_submit(PARTS MemCheck)
endif()
