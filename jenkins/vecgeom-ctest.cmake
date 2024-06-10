include(${CMAKE_CURRENT_LIST_DIR}/vecgeom-configure.cmake)

################################################################################
# Test custom update with a dashboard script.
message("Running CTest Dashboard Script (custom update)...")
include("${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake")

ctest_start(${MODEL} TRACK ${MODEL} APPEND)

ctest_test(BUILD ${CTEST_BINARY_DIRECTORY}
          APPEND)
ctest_submit(PARTS Test)

if(${MODEL} MATCHES NightlyMemoryCheck)
  ctest_submit(PARTS MemCheck)
endif()
