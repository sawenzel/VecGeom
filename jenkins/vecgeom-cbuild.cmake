include(${CMAKE_CURRENT_LIST_DIR}/vecgeom-configure.cmake)

################################################################################
# Test custom update with a dashboard script.
message("Running CTest Dashboard Script (custom update)...")
include("${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake")

# ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})
file(REMOVE_RECURSE ${CTEST_BINARY_DIRECTORY})

ctest_start(${MODEL} TRACK ${MODEL})
#ctest_update(SOURCE ${CTEST_SOURCE_DIRECTORY})

ctest_configure(BUILD   ${CTEST_BINARY_DIRECTORY}
                SOURCE  ${CTEST_SOURCE_DIRECTORY}
                OPTIONS "${config_options}"
                APPEND)
ctest_submit(PARTS Update Configure Notes)

ctest_build(BUILD ${CTEST_BINARY_DIRECTORY}
            TARGET install
            APPEND)
ctest_submit(PARTS Build)


