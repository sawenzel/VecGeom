if(NOT DEFINED VECGEOM_CDASH_SUBMIT_RETRY_COUNT)
  set(VECGEOM_CDASH_SUBMIT_RETRY_COUNT 3)
endif()

if(NOT DEFINED VECGEOM_CDASH_SUBMIT_RETRY_DELAY)
  set(VECGEOM_CDASH_SUBMIT_RETRY_DELAY 30)
endif()

function(vecgeom_ctest_submit)
  ctest_submit(${ARGN}
               RETRY_COUNT ${VECGEOM_CDASH_SUBMIT_RETRY_COUNT}
               RETRY_DELAY ${VECGEOM_CDASH_SUBMIT_RETRY_DELAY}
               RETURN_VALUE submit_result
               CAPTURE_CMAKE_ERROR submit_error)

  if(submit_error OR NOT submit_result EQUAL 0)
    message(WARNING
      "CDash submission failed for '${ARGN}' "
      "(result=${submit_result}, cmake_error=${submit_error}); continuing.")
  endif()
endfunction()
