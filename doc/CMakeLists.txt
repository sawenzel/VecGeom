# - Build Doxygen documentation if doxygen available
find_package(Doxygen 1.10 QUIET)
if(NOT DOXYGEN_FOUND)
  return()
endif()

# - Organised as per https://www.doxygen.nl/manual/config.html
# - Project related options
set(DOXYGEN_PROJECT_LOGO "${PROJECT_SOURCE_DIR}/vecgeom_logo_blue_small.png")
set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/doxygen")
set(DOXYGEN_STRIP_FROM_PATH "${PROJECT_SOURCE_DIR}")
set(DOXYGEN_STRIP_FROM_INC_PATH "${PROJECT_SOURCE_DIR}")

# Allow autobrief for JavaDoc (/**) and Qt (/*!, //!) comments to
# reduce need to use special commands like @brief and help readability
set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
set(DOXYGEN_QT_AUTOBRIEF YES)
set(DOXYGEN_MULTILINE_CPP_IS_BRIEF YES)
set(DOXYGEN_TAB_SIZE 2)
set(DOXYGEN_EXTENSION_MAPPING "cu=C++")

# - Build related configuration options
set(DOXYGEN_SHOW_USED_FILES NO)

# - Configuration options related to the input files
set(DOXYGEN_FILE_PATTERNS
  "*.h"
  "*.hh"
  "*.cpp"
  "*.cu"
  "*.md"
)
set(DOXYGEN_EXCLUDE_PATTERNS
  "vecgeom::evolution" 
  "vecgeom::cuda"
  "vecgeom::cxx"
  "vecgeom::Bench*"
  "vecgeom::kCuda*" 
  "Alignment*"
  "Polyhedra*"
  "VECGEOM_NAMESPACE"
)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${PROJECT_SOURCE_DIR}/README.md")

# - Configuration options related to the HTML output
# - Configuration options related to the preprocessor
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_EXPAND_ONLY_PREDEF YES)
set(DOXYGEN_PREDEFINED
  "VECGEOM_IMPL_NAMESPACE=cxx"
  "VECCORE_CUDA_DEVICE_COMPILATION"
  "CROSSCHECK"
  "VERBOSE"
)

# Call CMake function from FindDoxygen to generate
doxygen_add_docs(doxygen
  "${PROJECT_SOURCE_DIR}/README.md"
  "${PROJECT_SOURCE_DIR}/VecGeom"
  "${PROJECT_SOURCE_DIR}/persistency"
  "${PROJECT_SOURCE_DIR}/source"
  "${PROJECT_BINARY_DIR}/VecGeom/base/Version.h"
  "${PROJECT_BINARY_DIR}/VecGeom/base/Config.h"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Processing C++ documentation with Doxygen"
)  
