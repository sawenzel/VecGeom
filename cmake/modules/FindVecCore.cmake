#[=======================================================================[.rst:

FindVecCore
-----------

Find the vector core library and print information about its location, version, and features.

#]=======================================================================]

find_package(VecCore QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VecCore CONFIG_MODE HANDLE_COMPONENTS)

#-----------------------------------------------------------------------------#
