#
# CMake Extensions
# ----------------
# macro set_ifnot(<var> <value>)
#       If variable var is not set, set its value to that provided
#
# function enum_option(<option>
#                      VALUES <value1> ... <valueN>
#                      TYPE   <valuetype>
#                      DOC    <docstring>
#                      [DEFAULT <elem>]
#                      [CASE_INSENSITIVE])
#          Declare a cache variable <option> that can only take values
#          listed in VALUES. TYPE may be FILEPATH, PATH or STRING.
#          <docstring> should describe that option, and will appear in
#          the interactive CMake interfaces. If DEFAULT is provided,
#          <elem> will be taken as the zero-indexed element in VALUES
#          to which the value of <option> should default to if not
#          provided. Otherwise, the default is taken as the first
#          entry in VALUES. If CASE_INSENSITIVE is present, then
#          checks of the value of <option> against the allowed values
#          will ignore the case when performing string comparison.
#
#

include_guard(GLOBAL)

#-----------------------------------------------------------------------
# CMAKE EXTENSIONS
#-----------------------------------------------------------------------
# macro set_ifnot(<var> <value>)
#       If variable var is not set, set its value to that provided
#
macro(set_ifnot _var _value)
  if(NOT ${_var})
    set(${_var} ${_value})
  endif()
endmacro()

# .. command:: vecgeom_set_default
#
#   Set a value for the given variable if it is undefined. If a
#   third argument is given and VecGeom is the top-level project, create a cache
#   variable with the given documentation; otherwise, set the variable locally
#   so it's scoped only to vecgeom. ::
#
#      vecgeom_set_default(<variable> <value>)
#      vecgeom_set_default(<variable> <value> <doc>)
#      vecgeom_set_default(<variable> <value> <type> <doc>)
macro(vecgeom_set_default name value)
  if(NOT DEFINED ${name})
    message(VERBOSE "VecGeom: set default ${name}=${value}")
    if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME AND "${ARGC}" GREATER 2)
      if("${ARGC}" EQUAL 3)
        option(${name} "${ARGV2}" "${value}")
      else()
        set(${name} "${value}" CACHE ${ARGN})
      endif()
    else()
      set(${name} "${value}")
    endif()
  endif()
endmacro()

#-----------------------------------------------------------------------
# function enum_option(<option>
#                      VALUES <value1> ... <valueN>
#                      TYPE   <valuetype>
#                      DOC    <docstring>
#                      [DEFAULT <elem>]
#                      [CASE_INSENSITIVE])
#          Declare a cache variable <option> that can only take values
#          listed in VALUES. TYPE may be FILEPATH, PATH or STRING.
#          <docstring> should describe that option, and will appear in
#          the interactive CMake interfaces. If DEFAULT is provided,
#          <elem> will be taken as the zero-indexed element in VALUES
#          to which the value of <option> should default to if not
#          provided. Otherwise, the default is taken as the first
#          entry in VALUES. If CASE_INSENSITIVE is present, then
#          checks of the value of <option> against the allowed values
#          will ignore the case when performing string comparison.
#
function(enum_option _var)
  set(options CASE_INSENSITIVE)
  set(oneValueArgs DOC TYPE DEFAULT)
  set(multiValueArgs VALUES)
  cmake_parse_arguments(_ENUMOP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # - Validation as needed arguments
  if(NOT _ENUMOP_VALUES)
    message(FATAL_ERROR "enum_option must be called with non-empty VALUES\n(Called for enum_option '${_var}')")
  endif()

  # - Set argument defaults as needed
  if(_ENUMOP_CASE_INSENSITIVE)
    set(_ci_values )
    foreach(_elem ${_ENUMOP_VALUES})
      string(TOLOWER "${_elem}" _ci_elem)
      list(APPEND _ci_values "${_ci_elem}")
    endforeach()
    set(_ENUMOP_VALUES ${_ci_values})
  endif()

  set_ifnot(_ENUMOP_TYPE STRING)
  set_ifnot(_ENUMOP_DEFAULT 0)
  list(GET _ENUMOP_VALUES ${_ENUMOP_DEFAULT} _default)

  if(NOT DEFINED ${_var})
    set(${_var} ${_default} CACHE ${_ENUMOP_TYPE} "${_ENUMOP_DOC} (${_ENUMOP_VALUES})")
  else()
    set(_var_tmp ${${_var}})
    if(_ENUMOP_CASE_INSENSITIVE)
      string(TOLOWER ${_var_tmp} _var_tmp)
    endif()

    list(FIND _ENUMOP_VALUES ${_var_tmp} _elem)
    if(_elem LESS 0)
      message(FATAL_ERROR "Value '${${_var}}' for variable ${_var} is not allowed\nIt must be selected from the set: ${_ENUMOP_VALUES} (DEFAULT: ${_default})\n")
    else()
      # - convert to lowercase
      if(_ENUMOP_CASE_INSENSITIVE)
        set(${_var} ${_var_tmp} CACHE ${_ENUMOP_TYPE} "${_ENUMOP_DOC} (${_ENUMOP_VALUES})" FORCE)
      endif()
    endif()
  endif()
  if(_ENUMOP_TYPE STREQUAL "STRING")
    set_property(CACHE ${_var} PROPERTY STRINGS "${_ENUMOP_VALUES}")
  endif()
endfunction()

# - Support for colors in cmake message.
# Credits: https://stackoverflow.com/questions/18968979/how-to-make-colorized-message-with-cmake
#   Use:
#     message("This is normal")
#     message("${Red}This is Red${ColorReset}")
string(ASCII 27 Esc)
set(ColorReset  "${Esc}[m")
set(ColorBold   "${Esc}[1m")
set(Red         "${Esc}[31m")
set(Green       "${Esc}[32m")
set(Yellow      "${Esc}[33m")
set(Blue        "${Esc}[34m")
set(Magenta     "${Esc}[35m")
set(Cyan        "${Esc}[36m")
set(White       "${Esc}[37m")
set(BoldRed     "${Esc}[1;31m")
set(BoldGreen   "${Esc}[1;32m")
set(BoldYellow  "${Esc}[1;33m")
set(BoldBlue    "${Esc}[1;34m")
set(BoldMagenta "${Esc}[1;35m")
set(BoldCyan    "${Esc}[1;36m")
set(BoldWhite   "${Esc}[1;37m")
