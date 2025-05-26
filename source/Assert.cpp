//------------------------------- -*- C++ -*- -------------------------------//
//! \file Assert.cpp
//---------------------------------------------------------------------------//

#include "VecGeom/base/Assert.h"

#include <stdexcept>
#include <string>

namespace vecgeom {

[[nodiscard]] std::logic_error make_debug_error(char const *which, char const *condition, char const *file, int line)
{
  std::string msg;
  msg += which;
  msg += ": ";
  msg += condition;
  msg += " failed at ";
  msg += file;
  msg += ":";
  msg += std::to_string(line);
  return std::logic_error(std::move(msg));
}

[[nodiscard]] std::runtime_error make_runtime_error(char const *which, char const *what, char const *condition,
                                                    char const *file, int line)
{
  std::string msg;
  if (which) {
    msg += which;
  } else {
    msg += "unknown";
  }
  msg += " error: ";
  msg += what;
  msg += ": ";
  if (condition) {
    msg += condition;
    msg += " failed";
  }
  msg += " at ";
  msg += file;
  msg += ":";
  msg += std::to_string(line);
  return std::runtime_error(std::move(msg));
}

} // namespace vecgeom