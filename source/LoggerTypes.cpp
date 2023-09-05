//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LoggerTypes.cpp
//---------------------------------------------------------------------------//
#include "VecGeom/management/LoggerTypes.h"

#include <cassert>

namespace vecgeom {

//---------------------------------------------------------------------------//
/*!
 * Get the plain text equivalent of the LogLevel enum.
 */
char const *to_cstring(LogLevel lev)
{
  static const char *const data[] = {
      "debug", "diagnostic", "status", "info", "warning", "error", "critical",
  };
  assert(lev * sizeof(const char *) < sizeof(data));
  return data[static_cast<int>(lev)];
}

//---------------------------------------------------------------------------//
} // namespace vecgeom
