//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VecGeom/management/ColorUtils.h
//! \brief Helper functions for writing colors to the terminal
//---------------------------------------------------------------------------//
#pragma once

namespace vecgeom {

//---------------------------------------------------------------------------//
// Whether colors are enabled (currently read-only)
bool use_color();

//---------------------------------------------------------------------------//
// Get an ANSI color code: [y]ellow / [r]ed / [ ]default / ...
char const *color_code(char abbrev);

//---------------------------------------------------------------------------//
} // namespace vecgeom