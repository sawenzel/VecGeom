//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VecGeom/management/NullLoggerMessage.h
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>
#include "VecGeom/base/Global.h"

namespace vecgeom {

namespace detail {
//---------------------------------------------------------------------------//
/*!
 * Stream-like helper class that \em discards everything passed to it.
 *
 * This helper class should simply eat any messages and objects passed to it.
 */
class NullLoggerMessage
{
  public:
    //!@{
    //! \name Type aliases
    using StreamManip = std::ostream& (*)(std::ostream&);
    //!@}

  public:
    // Default constructor.
    NullLoggerMessage() = default;

    //! Do not print this object
    template<class T>
    VECCORE_ATT_HOST_DEVICE NullLoggerMessage& operator<<(T&&)
    {
        return *this;
    }

    //! Ignore this manipulator function
    VECCORE_ATT_HOST_DEVICE NullLoggerMessage& operator<<(StreamManip)
    {
        return *this;
    }

    //! Do not set any state
    VECCORE_ATT_HOST_DEVICE void setstate(std::ostream::iostate) {}
};

//---------------------------------------------------------------------------//
}  // namespace detail
} // namespace vecgeom
