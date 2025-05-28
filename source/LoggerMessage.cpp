//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LoggerMessage.cpp
//---------------------------------------------------------------------------//
#include "VecGeom/management/LoggerMessage.h"

#include "VecGeom/base/Assert.h"
#include <exception>
#include <functional>
#include <sstream>

#include "VecGeom/management/Logger.h"
#include "VecGeom/management/LoggerTypes.h"

namespace vecgeom {

namespace detail {
//---------------------------------------------------------------------------//
/*!
 * Construct with reference to function object, etc.
 *
 * The handle *may be* null, indicating that the output of this message will
 * not be displayed.
 */
LoggerMessage::LoggerMessage(LogHandler *handle, Provenance prov, LogLevel lev)
    : handle_(handle), prov_(prov), lev_(lev)
{
  VECGEOM_ASSERT(!handle_ || *handle_);
  if (handle_) {
    // std::function is defined, so create the output stream
    os_ = std::make_unique<std::ostringstream>();
  }
}

//---------------------------------------------------------------------------//
/*!
 * Flush message on destruction.
 */
LoggerMessage::~LoggerMessage()
{
  if (os_) {
    try {
      auto &os = dynamic_cast<std::ostringstream &>(*os_);

      // Write to the handler
      (*handle_)(prov_, lev_, os.str());
    } catch (std::exception const &e) {
      std::cerr << "An error occurred writing a log message: " << e.what() << std::endl;
    }
  }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace vecgeom
