//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VecGeom/management/Logger.h
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <utility>

#include "LoggerTypes.h"
#include "LoggerMessage.h"
#include "NullLoggerMessage.h"

//---------------------------------------------------------------------------//
// MACROS
//---------------------------------------------------------------------------//
//! Inject the source code provenance (current file and line)
#define VECGEOM_CODE_PROVENANCE \
  ::vecgeom::Provenance         \
  {                             \
    __FILE__, __LINE__          \
  }

/*!
 * \def VECGEOM_LOG
 *
 * Return a LogMessage object for streaming into at the given level. The
 * regular \c VECGEOM_LOG call is for code paths that happen uniformly in
 * parallel.
 *
 * The logger will only format and print messages. It is not responsible
 * for cleaning up the state or exiting an app.
 *
 * \code
 VECGEOM_LOG(debug) << "Don't print this in general";
 VECGEOM_LOG(warning) << "You may want to reconsider your life choices";
 VECGEOM_LOG(critical) << "Caught a fatal exception: " << e.what();
 * \endcode
 */
#define VECGEOM_LOG(LEVEL) ::vecgeom::logger()(VECGEOM_CODE_PROVENANCE, ::vecgeom::LogLevel::LEVEL)

//---------------------------------------------------------------------------//
/*!
 * \def VECGEOM_LOG_LOCAL
 *
 * Like \c VECGEOM_LOG but for code paths that may only happen on a single
 * process. Use sparingly.
 */
#define VECGEOM_LOG_LOCAL(LEVEL) ::vecgeom::self_logger()(VECGEOM_CODE_PROVENANCE, ::vecgeom::LogLevel::LEVEL)

// Allow VECGEOM_LOGto be present (but ignored) in device code
#ifdef __CUDA_ARCH__
#undef VECGEOM_LOG
#define VECGEOM_LOG(LEVEL) ::vecgeom::detail::NullLoggerMessage()
#undef VECGEOM_LOG_LOCAL
#define VECGEOM_LOG_LOCAL(LEVEL) ::vecgeom::detail::NullLoggerMessage()
#endif

namespace vecgeom {

//---------------------------------------------------------------------------//
/*!
 * Manage logging in serial and parallel.
 *
 * This should generally be called by the \c world_logger and \c
 * self_logger functions below. The call \c operator() returns an object that
 * should be streamed into in order to create a log message.
 *
 * This object \em is assignable, so to replace the default log handler with a
 * different one, you can call \code
   world_logger = Logger(MpiCommunicator::comm_world(), my_handler);
 * \endcode
 */
class Logger {
public:
  //!@{
  //! \name Type aliases
  using Message = detail::LoggerMessage;
  //!@}

public:
  //! Get the default log level
  static constexpr LogLevel default_level() { return LogLevel::status; }

  // Construct with default communicator
  explicit Logger(LogHandler handle);

  // Create a logger that flushes its contents when it destructs
  inline Message operator()(Provenance prov, LogLevel lev);

  //! Set the minimum logging verbosity
  void level(LogLevel lev) { min_level_ = lev; }

  //! Get the current logging verbosity
  LogLevel level() const { return min_level_; }

private:
  LogHandler handle_;
  LogLevel min_level_{default_level()};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
//! Create a logger that flushes its contents when it destructs
auto Logger::operator()(Provenance prov, LogLevel lev) -> Message
{
  LogHandler *handle = nullptr;
  if (handle_ && lev >= min_level_) {
    handle = &handle_;
  }
  return {handle, std::move(prov), lev};
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Get the log level from an environment variable
LogLevel log_level_from_env(std::string const &);

// Create a logger with reasonable default behaviors.
Logger make_default_logger();

// Main VecGeom logger
Logger &logger();

//---------------------------------------------------------------------------//
} // namespace vecgeom
