//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Logger.cpp
//---------------------------------------------------------------------------//
#include "VecGeom/management/Logger.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <cassert>
#include <mutex>
#include <sstream>
#include <string>

#include "VecGeom/management/LoggerTypes.h"
#include "VecGeom/management/Environment.h"
#include "VecGeom/management/ColorUtils.h"

namespace vecgeom {

namespace {
//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//
//! Default global logger prints the error message with basic colors
void default_global_handler(Provenance prov, LogLevel lev, std::string msg)
{
  static std::mutex log_mutex;
  std::lock_guard<std::mutex> scoped_lock{log_mutex};

  if (lev == LogLevel::debug || lev >= LogLevel::warning) {
    // Output problem line/file for debugging or high level
    std::clog << color_code('x') << prov.file;
    if (prov.line) std::clog << ':' << prov.line;
    std::clog << color_code(' ') << ": ";
  }

  // clang-format off
    char c = ' ';
    switch (lev)
    {
        case LogLevel::debug:      c = 'x'; break;
        case LogLevel::diagnostic: c = 'x'; break;
        case LogLevel::status:     c = 'b'; break;
        case LogLevel::info:       c = 'g'; break;
        case LogLevel::warning:    c = 'y'; break;
        case LogLevel::error:      c = 'r'; break;
        case LogLevel::critical:   c = 'R'; break;
        case LogLevel::size_: assert(false);
    };
  // clang-format on
  std::clog << color_code(c) << to_cstring(lev) << ": " << color_code(' ') << msg << std::endl;
}

//---------------------------------------------------------------------------//
/*!
 * Set the log level from an environment variable, warn on failure.
 */
void set_log_level_from_env(Logger *log, std::string const &level_env)
{
  assert(log);
  try {
    log->level(log_level_from_env(level_env));
  } catch (std::runtime_error const &e) {
    (*log)(VECGEOM_CODE_PROVENANCE, LogLevel::warning) << e.what();
  }
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with log handler.
 */
Logger::Logger(LogHandler handle) : handle_{std::move(handle)} {}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get the log level from an environment variable.
 */
LogLevel log_level_from_env(std::string const &level_env)
{
  // Search for the provided environment variable to set the default
  // logging level using the `to_cstring` function in LoggerTypes.
  std::string const &env_value = vecgeom::getenv(level_env);
  if (env_value.empty()) {
    return Logger::default_level();
  }

  for (int i = 0; i < static_cast<int>(LogLevel::size_); ++i) {
    if (env_value == to_cstring(static_cast<LogLevel>(i))) {
      return static_cast<LogLevel>(i);
    }
  }
  throw std::runtime_error("invalid log level in environment variable");
}

//---------------------------------------------------------------------------//
/*!
 * Create a default logger using the world communicator.
 *
 * This function can be useful when resetting a test harness.
 */
Logger make_default_logger()
{
  Logger log{&default_global_handler};
  set_log_level_from_env(&log, "VECGEOM_LOG");
  return log;
}

//---------------------------------------------------------------------------//
/*!
 * Parallel-enabled logger: print only on "main" process.
 *
 * Setting the "VECGEOM_LOG" environment variable to "debug", "info", "error",
 * etc. will change the default log level.
 */
Logger &logger()
{
  static Logger logger = make_default_logger();
  return logger;
}

//---------------------------------------------------------------------------//
} // namespace vecgeom
