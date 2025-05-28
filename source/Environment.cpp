//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Environment.cpp
//---------------------------------------------------------------------------//
#include "VecGeom/management/Environment.h"

#include "VecGeom/base/Assert.h"
#include <cstdlib>
#include <mutex>
#include <iostream>

namespace vecgeom {

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Access a static global environment variable.
 *
 * This static variable should be shared among Celeritas objects.
 */
Environment &environment()
{
  static Environment environ;
  return environ;
}

//---------------------------------------------------------------------------//
/*!
 * Thread-safe access to global modified environment variables.
 */
std::string const &getenv(std::string const &key)
{
  static std::mutex getenv_mutex;
  std::lock_guard<std::mutex> scoped_lock{getenv_mutex};
  return environment()[key];
}

//---------------------------------------------------------------------------//
/*!
 * Write the accessed environment variables to a stream.
 */
std::ostream &operator<<(std::ostream &os, Environment const &env)
{
  os << "{\n";
  for (auto const &kvref : env.ordered_environment()) {
    Environment::value_type const &kv = kvref;
    os << "  " << kv.first << ": '" << kv.second << "',\n";
  }
  os << '}';
  return os;
}

//---------------------------------------------------------------------------//
// MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Set a value from the system environment.
 */
auto Environment::load_from_getenv(key_type const &key) -> mapped_type const &
{
  std::string value;
  if (char const *sys_value = std::getenv(key.c_str())) {
    // Variable is set in the user environment
    value = sys_value;
  }

  // Insert value and ordering. Note that since the elements are never
  // erased, pointers to the keys are guaranteed to always be valid.
  auto [iter, inserted] = vars_.emplace(key, std::move(value));
  VECGEOM_ASSERT(inserted);
  ordered_.push_back(std::ref(*iter));

  VECGEOM_ASSERT(ordered_.size() == vars_.size());
  return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Set a single environment variable that hasn't yet been set.
 *
 * Existing environment variables will *not* be overwritten.
 */
void Environment::insert(value_type const &value)
{
  auto [iter, inserted] = vars_.insert(value);
  if (inserted) {
    ordered_.push_back(std::ref(*iter));
  }
  VECGEOM_ASSERT(ordered_.size() == vars_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Remove all entries.
 */
void Environment::clear()
{
  vars_.clear();
  ordered_.clear();
}

//---------------------------------------------------------------------------//
/*!
 * Insert but don't override from another environment.
 */
void Environment::merge(Environment const &other)
{
  for (auto const &kv : other.ordered_environment()) {
    this->insert(kv);
  }
}

//---------------------------------------------------------------------------//
} // namespace vecgeom
