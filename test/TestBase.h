//------------------------------- -*- C++ -*- -------------------------------//
// Copyright VecGeom contributors: see top-level LICENSE file for details
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
//! \file TestBase.h
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <gtest/gtest.h>
#include "VecGeom/volumes/PlacedVolume.h"

namespace vecgeom {
namespace test {
//---------------------------------------------------------------------------//

//! Implementation base class for setting up VecGeom manager etc
class TestBase : public ::testing::Test {
public:
  void SetUp() override;

  virtual void LoadWorld() = 0;

  // GDML or test identifier
  virtual std::string GetBasename() const = 0;

private:
  void SetUpVolumeTracking();
};

/*!
 * Base class to create your own world via placed volume.
 *
 * - Override \c MakeWorld to return a world volume.
 * - The base name is determined by the test suite name: each subclass will try
 *   to recreate a new world.
 */
class CustomTestBase : public TestBase {
public:
  // Build the world volume and set in GeoManager
  void LoadWorld() final;
  // Use test suite name
  std::string GetBasename() const final;

  // Implement this to create custom volumes
  virtual cxx::VPlacedVolume *MakeWorld() = 0;
};

/*!
 * Base class to load a world via VGDML.
 *
 * Implement \c GetBasename to return the name component of a file in \c VecGeom/test/gdml/gdmls .
 *
 * \par Example:
 * \code
 * class BoxTest : public GdmlTestBase
 * {
 * public
 *  std::string GetBasename() final { return "box"; }
 * };
 *
 * TEST_F(BoxTest, tracking)
 * {
 *    // ...
 * }
 *
 * \endcode
 */
class GdmlTestBase : public TestBase {
public:
  //! Unit system for length scale
  enum class UnitLength { mm, cm };

  //! Default length is mm but many test GDML files use cm
  virtual UnitLength GetUnitLength() const { return UnitLength::mm; }

  void LoadWorld() final;
};

//---------------------------------------------------------------------------//
} // namespace test
} // namespace vecgeom
