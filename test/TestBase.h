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
class NavView;

namespace test {
//---------------------------------------------------------------------------//

//! Implementation base class for setting up VecGeom manager etc
class TestBase : public ::testing::Test {
public:
  void SetUp() override;

  virtual void LoadWorld() = 0;

  // GDML or test identifier: use test harness name as default
  virtual std::string GetBasename() const;

  // Get the "top" logical volume from a navigation view
  std::string LvStr(NavView const &) const;

  // Get a slash-joined path string from a nav view's state
  std::string PathStr(NavView const &) const;

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

  // Implement this to create custom volumes
  virtual cxx::VPlacedVolume *MakeWorld() = 0;
};

/*!
 * Base class to load a world via VGDML.
 *
 * Implement \c GetBasename to return the name component of a file in \c VecGeom/test/{GetGdmlDir}/{GetBasename}.gdml .
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

  //! GDML base directory relative to test dir
  virtual std::string GetGdmlDir() const { return "gdml/gdmls"; }

  void LoadWorld() final;
};

//---------------------------------------------------------------------------//
} // namespace test
} // namespace vecgeom
