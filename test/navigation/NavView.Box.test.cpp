//------------------------------- -*- C++ -*- -------------------------------//
// Copyright VecGeom contributors: see top-level LICENSE file for details
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
//! \file navigation/NavView.test.cpp
//---------------------------------------------------------------------------//
#include "VecGeom/navigation/NavView.h"
#include "NavViewTest.h"

#include <gtest/gtest.h>
#include <stdexcept>

namespace vecgeom {
namespace test {
//---------------------------------------------------------------------------//
// TwoBoxes
//---------------------------------------------------------------------------//

/*!
 * Test harness for basic navigation.
 *
 * This is two nested boxes, with the GDML units in cm.
 */
class Box : public NavViewTest {
  UnitLength GetUnitLength() const final { return UnitLength::cm; }
};

TEST_F(Box, InitFromRelocation)
{
  NavData data;
  auto view = data.make_view();
  // Uninitialized data is expected to be outside
  EXPECT_TRUE(view.IsOutside());

  // Initialize inside the inner box
  view.Initialize({1, 2, 3}, {0, 0, 1});
  EXPECT_FALSE(view.IsOutside());
  EXPECT_TRUE(data.next.IsOutside()); // TODO: replace with view.HasNextState() ?
  EXPECT_EQ(Real3(1, 2, 3), view.Position());
  EXPECT_EQ(Real3(0, 0, 1), view.Direction());
  EXPECT_EQ(LvId{0}, view.GetLogicalVolumeId());
  EXPECT_EQ(PvId{0}, view.GetPlacedVolumeId());
  EXPECT_EQ("BOX", this->LvStr(view));
  EXPECT_EQ("/TOP_PV/BOX_1", this->PathStr(view));

  // Initialize outside
  view.Initialize({1000, 2, 3}, {0, 0, -1});
  EXPECT_TRUE(view.IsOutside());
  EXPECT_EQ("/", this->PathStr(view));
  EXPECT_THROW(view.GetLogicalVolumeId(), std::runtime_error);
  EXPECT_THROW(view.GetPlacedVolumeId(), std::runtime_error);
}

TEST_F(Box, InitFromOpaquePath)
{
  NavData data;
  auto view = data.make_view();

  NavView::OpaquePath opaque_path;
  // World is always stored at index 1; and navtuple supports implicit conversion from navindex
  opaque_path = NavIndex_t{1};

  view.Initialize(opaque_path, {25, 2, 3}, {0, 0, 1});
  EXPECT_FALSE(view.IsOutside());
  EXPECT_TRUE(data.next.IsOutside()); // TODO: replace with view.HasNextState() ?
  EXPECT_EQ(opaque_path, view.GetOpaquePath());
  EXPECT_EQ(Real3(25, 2, 3), view.Position());
  EXPECT_EQ(Real3(0, 0, 1), view.Direction());

  EXPECT_EQ(LvId{1}, view.GetLogicalVolumeId());
  EXPECT_EQ("TOP", this->LvStr(view));
  EXPECT_EQ("/TOP_PV", this->PathStr(view));
}

} // namespace test
} // namespace vecgeom
