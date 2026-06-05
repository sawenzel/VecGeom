//------------------------------- -*- C++ -*- -------------------------------//
// Copyright VecGeom contributors: see top-level LICENSE file for details
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
//! \file navigation/NavView.test.cpp
//---------------------------------------------------------------------------//
#include "VecGeom/navigation/NavView.h"
#include "TestBase.h"
#include "VecGeom/base/Global.h"

#include <gtest/gtest.h>
#include <stdexcept>

namespace vecgeom {
namespace test {
//---------------------------------------------------------------------------//

using NavState = NavView::NavState;
using Real3    = NavView::Real3;
using LvId     = NavView::LogicalVolumeId;
using PvId     = NavView::PlacedVolumeId;

//---------------------------------------------------------------------------//
// NavViewBox
//---------------------------------------------------------------------------//

//! Helper to store navigation data locally
struct NavData {
  NavState cur;
  NavState next{NavIndex_t{1}}; // Initialize with garbage data
  Real3 pos;
  Real3 dir{0, 0, 0};

  //! Return a view to this data
  auto make_view() & { return NavView{cur, next, pos, dir}; }
};

/*!
 * Test harness for basic navigation.
 *
 * This is two nested boxes, with the GDML units in cm.
 */
class NavViewBox : public GdmlTestBase {
public:
  UnitLength GetUnitLength() const final { return UnitLength::cm; }
  std::string GetBasename() const final { return "box"; };
};

TEST_F(NavViewBox, InitFromRelocation)
{
  NavData data;
  auto view = data.make_view();
  // Uninitialized data is expected to be outside
  EXPECT_TRUE(view.IsOutside());

  // Initialize inside the world
  view.Initialize({1, 2, 3}, {0, 0, 1});
  EXPECT_FALSE(view.IsOutside());
  EXPECT_TRUE(data.next.IsOutside()); // TODO: replace with view.HasNextState() ?
  EXPECT_EQ(Real3(1, 2, 3), view.Position());
  EXPECT_EQ(Real3(0, 0, 1), view.Direction());
  EXPECT_EQ(LvId{0}, view.GetLogicalVolumeId());
  EXPECT_EQ(PvId{0}, view.GetPlacedVolumeId());

  // Initialize outside
  view.Initialize({1000, 2, 3}, {0, 0, -1});
  EXPECT_TRUE(view.IsOutside());
  EXPECT_THROW(view.GetLogicalVolumeId(), std::runtime_error);
  EXPECT_THROW(view.GetPlacedVolumeId(), std::runtime_error);
}

TEST_F(NavViewBox, InitFromOpaquePath)
{
  NavData data;
  auto view = data.make_view();

  NavView::OpaquePath opaque_path;
  // World is always stored at index 1; and navtuple supports implicit conversion from navindex
  opaque_path = NavIndex_t{1};

  view.Initialize(opaque_path, {1, 2, 3}, {0, 0, 1});
  EXPECT_FALSE(view.IsOutside());
  EXPECT_TRUE(data.next.IsOutside()); // TODO: replace with view.HasNextState() ?
  EXPECT_EQ(opaque_path, view.GetOpaquePath());
  EXPECT_EQ(Real3(1, 2, 3), view.Position());
  EXPECT_EQ(Real3(0, 0, 1), view.Direction());

  // XXX: how do we correlate these to the GDML volume names?
  EXPECT_EQ(LvId{1}, view.GetLogicalVolumeId());
  EXPECT_EQ(PvId{1}, view.GetPlacedVolumeId());
}

} // namespace test
} // namespace vecgeom
