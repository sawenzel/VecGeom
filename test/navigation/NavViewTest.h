//------------------------------- -*- C++ -*- -------------------------------//
// Copyright VecGeom contributors: see top-level LICENSE file for details
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
//! \file navigation/NavViewTest.h
//---------------------------------------------------------------------------//
#include "VecGeom/navigation/NavView.h"
#include "TestBase.h"

namespace vecgeom {
namespace test {
//---------------------------------------------------------------------------//

using NavState = NavView::NavState;
using Real3    = NavView::Real3;
using LvId     = NavView::LogicalVolumeId;
using PvId     = NavView::PlacedVolumeId;

//! Helper to store navigation data locally
struct NavData {
  NavState cur;
  NavState next{NavIndex_t{1}}; // Initialize with garbage data
  Real3 pos;
  Real3 dir{0, 0, 0};

  //! Return a view to this data
  auto make_view() & { return NavView{cur, next, pos, dir}; }
};

//---------------------------------------------------------------------------//

/*!
 * Look for tests inside {source}/test/navigation/data/{suitename}.gdml .
 *
 * NOTE: GDML name is inferred from test suite via TestBase::GetBasename ,
 * and default units are cm.
 */
class NavViewTest : public GdmlTestBase {
public:
  std::string GetGdmlDir() const final { return "navigation/data"; };
};

} // namespace test
} // namespace vecgeom
