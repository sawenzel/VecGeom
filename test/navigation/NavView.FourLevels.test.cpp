//------------------------------- -*- C++ -*- -------------------------------//
// Copyright VecGeom contributors: see top-level LICENSE file for details
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
//! \file navigation/NavView.test.cpp
//---------------------------------------------------------------------------//
#include "VecGeom/navigation/NavView.h"
#include "NavViewTest.h"

#include <gtest/gtest.h>

namespace vecgeom {
namespace test {
//---------------------------------------------------------------------------//
// TwoBoxes
//---------------------------------------------------------------------------//

/*!
 * Test harness for navigation with transformed children.
 */
using FourLevels = NavViewTest;

TEST_F(FourLevels, InitFromRelocation)
{
  NavData data;
  auto view   = data.make_view();
  auto PathAt = [&](Real3 const &pos) {
    view.Initialize(unknown_path, pos, {0, 0, 1});
    return this->PathStr(view);
  };
  EXPECT_EQ("/World_PV/env1/Shape1_PV/Shape2_PV", PathAt({100, 100, 100}));
  EXPECT_EQ("/", PathAt({10000, -100000, 0}));
  EXPECT_EQ("/World_PV/env5/Shape1_PV", PathAt({-100, -100, 45}));
  EXPECT_EQ("/World_PV", PathAt({0, 0, 0}));
  EXPECT_EQ("/World_PV", PathAt({10, 20, 50}));
  EXPECT_EQ("/World_PV/env8", PathAt({-30, -75, -45}));
}

} // namespace test
} // namespace vecgeom
