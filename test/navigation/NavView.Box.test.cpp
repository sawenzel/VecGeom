//------------------------------- -*- C++ -*- -------------------------------//
// Copyright VecGeom contributors: see top-level LICENSE file for details
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
//! \file navigation/NavView.test.cpp
//---------------------------------------------------------------------------//
#include "VecGeom/navigation/NavView.h"
#include "NavViewTest.h"
#include "VecGeom/base/Math.h"
#include "VecGeom/navigation/BVHNavigator.h"

#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

namespace vecgeom {
namespace test {

using NFRKind = NavFindResultKind;

TEST(NavFindResult, default_ctor)
{
  NavFindResult nfr;
  EXPECT_EQ(NFRKind::miss, nfr.GetKind());
  if (VECGEOM_DEBUG) {
    EXPECT_THROW(nfr.GetDistance(), std::logic_error);
  }
}

TEST(NavFindResult, error_ctor)
{
  NavFindResult nfr{NFRKind::error};
  EXPECT_EQ(NFRKind::error, nfr.GetKind());
  if (VECGEOM_DEBUG) {
    EXPECT_THROW(nfr.GetDistance(), std::logic_error);
  }
}

TEST(NavFindResult, hit_ctor)
{
  NavFindResult nfr{1.25};
  EXPECT_EQ(NFRKind::hit, nfr.GetKind());
  EXPECT_DOUBLE_EQ(1.25, nfr.GetDistance());
  if (VECGEOM_DEBUG) {
    // Cannot construct with nonpositive values
    EXPECT_THROW(NavFindResult{0.0}, std::logic_error);
    EXPECT_THROW(NavFindResult{-1234.0}, std::logic_error);
  }
}

//---------------------------------------------------------------------------//
// Basic box tests
//---------------------------------------------------------------------------//

/*!
 * Test harness for basic navigation.
 *
 * This is two nested boxes, with the GDML units in cm.
 */
class Box : public NavViewTest {
  UnitLength GetUnitLength() const final { return UnitLength::cm; }
  std::string GetBasename() const final { return "Box"; }
};

TEST_F(Box, InitFromRelocation)
{
  NavData data;
  auto view = data.make_view();
  // Uninitialized data is expected to be outside
  EXPECT_TRUE(view.IsOutside());
  EXPECT_FALSE(view.IsOnBoundary());

  // Non-unit vector should be an error
  EXPECT_THROW(view.Initialize(unknown_path, {1, 2, 3}, {0, 0, 1.001}), std::runtime_error);

  // Initialize inside the inner box
  view.Initialize(unknown_path, {1, 2, 3}, {0, 0, 1});
  EXPECT_FALSE(view.IsOutside());
  EXPECT_TRUE(data.next.IsOutside()); // TODO: replace with view.HasNextState() ?
  EXPECT_EQ(Real3(1, 2, 3), view.GetPosition());
  EXPECT_EQ(Real3(0, 0, 1), view.GetDirection());
  EXPECT_EQ(LvId{0}, view.GetLogicalVolumeId());
  EXPECT_EQ(PvId{0}, view.GetPlacedVolumeId());
  EXPECT_EQ("BOX", this->LvStr(view));
  EXPECT_EQ("/TOP_PV/BOX_1", this->PathStr(view));

  // Initialize on boundary between inner and world, into, along, and away from the face
  for (auto dir : {Real3{-1, 0, 0}, Real3{0, 0, 1}, Real3{1, 0, 0}}) {
    view.Initialize(unknown_path, {20, 0, 0}, dir);
    EXPECT_TRUE(view.IsOnBoundary());
    EXPECT_EQ("/TOP_PV/BOX_1", this->PathStr(view));
  }

  // Initialize outside
  view.Initialize(unknown_path, {1000, 2, 3}, {0, 0, -1});
  EXPECT_TRUE(view.IsOutside());
  EXPECT_EQ("/", this->PathStr(view));
  EXPECT_THROW(view.GetLogicalVolumeId(), std::runtime_error);
  EXPECT_THROW(view.GetPlacedVolumeId(), std::runtime_error);

  // Initialize on boundary of world: into, parallel to, and away from the face
  for (auto dir : {Real3{-1, 0, 0}, Real3{0, 0, 1}, Real3{1, 0, 0}}) {
    view.Initialize(unknown_path, {100, 0, 0}, dir);
    EXPECT_FALSE(view.IsOutside());
    EXPECT_TRUE(view.IsOnBoundary());
    EXPECT_EQ("/TOP_PV", this->PathStr(view));
  }
}

TEST_F(Box, InitFromOpaquePath)
{
  NavData data;
  auto view = data.make_view();

  // Initialize in world
  view.Initialize({NavView::GetWorldPath()}, {25, 2, 3}, {0, 0, 1});
  EXPECT_FALSE(view.IsOutside());
  EXPECT_FALSE(view.IsOnBoundary());
  EXPECT_TRUE(data.next.IsOutside()); // TODO: replace with view.HasNextState() ?
  EXPECT_EQ(NavView::GetWorldPath(), view.GetOpaquePath());
  EXPECT_EQ(Real3(25, 2, 3), view.GetPosition());
  EXPECT_EQ(Real3(0, 0, 1), view.GetDirection());

  EXPECT_EQ("TOP", this->LvStr(view));
  EXPECT_EQ("/TOP_PV", this->PathStr(view));

  // Initialize on boundary headed into box
  view.Initialize({NavView::GetWorldPath(), true}, {20, 0, 0}, {-1, 0, 0});
  EXPECT_EQ("/TOP_PV", this->PathStr(view));
  EXPECT_TRUE(view.IsOnBoundary());

  // Allow initialization in outside state
  view.Initialize({NavView::GetOutsidePath()}, {1000, 2, 3}, {0, 0, 1});
  EXPECT_EQ("/", this->PathStr(view));
  EXPECT_FALSE(view.IsOnBoundary());
}

TEST_F(Box, FindSafety)
{
  NavData data;
  auto view = data.make_view();

  // Initialize on boundary headed into box
  view.Initialize({NavView::GetWorldPath(), true}, {20, 0, 0}, {-1, 0, 0});
  // Cannot check safety when on boundary
  EXPECT_THROW(view.FindSafety(100), std::runtime_error);

  // Logically off boundary but still on boundary
  view.Initialize({NavView::GetWorldPath(), false}, {20, 0, 0}, {-1, 0, 0});
  EXPECT_DOUBLE_EQ(0.0, view.FindSafety(100));
  // Near boundary
  view.Initialize({NavView::GetWorldPath(), false}, {20 + 1e-3, 0, 0}, {-1, 0, 0});
  EXPECT_NEAR(1e-3, view.FindSafety(100), 1e-12);
  // Far from boundary
  view.Initialize({NavView::GetWorldPath(), false}, {25, 0, 0}, {-1, 0, 0});
  EXPECT_DOUBLE_EQ(5, view.FindSafety(100));
  // Far from boundary with limit
  view.Initialize({NavView::GetWorldPath(), false}, {25, 0, 0}, {-1, 0, 0});
  EXPECT_DOUBLE_EQ(1.25, view.FindSafety(1.25));
  // On wrong side of boundary due to numerical imprecision
  view.Initialize({NavView::GetWorldPath(), false}, {20 - 1e-3, 0, 0}, {-1, 0, 0});
  EXPECT_DOUBLE_EQ(0.0, view.FindSafety(100));
}

//---------------------------------------------------------------------------//

/*!
 * Test harness for basic navigation.
 *
 * This is two nested boxes, with the GDML units in cm.
 */
class BoxFindNextBoundary : public Box {
public:
  void SetUp() override
  {
    Box::SetUp();

    // Save inner/outer paths
    auto view = data_.make_view();

    view.Initialize(unknown_path, {0, 0, 0}, {0, 0, 1});
    inner_path_ = view.GetOpaquePath();
    outer_path_ = view.GetWorldPath();
  }

  std::string CompareResult(NavFindResult const &nfr, double distance) const
  {
    if (nfr.GetKind() != NavFindResultKind::hit) {
      return ToString(nfr.GetKind());
    }
    auto delta = nfr.GetDistance() - distance;
    if (std::fabs(delta) < 1e-10) {
      return "hit";
    } else {
      return std::to_string(delta);
    }
  };

protected:
  NavData data_;
  OpaqueNavPath inner_path_;
  OpaqueNavPath outer_path_;
};

TEST_F(BoxFindNextBoundary, error)
{
  auto view = data_.make_view();
  view.Initialize({view.GetOutsidePath()}, {1000, 0, 0}, {1, 0, 0});

  // Cannot search when outside the geometry
  EXPECT_THROW(view.FindNextBoundary(100), std::runtime_error);
}

TEST_F(BoxFindNextBoundary, interior)
{
  auto view = data_.make_view();

  // Note: box halfwidth=20cm
  view.Initialize({inner_path_}, {10, 0, 0}, {1, 0, 0});
  // Boundary is less than limit
  auto nfr = view.FindNextBoundary(1000);
  ASSERT_EQ(NFRKind::hit, nfr.GetKind());
  EXPECT_DOUBLE_EQ(10.0, nfr.GetDistance());
  // Boundary is beyond limit
  nfr = view.FindNextBoundary(1);
  EXPECT_EQ(NFRKind::miss, nfr.GetKind());
  ;
  // Boundary exactly limit
  nfr = view.FindNextBoundary(10.0);
  EXPECT_EQ(NFRKind::miss, nfr.GetKind()); // !!! FIXME !!! should be hit
  // EXPECT_DOUBLE_EQ(10.0, nfr.GetDistance());
  // Boundary epsilon beyond limit
  nfr = view.FindNextBoundary(10.0 - 1e-12);
  EXPECT_EQ(NFRKind::miss, nfr.GetKind());
}

TEST_F(BoxFindNextBoundary, edge_cases)
{
  // Test potential overlap/error/start-on-boundary situations
  // Logically on boundary between box+world, physically near it:
  // - X position: location of face
  // - X normal: outward normal of face
  // - Next: distance to next boundary inside volume along normal
  struct {
    char const *const label{};
    OpaqueNavPath inside{};
    OpaqueNavPath outside{};
    double xpos{};
    double xnorm{};
    double next{};
  } attempts[] = {
      {"left inward", inner_path_, outer_path_, -20, 1, 40},
      {"right inward", inner_path_, outer_path_, 20, -1, 40},
      {"left outward", outer_path_, inner_path_, -20, -1, 80},
      {"right outward", outer_path_, inner_path_, 20, 1, 80},
  };

  for (auto const &v : attempts) {
    SCOPED_TRACE(v.label);
    auto view = data_.make_view();

    // Exactly on boundary, entering face (reentrant)
    view.Initialize({v.inside, true}, {v.xpos, 0, 0}, {-v.xnorm, 0, 0});
    auto nfr = view.FindNextBoundary(1.0);
    EXPECT_EQ("reentrant", CompareResult(nfr, v.next)) << nfr;

    // Exactly on boundary, headed away from boundary
    view.Initialize({v.inside, true}, {v.xpos, 0, 0}, {v.xnorm, 0, 0});
    nfr = view.FindNextBoundary(v.next + 1.0);
    EXPECT_EQ("hit", CompareResult(nfr, v.next)) << nfr;

    // Well away from boundary, headed away from boundary
    double dx = 0.1;
    view.Initialize({v.inside, false}, {v.xpos + v.xnorm * dx, 0, 0}, {v.xnorm, 0, 0});
    nfr = view.FindNextBoundary(v.next / 2);
    EXPECT_EQ("miss", CompareResult(nfr, v.next - dx)) << nfr;
    nfr = view.FindNextBoundary(v.next * 2);
    EXPECT_EQ("hit", CompareResult(nfr, v.next - dx)) << nfr;

    // Epsilon away from boundary
    for (auto tol : {kTolerance, 10 * kTolerance, kPushTolerance, kSqrtTolerance,
                     static_cast<double>(kToleranceDist<float>), kToleranceDist<double>}) {
      for (auto frac : {0.9, 1.0, 1.1}) {
        dx = tol * frac;
        SCOPED_TRACE(testing::Message{} << "dx=" << dx);

        // Epsilon past boundary, headed into interior
        for (auto start_on_bnd : {false, true}) {
          SCOPED_TRACE(start_on_bnd ? "on boundary" : "interior");
          view.Initialize({v.inside, start_on_bnd}, {v.xpos + v.xnorm * dx, 0, 0}, {v.xnorm, 0, 0});
          nfr = view.FindNextBoundary(v.next + 1.0);
          EXPECT_EQ("hit", CompareResult(nfr, v.next - dx)) << nfr;
        }

        // Epsilon before boundary, headed into interior: should ignore face being crossed
        view.Initialize({v.inside, true}, {v.xpos - v.xnorm * dx, 0, 0}, {v.xnorm, 0, 0});
        nfr = view.FindNextBoundary(v.next + 1.0);
        if (dx <= BVHNavigator::kBoundaryPush) {
          EXPECT_EQ("hit", CompareResult(nfr, v.next + dx)) << nfr;
        } else {
          // !!! FIXME !!!: 'max' in BVH navigator forces the return distance to be zero
          // should be "error"
          EXPECT_EQ("reentrant", CompareResult(nfr, v.next + dx)) << nfr;
        }
        // std::cout << v.label << ": " << tol << " * " << frac << " -> dx=" << dx << ": "
        // << compare_result(nfr, v.next + dx) << std::endl;
      }
    }
  }
}

using BoxCrossBoundary = BoxFindNextBoundary;

TEST_F(BoxCrossBoundary, errors)
{
  auto view = data_.make_view();

  // Cannot cross when outside/uninitialized
  EXPECT_THROW(view.CrossBoundary(), std::runtime_error);

  // Not on a boundary
  view.Initialize({inner_path_}, {0, 0, 0}, {1, 0, 0});
  ASSERT_FALSE(view.IsOnBoundary());
  EXPECT_THROW(view.CrossBoundary(), std::runtime_error);

  // On boundary but without "next"
  view.Initialize({inner_path_, true}, {20, 0, 0}, {1, 0, 0});
  ASSERT_TRUE(view.IsOnBoundary());
  EXPECT_THROW(view.CrossBoundary(), std::runtime_error);

  // Boundary not found
  view.Initialize({inner_path_, false}, {10, 0, 0}, {1, 0, 0});
  auto nfr = view.FindNextBoundary(5);
  ASSERT_EQ(NFRKind::miss, nfr.GetKind());
  EXPECT_THROW(view.CrossBoundary(), std::runtime_error);

  // Boundary found but not moved to it
  nfr = view.FindNextBoundary(20);
  ASSERT_EQ(NFRKind::hit, nfr.GetKind());
  EXPECT_THROW(view.CrossBoundary(), std::runtime_error);
}

//! Demonstrate standard usage without any edge cases
TEST_F(BoxCrossBoundary, with_movement)
{
  // Initialize
  auto view = data_.make_view();
  view.Initialize(unknown_path, {0, 0, 0}, {1, 0, 0});
  EXPECT_EQ(inner_path_, view.GetOpaquePath());

  // Physics distance is small
  auto nfr = view.FindNextBoundary(5);
  EXPECT_EQ(NFRKind::miss, nfr.GetKind());
  view.MoveInternal(5);
  EXPECT_EQ(Real3(5, 0, 0), view.GetPosition());
  // Scatter to +y
  view.ChangeDirection({0, 1, 0});
  EXPECT_EQ(Real3(0, 1, 0), view.GetDirection());
  EXPECT_EQ(Real3(0, 1, 0), data_.dir);

  // Boundary is closer: move and cross
  nfr = view.FindNextBoundary(40);
  ASSERT_EQ(NFRKind::hit, nfr.GetKind());
  EXPECT_DOUBLE_EQ(30, nfr.GetDistance());
  view.MoveToBoundary(nfr.GetDistance());
  EXPECT_TRUE(view.IsOnBoundary());
  EXPECT_EQ(Real3(5, 30, 0), view.GetPosition());
  view.CrossBoundary();
  EXPECT_EQ(outer_path_, view.GetOpaquePath());
  EXPECT_TRUE(view.IsOnBoundary());

  // Find new direction and move away, then backscatter
  nfr = view.FindNextBoundary(2);
  EXPECT_EQ(NFRKind::miss, nfr.GetKind());
  view.MoveInternal(1.5);
  EXPECT_FALSE(view.IsOnBoundary());
  EXPECT_EQ(Real3(5, 31.5, 0), view.GetPosition());
  view.ChangeDirection({0, -1, 0});

  // Cross back into old volume
  nfr = view.FindNextBoundary(2);
  ASSERT_EQ(NFRKind::hit, nfr.GetKind());
  view.MoveToBoundary(nfr.GetDistance());
  EXPECT_TRUE(view.IsOnBoundary());
  EXPECT_EQ(Real3(5, 30, 0), view.GetPosition());
  view.CrossBoundary();
  EXPECT_EQ(inner_path_, view.GetOpaquePath());
  EXPECT_TRUE(view.IsOnBoundary());
}

// TODO: test edge cases:
// - backscatter after boundary
// - perpendicular scatter on boundary
// - crossing and back again

} // namespace test
} // namespace vecgeom
