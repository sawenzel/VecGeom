//
// File:    TestVector.cpp
// Purpose: Unit tests for the vecgeom::Vector
//

#include "VecGeom/base/Vector.h"
#include "VecGeom/base/Vector2D.h"

#include <gtest/gtest.h>

TEST(VecgeomBaseVector, Vector)
{
  vecgeom::Vector<double> aVector;
  aVector.resize(2, 0.0);
  size_t newSize = aVector.size();

  EXPECT_EQ(newSize, 2);

  aVector.reserve(10);
  EXPECT_EQ(aVector.capacity(), 10);
  EXPECT_EQ(aVector.size(), 2);

  for (int i = 0; i < 12; ++i) {
    aVector.push_back(i);
  }
  EXPECT_TRUE(aVector.capacity() > 10);
  EXPECT_EQ(aVector.size(), 14);

  int i = 0;
  for (auto val : aVector) {
    if (i < 2)
      EXPECT_EQ(val, 0);
    else
      EXPECT_EQ(val, (i - 2));
    ++i;
  }
  for (i = 0; i < 12; ++i) {
    if (i < 2)
      EXPECT_EQ(aVector[i], 0);
    else
      EXPECT_EQ(aVector[i], (i - 2));
  }

  aVector.clear();
  EXPECT_TRUE(aVector.capacity() > 10);
  EXPECT_EQ(aVector.size(), 0);
}

TEST(VecgeomBaseVector, VectorOfVector)
{
  // Testing Vector<Vector<T>>
  using Vec2D     = vecgeom::Vector2D<int>;
  auto equal_vect = [](vecgeom::Vector<Vec2D> const &v1, vecgeom::Vector<Vec2D> const &v2) {
    if (v1.size() != v2.size()) return false;
    for (size_t i = 0; i < v1.size(); ++i)
      if (v1[i] != v2[i]) return false;
    return true;
  };
  vecgeom::Vector<Vec2D> va;
  // push_back / resize
  for (auto i = 0; i < 10; ++i)
    va.push_back({i, 10 - i});
  EXPECT_EQ(va.size(), 10);
  EXPECT_EQ(va[7], Vec2D(7, 3));
  // copy constructor and assignment operator
  auto vb(va);
  EXPECT_TRUE(equal_vect(va, vb));
  vb = va;
  EXPECT_TRUE(equal_vect(va, vb));
  // resize shrinking
  vb.resize(2, {});
  EXPECT_EQ(vb.size(), 2);
  EXPECT_EQ(vb[2], Vec2D());
  // clear
  vb.clear();
  EXPECT_EQ(vb.size(), 0);
  EXPECT_EQ(vb[7], Vec2D()); // still valid since the support array is not shrinked
  // Vector<Vector<T>>
  vecgeom::Vector<vecgeom::Vector<Vec2D>> vc;
  for (auto i = 0; i < 10; ++i) {
    vc.push_back(va);
    EXPECT_TRUE(equal_vect(va, vc[i]));
  }
}

