//
// File:    TestVector.cpp
// Purpose: Unit tests for the vecgeom::Vector
//

//-- ensure asserts are compiled in
#undef NDEBUG

#include "VecGeom/base/Vector.h"
#include "VecGeom/base/Vector2D.h"

int test()
{
  vecgeom::Vector<double> aVector;
  aVector.resize(2, 0.0);
  size_t newSize = aVector.size();

  VECGEOM_ASSERT(newSize == 2);

  aVector.reserve(10);
  VECGEOM_ASSERT(aVector.capacity() == 10);
  VECGEOM_ASSERT(aVector.size() == 2);

  for (int i = 0; i < 12; ++i) {
    aVector.push_back(i);
  }
  VECGEOM_ASSERT(aVector.capacity() > 10);
  VECGEOM_ASSERT(aVector.size() == 14);

  int i = 0;
  for (auto val : aVector) {
    if (i < 2)
      VECGEOM_ASSERT(val == 0);
    else
      VECGEOM_ASSERT(val == (i - 2));
    ++i;
  }
  for (i = 0; i < 12; ++i) {
    if (i < 2)
      VECGEOM_ASSERT(aVector[i] == 0);
    else
      VECGEOM_ASSERT(aVector[i] == (i - 2));
  }

  aVector.clear();
  VECGEOM_ASSERT(aVector.capacity() > 10);
  VECGEOM_ASSERT(aVector.size() == 0);

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
  VECGEOM_ASSERT(va.size() == 10);
  VECGEOM_ASSERT(va[7] == Vec2D(7, 3));
  // copy constructor and assignment operator
  auto vb(va);
  VECGEOM_ASSERT(equal_vect(va, vb));
  vb = va;
  VECGEOM_ASSERT(equal_vect(va, vb));
  // resize shrinking
  vb.resize(2, {});
  VECGEOM_ASSERT(vb.size() == 2);
  VECGEOM_ASSERT(vb[2] == Vec2D());
  // clear
  vb.clear();
  VECGEOM_ASSERT(vb.size() == 0);
  VECGEOM_ASSERT(vb[7] == Vec2D()); // still valid since the support array is not shrinked
  // Vector<Vector<T>>
  vecgeom::Vector<vecgeom::Vector<Vec2D>> vc;
  for (auto i = 0; i < 10; ++i) {
    vc.push_back(va);
    VECGEOM_ASSERT(equal_vect(va, vc[i]));
  }

  return 0;
}

int main(int, char **) { return test(); }
