//
// File:    TestVector.cpp
// Purpose: Unit tests for the vecgeom::Vector
//

//-- ensure asserts are compiled in
#undef NDEBUG

#include "VecGeom/base/Vector.h"

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
  return 0;
}

int main(int, char **)
{
  return test();
}
