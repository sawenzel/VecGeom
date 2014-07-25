#include "base/Transformation3D.h"

using namespace vecgeom;

int main() {
  Vector3D<Precision> point(-1, 1, 2);
  Transformation3D t0;
  assert(t0 == Transformation3D::kIdentity);
  assert(t0.IsIdentity());
  assert(!t0.HasRotation());
  assert(!t0.HasTranslation());
  assert(t0.Transform(point) == point);
  Transformation3D t1(-2, -2, -2);
  assert(!t1.IsIdentity());
  assert(t1.HasTranslation());
  assert(!t1.HasRotation());
  Transformation3D t2(t1);
  assert(t1 == t2);
  assert(t2.Transform(point) == Vector3D<Precision>(1, 3, 4));
  Assert(t2.Transform<translation::kIdentity, rotation::kIdentity>(point)
         == point);
  Transformation3D t3(2, 2, 2);
  assert(t3.Transform(t1.Transform(point)) == point);
  Transformation3D t4(0, 0, 0, 90, 0, 0);
  assert(!t4.IsIdentity());
  assert(!t4.HasTranslation());
  assert(t4.HasRotation());
  assert(t4.Transform(t4.Transform(point)) == Vector3D<Precision>(1, -1, 2));
  return 0;
}