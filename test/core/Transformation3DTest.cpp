#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Vector3D.h"

#ifdef VECGEOM_GEANT4
#include <G4AffineTransform.hh>
#include <G4RotationMatrix.hh>
#endif

using namespace vecgeom;

int main()
{
  Vector3D<Precision> point(-1, 1, 2);
  // identity
  Transformation3D t0;
  // translation and its inverse
  Transformation3D t1(-2, -2, -2);
  Transformation3D t3(2, 2, 2);
  // rotation
  Transformation3D t4(0, 0, 0, 90, 0, 0);
  // general transformations
  Transformation3D t5(1, 2, 3, 15, 45, 30);
  Transformation3D t6(-3, -2, 1, 75, 30, 90);
  // Test identity
  assert(t0 == Transformation3D::kIdentity);
  assert(t0.IsIdentity());
  assert(!t0.HasRotation());
  assert(!t0.HasTranslation());
  assert(t0.Transform(point) == point);
  // Test pure translation
  assert(!t1.IsIdentity());
  assert(t1.HasTranslation());
  assert(!t1.HasRotation());
  // Test copy constructor
  Transformation3D t2(t1);
  assert(t1 == t2);
  // Test composing translations
  assert(t2.Transform(point) == Vector3D<Precision>(1, 3, 4));
  assert(t3.Transform(t1.Transform(point)) == point);
  // Test pure rotation
  assert(!t4.IsIdentity());
  assert(!t4.HasTranslation());
  assert(t4.HasRotation());
  assert(t4.Transform(t4.Transform(point)) == Vector3D<Precision>(1, -1, 2));
  // Test multiplications
  auto testMultiply = [](Transformation3D const &tr1, Transformation3D const &tr2) {
    Transformation3D tr1c = tr1;
    Transformation3D tr2c = tr2;
    tr1c *= tr2;
    tr2c.MultiplyFromRight(tr1);
    std::cout << tr1c << "\n" << tr2c << "\n";
    return tr1c.ApproxEqual(tr2c);
  };
  assert(testMultiply(t0, t1));
  assert(testMultiply(t5, t0));
  assert(testMultiply(t1, t3));
  assert(testMultiply(t3, t4));
  assert(testMultiply(t5, t6));

#ifdef VECGEOM_GEANT4
  Transformation3D t5;
  t5.RotateX(45 * kDegToRad);
  t5.RotateY(120 * kDegToRad);
  t5.RotateZ(-30 * kDegToRad);
  t5.Rectify();
  Transformation3D t6(10., 20., 30, t5);
  auto vec1 = t6.Transform(Vector3D<double>{1, 1, 1});

  G4RotationMatrix rg5;
  rg5.rotateX(45 * kDegToRad);
  rg5.rotateY(120 * kDegToRad);
  rg5.rotateZ(-30 * kDegToRad);
  G4AffineTransform tg5nr(rg5, {10, 20, 30});
  rg5.rectify();
  G4AffineTransform tg5(rg5, {10, 20, 30});
  auto vec2 = tg5.InverseTransformPoint({1, 1, 1});
  assert(vec1[0] == vec2[0] && vec1[1] == vec2[1] && vec1[2] == vec2[2]);
#endif

  return 0;
}
