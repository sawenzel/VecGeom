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
  VECGEOM_ASSERT(t0 == Transformation3D::kIdentity);
  VECGEOM_ASSERT(t0.IsIdentity());
  VECGEOM_ASSERT(!t0.HasRotation());
  VECGEOM_ASSERT(!t0.HasTranslation());
  VECGEOM_ASSERT(t0.Transform(point) == point);
  // Test pure translation
  VECGEOM_ASSERT(!t1.IsIdentity());
  VECGEOM_ASSERT(t1.HasTranslation());
  VECGEOM_ASSERT(!t1.HasRotation());
  // Test copy constructor
  Transformation3D t2(t1);
  VECGEOM_ASSERT(t1 == t2);
  // Test composing translations
  VECGEOM_ASSERT(t2.Transform(point) == Vector3D<Precision>(1, 3, 4));
  VECGEOM_ASSERT(t3.Transform(t1.Transform(point)) == point);
  // Test pure rotation
  VECGEOM_ASSERT(!t4.IsIdentity());
  VECGEOM_ASSERT(!t4.HasTranslation());
  VECGEOM_ASSERT(t4.HasRotation());
  VECGEOM_ASSERT(t4.Transform(t4.Transform(point)) == Vector3D<Precision>(1, -1, 2));
  // Test multiplications
  auto testMultiply = [](Transformation3D const &tr1, Transformation3D const &tr2) {
    Transformation3D tr1c = tr1;
    Transformation3D tr2c = tr2;
    tr1c *= tr2;
    tr2c.MultiplyFromRight(tr1);
    std::cout << tr1c << "\n" << tr2c << "\n";
    return tr1c.ApproxEqual(tr2c);
  };
  VECGEOM_ASSERT(testMultiply(t0, t1));
  VECGEOM_ASSERT(testMultiply(t5, t0));
  VECGEOM_ASSERT(testMultiply(t1, t3));
  VECGEOM_ASSERT(testMultiply(t3, t4));
  VECGEOM_ASSERT(testMultiply(t5, t6));

#ifdef VECGEOM_GEANT4
  Transformation3D t7;
  t7.RotateX(45 * kDegToRad);
  t7.RotateY(120 * kDegToRad);
  t7.RotateZ(-30 * kDegToRad);
  t7.Rectify();
  Transformation3D t8(10., 20., 30, t7);
  auto vec1 = t8.Transform(Vector3D<double>{1, 1, 1});

  G4RotationMatrix rg5;
  rg5.rotateX(45 * kDegToRad);
  rg5.rotateY(120 * kDegToRad);
  rg5.rotateZ(-30 * kDegToRad);
  G4AffineTransform tg5nr(rg5, {10, 20, 30});
  rg5.rectify();
  G4AffineTransform tg5(rg5, {10, 20, 30});
  auto vec2 = tg5.InverseTransformPoint({1, 1, 1});
  VECGEOM_ASSERT(vec1[0] == vec2[0] && vec1[1] == vec2[1] && vec1[2] == vec2[2]);
#endif

  return 0;
}
