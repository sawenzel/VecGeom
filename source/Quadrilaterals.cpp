#include "volumes/Quadrilaterals.h"

#include "backend/scalar/Backend.h"

namespace VECGEOM_NAMESPACE {

#ifdef VECGEOM_STD_CXX11
Quadrilaterals::Quadrilaterals(int size)
    : fNormal(size), fDistance(NULL), fSides{size, size, size, size},
      fCorners{size, size, size, size} {
  fDistance = AlignedAllocate<Precision>(size);  
}
#endif

Quadrilaterals::Quadrilaterals() : fNormal(), fDistance(NULL) {}

Quadrilaterals::~Quadrilaterals() {
  AlignedFree(size);
}

Quadrilaterals::Set() {
    int index,
    Vector3D<Precision> const &corner0,
    Vector3D<Precision> const &corner1,
    Vector3D<Precision> const &corner2,
    Vector3D<Precision> const &corner3) {

  fCorners[0].set(index, corner0);
  fCorners[1].set(index, corner1);
  fCorners[2].set(index, corner2);
  fCorners[3].set(index, corner3);
  fSides[0].set(index, (corner1 - corner0).Normalized());
  fSides[1].set(index, (corner2 - corner1).Normalized());
  fSides[2].set(index, (corner3 - corner2).Normalized());
  fSides[3].set(index, (corner0 - corner3).Normalized());
  // TODO: It should be asserted that the quadrilateral is planar and convex.

  // Compute plane equation to retrieve normal and distance to origin
  // ax + by + cz + d = 0
  Precision a, b, c, d;
  a = corner0[1]*(corner1[2] - corner2[2]) +
      corner1[1]*(corner2[2] - corner0[2]) +
      corner2[1]*(corner0[2] - corner1[2]);
  b = corner0[2]*(corner1[0] - corner2[0]) +
      corner1[2]*(corner2[0] - corner0[0]) +
      corner2[2]*(corner0[0] - corner1[0]);
  c = corner0[0]*(corner1[1] - corner2[1]) +
      corner1[0]*(corner2[1] - corner0[1]) +
      corner2[0]*(corner0[1] - corner1[1]);
  d = - corner0[0]*(corner1[1]*corner2[2] - corner2[1]*corner1[2])
      - corner1[0]*(corner2[1]*corner0[2] - corner0[1]*corner2[2])
      - corner2[0]*(corner0[1]*corner1[2] - corner1[1]*corner0[2]);
  Vector3D<Precision> normal(a, b, c);
  // Normalize the plane equation
  // (ax + by + cz + d) / sqrt(a^2 + b^2 + c^2) = 0 =>
  // n0*x + n1*x + n2*x + p = 0
  Precision inverseLength = 1. / normal.Length();
  normal *= inverseLength;
  d *= inverseLength;
  if (d >= 0) {
    // Ensure normal is pointing away from origin
    normal = -normal;
    d = -d;
  }

  fNormal.set(index, normal);
  fDistance[index] = d;
}

} // End global namespace