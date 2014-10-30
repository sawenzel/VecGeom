#include "volumes/Quadrilaterals.h"

#include "backend/scalar/Backend.h"

namespace VECGEOM_NAMESPACE {

#ifdef VECGEOM_STD_CXX11
Quadrilaterals::Quadrilaterals(int size)
    : fPlanes(size), fSideVectors{size, size, size, size},
      fCorners{size, size, size, size} {}
#endif

Quadrilaterals::~Quadrilaterals() {}

Quadrilaterals& Quadrilaterals::operator=(Quadrilaterals const &other) {
  fPlanes = other.fPlanes;
  for (int i = 0; i < 4; ++i) {
    fSideVectors[i] = other.fSideVectors[i];
    fCorners[i] = other.fCorners[i];
  }
  return *this;
}

#ifdef VECGEOM_STD_CXX11
void Quadrilaterals::Set(
    int index,
    Vector3D<Precision> const &corner0,
    Vector3D<Precision> const &corner1,
    Vector3D<Precision> const &corner2,
    Vector3D<Precision> const &corner3) {

  // TODO: It should be asserted that the quadrilateral is planar and convex.

  fCorners[0].set(index, corner0);
  fCorners[1].set(index, corner1);
  fCorners[2].set(index, corner2);
  fCorners[3].set(index, corner3);

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

  fPlanes.Set(index, normal, d);

  auto ComputeSideVector = [&index, &normal] (
      Planes &sideVectors,
      Vector3D<Precision> const &c0,
      Vector3D<Precision> const &c1) {
    Vector3D<Precision> sideVector = normal.Cross(c1-c0).Normalized();
    sideVectors.Set(index, sideVector, c0);
  };

  ComputeSideVector(fSideVectors[0], corner0, corner1);
  ComputeSideVector(fSideVectors[1], corner1, corner2);
  ComputeSideVector(fSideVectors[2], corner2, corner3);
  ComputeSideVector(fSideVectors[3], corner3, corner0);
}
#endif

void Quadrilaterals::FlipSign(int index) {
  fPlanes.FlipSign(index);
}

std::ostream& operator<<(std::ostream &os, Quadrilaterals const &quads) {
  for (int i = 0, iMax = quads.size(); i < iMax; ++i) {
    os << "{(" << quads.GetNormal(i) << ", " << quads.GetDistance(i)
       << "), {(";
    for (int j = 0; j < 3; ++j) {
      os << quads.GetSideVectors()[j].GetNormals()[i]
         << ", " << quads.GetSideVectors()[j].GetDistances()[i] << "), ";
    }
    os << quads.GetSideVectors()[3].GetNormals()[i]
       << ", " << quads.GetSideVectors()[3].GetDistances()[i] << ")}}\n";
  }
  return os;
}

} // End global namespace