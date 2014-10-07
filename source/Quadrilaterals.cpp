#include "volumes/Quadrilaterals.h"

#include "backend/scalar/Backend.h"

namespace VECGEOM_NAMESPACE {

Quadrilaterals::Quadrilaterals(int size)
    : fPlanes(size), fSides{size, size, size, size},
      fCorners{size, size, size, size} {}

Quadrilaterals::~Quadrilaterals() {}

Quadrilaterals& Quadrilaterals::operator=(Quadrilaterals const &other) {
  fPlanes = other.fPlanes;
  for (int i = 0; i < 4; ++i) {
    fSides[i] = other.fSides[i];
    fCorners[i] = other.fCorners[i];
  }
  return *this;
}

void Quadrilaterals::Set(
    int index,
    Vector3D<Precision> const &corner0,
    Vector3D<Precision> const &corner1,
    Vector3D<Precision> const &corner2,
    Vector3D<Precision> const &corner3) {

  fSides[0].set(index, (corner1 - corner0).Normalized());
  fSides[1].set(index, (corner2 - corner1).Normalized());
  fSides[2].set(index, (corner3 - corner2).Normalized());
  fSides[3].set(index, (corner0 - corner3).Normalized());

  fCorners[0].set(index, corner0);
  fCorners[1].set(index, corner1);
  fCorners[2].set(index, corner2);
  fCorners[3].set(index, corner3);
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

  fPlanes.Set(index, normal, d);
}

void Quadrilaterals::FixNormalSign(int component, bool positive) {
  for (int i = 0, iMax = size(); i < iMax; ++i) {
    Vector3D<Precision> normal = GetNormal(i);
    if ((positive  && normal[component] < 0) ||
        (!positive && normal[component] > 0)) {
      fPlanes.Set(i, -normal, -GetDistance(i));
    }
  }
}

std::ostream& operator<<(std::ostream &os, Quadrilaterals const &quads) {
  for (int i = 0, iMax = quads.size(); i < iMax; ++i) {
    os << "{" << quads.GetNormal(i) << ", " << quads.GetDistance(i)
       << ", {";
    for (int j = 0; j < 4; ++j) os << quads.GetCorners()[j][i] << ", ";
    os << ", ";
    for (int j = 0; j < 4; ++j) os << quads.GetSides()[j][i] << ", ";
    os << "}\n";
  }
  return os;
}

} // End global namespace