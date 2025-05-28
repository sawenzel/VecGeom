#include "VecGeom/volumes/Quadrilaterals.h"
#include "VecGeom/backend/scalar/Backend.h"
#include <memory>
#include <iostream>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
Quadrilaterals::Quadrilaterals(int size, bool convex)
    : fPlanes(size, convex), fSideVectors{{size}, {size}, {size}, {size}},
      fCorners{{(size_t)size}, {(size_t)size}, {(size_t)size}, {(size_t)size}}
{
}

VECCORE_ATT_HOST_DEVICE
Quadrilaterals::Quadrilaterals(int size, AlignedAllocator &a, bool convex)
    : fPlanes(size, a, convex), fSideVectors{{size, a}, {size, a}, {size, a}, {size, a}},
      fCorners{{(size_t)size, a}, {(size_t)size, a}, {(size_t)size, a}, {(size_t)size, a}}
{
}

VECCORE_ATT_HOST_DEVICE
Quadrilaterals::~Quadrilaterals() {}

VECCORE_ATT_HOST_DEVICE
Quadrilaterals &Quadrilaterals::operator=(Quadrilaterals const &other)
{
  fPlanes = other.fPlanes;
  for (int i = 0; i < 4; ++i) {
    fSideVectors[i] = other.fSideVectors[i];
    fCorners[i]     = other.fCorners[i];
  }
  return *this;
}

VECCORE_ATT_HOST_DEVICE
void Quadrilaterals::Set(int index, Vector3D<Precision> const &corner0, Vector3D<Precision> const &corner1,
                         Vector3D<Precision> const &corner2, Vector3D<Precision> const &corner3)
{

  // TODO: It should be asserted that the quadrilateral is planar and convex.

  fCorners[0].set(index, corner0);
  fCorners[1].set(index, corner1);
  fCorners[2].set(index, corner2);
  fCorners[3].set(index, corner3);

  // Compute plane equation to retrieve normal and distance to origin
  // ax + by + cz + d = 0
  // use only 3 corners given
  // problem: if two those three corners are degenerate we have to use the other
  // so we have to check first of all if there is a degenerate case

  // chose 3 corners out of 4: usually this will choose 0,1,2 unless there are degenerate points
  Vector3D<Precision> chosencorners[4];
  chosencorners[0]    = fCorners[0][index];
  int cornersassigned = 1;
  int cornerstested   = 1;

  while (cornerstested < 4) {
    bool chosen = true;
    for (int j = 0; j < cornersassigned; ++j)
      if (chosencorners[j] == fCorners[cornerstested][index]) {
        chosen = false;
        break;
      }
    if (chosen) {
      chosencorners[cornersassigned] = fCorners[cornerstested][index];
      ++cornersassigned;
    }
    ++cornerstested;
  }
#ifndef VECCORE_CUDA
  if (cornersassigned < 3) std::cerr << "Quadrilaterals::Set: could not find three non degenerated points" << std::endl;
#endif

  Precision a, b, c, d;
  a = chosencorners[0][1] * (chosencorners[1][2] - chosencorners[2][2]) +
      chosencorners[1][1] * (chosencorners[2][2] - chosencorners[0][2]) +
      chosencorners[2][1] * (chosencorners[0][2] - chosencorners[1][2]);
  b = chosencorners[0][2] * (chosencorners[1][0] - chosencorners[2][0]) +
      chosencorners[1][2] * (chosencorners[2][0] - chosencorners[0][0]) +
      chosencorners[2][2] * (chosencorners[0][0] - chosencorners[1][0]);
  c = chosencorners[0][0] * (chosencorners[1][1] - chosencorners[2][1]) +
      chosencorners[1][0] * (chosencorners[2][1] - chosencorners[0][1]) +
      chosencorners[2][0] * (chosencorners[0][1] - chosencorners[1][1]);
  d = -chosencorners[0][0] * (chosencorners[1][1] * chosencorners[2][2] - chosencorners[2][1] * chosencorners[1][2]) -
      chosencorners[1][0] * (chosencorners[2][1] * chosencorners[0][2] - chosencorners[0][1] * chosencorners[2][2]) -
      chosencorners[2][0] * (chosencorners[0][1] * chosencorners[1][2] - chosencorners[1][1] * chosencorners[0][2]);
  Vector3D<Precision> normal(a, b, c);
  // Normalize the plane equation
  // (ax + by + cz + d) / sqrt(a^2 + b^2 + c^2) = 0 =>
  // n0*x + n1*x + n2*x + p = 0

  // VECGEOM_ASSERT( a+b+c != 0 ); // this happens in extremely degenerate cases and would lead to ill defined planes
  Precision inverseLength = 1. / normal.Length();
  normal *= inverseLength;
  d *= inverseLength;

  normal.FixZeroes();
  fPlanes.Set(index, normal, d);

  auto ComputeSideVector = [&index, &normal](Planes &sideVectors, Vector3D<Precision> const &c0,
                                             Vector3D<Precision> const &c1) {
    // protect against degenerate points
    if (!(c1 == c0)) {
      Vector3D<Precision> sideVector = normal.Cross(c1 - c0).Normalized();
      sideVectors.Set(index, sideVector, c0);
    } else {
      // the choice (0,0,0), 0 is motivated barely from the fact that
      // it does not do anything harmful in the hit checks
      sideVectors.Set(index, Vector3D<Precision>(0, 0, 0), 0);
    }
  };

  ComputeSideVector(fSideVectors[0], corner0, corner1);
  ComputeSideVector(fSideVectors[1], corner1, corner2);
  ComputeSideVector(fSideVectors[2], corner2, corner3);
  ComputeSideVector(fSideVectors[3], corner3, corner0);
}

VECCORE_ATT_HOST_DEVICE
void Quadrilaterals::FlipSign(int index) { fPlanes.FlipSign(index); }

VECCORE_ATT_HOST_DEVICE
void Quadrilaterals::Print() const
{
  for (int i = 0, iMax = size(); i < iMax; ++i) {
    printf("{(%.2f, %.2f, %.2f, %.2f), {", GetNormals().x(i), GetNormals().y(i), GetNormals().z(i), GetDistance(i));
    for (int j = 0; j < 3; ++j) {
      printf("(%.2f, %.2f, %.2f, %.2f), ", GetSideVectors()[j].GetNormals().x(i), GetSideVectors()[j].GetNormals().y(i),
             GetSideVectors()[j].GetNormals().z(i), GetSideVectors()[j].GetDistance(i));
    }
    printf("(%.2f, %.2f, %.2f, %.2f)}}", GetSideVectors()[3].GetNormals().x(i), GetSideVectors()[3].GetNormals().y(i),
           GetSideVectors()[3].GetNormals().z(i), GetSideVectors()[3].GetDistance(i));
  }
}

std::ostream &operator<<(std::ostream &os, Quadrilaterals const &quads)
{
  for (int i = 0, iMax = quads.size(); i < iMax; ++i) {
    os << "{(" << quads.GetNormal(i) << ", " << quads.GetDistance(i) << "), {(";
    for (int j = 0; j < 3; ++j) {
      os << quads.GetSideVectors()[j].GetNormals()[i] << ", " << quads.GetSideVectors()[j].GetDistances()[i] << "), ";
    }
    os << quads.GetSideVectors()[3].GetNormals()[i] << ", " << quads.GetSideVectors()[3].GetDistances()[i] << ")}}\n";
  }
  return os;
}

} // namespace VECGEOM_IMPL_NAMESPACE

} // End namespace vecgeom
