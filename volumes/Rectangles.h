/// \file Rectangles.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_RECTANGLES_H_
#define VECGEOM_VOLUMES_RECTANGLES_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/SOA3D.h"
#include "base/Vector3D.h"
#include "volumes/Planes.h"


namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class Rectangles; )
VECGEOM_DEVICE_DECLARE_CONV( Rectangles );


inline namespace VECGEOM_IMPL_NAMESPACE {

/// \class Rectangles
///
/// \brief Stores a number of rectangles in SOA form to allow vectorized
///        operations.
///
/// To allow efficient computation, two corners, one normalized side and
/// the plane equation in which the rectangle lies is stored.
/// If the set of rectangles are assumed to be convex, the convex methods
/// can be called for faster computation, falling back on the implementations
/// for planes.
class Rectangles : public AlignedBase {

  //   fCorners[0]
  //             p0----------
  //       |      |         |
  // (Normalized) |         |
  //    fSides    |   -o-   |
  //       |      |         |
  //       v      |         |
  //             p1---------p2
  //                          fCorners[1]

private:

  Planes fPlanes;
  SOA3D<Precision> fSides;
  SOA3D<Precision> fCorners[2];

public:

  typedef SOA3D<Precision> Corners_t[2];

  VECGEOM_CUDA_HEADER_BOTH
  Rectangles(int size);

  VECGEOM_CUDA_HEADER_BOTH
  ~Rectangles();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int size() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> GetNormal(int i) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  SOA3D<Precision> const& GetNormals() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDistance(int i) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Array<Precision> const& GetDistances() const;

  VECGEOM_CUDA_HEADER_BOTH
  inline Vector3D<Precision> GetCenter(int i) const;

  VECGEOM_CUDA_HEADER_BOTH
  inline Vector3D<Precision> GetCorner(int i, int j) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Corners_t const& GetCorners() const;

  VECGEOM_CUDA_HEADER_BOTH
  inline Vector3D<Precision> GetSide(int i) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  SOA3D<Precision> const& GetSides() const;

  VECGEOM_CUDA_HEADER_BOTH
  void Set(
      int index,
      Vector3D<Precision> const &p0,
      Vector3D<Precision> const &p1,
      Vector3D<Precision> const &p2);

  VECGEOM_CUDA_HEADER_BOTH
  inline Precision Distance(
      Vector3D<Precision> const &point,
      Vector3D<Precision> const &direction) const;

};

VECGEOM_CUDA_HEADER_BOTH
int Rectangles::size() const { return fPlanes.size(); }

VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Rectangles::GetNormal(int i) const {
  return fPlanes.GetNormal(i);
}

VECGEOM_CUDA_HEADER_BOTH
SOA3D<Precision> const& Rectangles::GetNormals() const {
  return fPlanes.GetNormals();
}

VECGEOM_CUDA_HEADER_BOTH
Precision Rectangles::GetDistance(int i) const {
  return fPlanes.GetDistance(i);
}

VECGEOM_CUDA_HEADER_BOTH
Array<Precision> const& Rectangles::GetDistances() const {
  return fPlanes.GetDistances();
}

VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Rectangles::GetCenter(int i) const {
  return -GetDistance(i)*GetNormal(i);
}

VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Rectangles::GetCorner(int i, int j) const {
  return Vector3D<Precision>(fCorners[i][0][j], fCorners[i][1][j],
                             fCorners[i][2][j]);
}

VECGEOM_CUDA_HEADER_BOTH
Rectangles::Corners_t const& Rectangles::GetCorners() const {
  return fCorners;
}

VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Rectangles::GetSide(int i) const {
  return Vector3D<Precision>(fSides[0][i], fSides[1][i], fSides[2][i]);
}

VECGEOM_CUDA_HEADER_BOTH
SOA3D<Precision> const& Rectangles::GetSides() const {
  return fSides;
}

VECGEOM_CUDA_HEADER_BOTH
void Rectangles::Set(
    int index,
    Vector3D<Precision> const &p0,
    Vector3D<Precision> const &p1,
    Vector3D<Precision> const &p2) {

  // Store corners and sides
  fCorners[0].set(index, p0);
  fCorners[1].set(index, p2);
  Vector3D<Precision> side = p1 - p0;
  side.Normalize();
  fSides.set(index, side);

  // Compute plane equation to retrieve normal and distance to origin
  // ax + by + cz + d = 0
  Precision a, b, c, d;
  a = p0[1]*(p1[2] - p2[2]) + p1[1]*(p2[2] - p0[2]) + p2[1]*(p0[2] - p1[2]);
  b = p0[2]*(p1[0] - p2[0]) + p1[2]*(p2[0] - p0[0]) + p2[2]*(p0[0] - p1[0]);
  c = p0[0]*(p1[1] - p2[1]) + p1[0]*(p2[1] - p0[1]) + p2[0]*(p0[1] - p1[1]);
  d = - p0[0]*(p1[1]*p2[2] - p2[1]*p1[2])
      - p1[0]*(p2[1]*p0[2] - p0[1]*p2[2])
      - p2[0]*(p0[1]*p1[2] - p1[1]*p0[2]);
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

  fPlanes.Set(index, normal, d);
}

VECGEOM_CUDA_HEADER_BOTH
Precision Rectangles::Distance(
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction) const {
  Precision bestDistance = kInfinity;
  for (int i = 0, iMax = size(); i < iMax; ++i) {
    Vector3D<Precision> normal = GetNormal(i);
    Vector3D<Precision> side = GetSide(i);
    Precision t = -(normal.Dot(point) + GetDistance(i)) / normal.Dot(direction);
    Vector3D<Precision> intersection = point + t*direction;
    Vector3D<Precision> fromP0 = intersection - fCorners[0][i];
    Vector3D<Precision> fromP2 = intersection - fCorners[1][i];
    if (t >= 0 && t < bestDistance && side.Dot(fromP0) >= 0 &&
        (-side).Dot(fromP2) >= 0) {
      bestDistance = t;
    }
  }
  return bestDistance;
}

std::ostream& operator<<(std::ostream &os, Rectangles const &rhs);

} // End inline impl namespace

} // End global namespace

#endif // VECGEOM_VOLUMES_RECTANGLES_H_
