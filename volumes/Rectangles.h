/// \file Rectangles.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_RECTANGLES_H_
#define VECGEOM_VOLUMES_RECTANGLES_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/SOA3D.h"
#include "base/Vector3D.h"

#include <ostream>

namespace VECGEOM_NAMESPACE {

namespace {
template <bool treatSurfaceT> struct SurfaceTraits;
template <> struct SurfaceTraits<true>  { typedef Inside_t Surface_t; };
template <> struct SurfaceTraits<false> { typedef bool Surface_t; };
template <bool insideT> struct DistanceTraits;
}

/// \class Rectangles
///
/// \brief Stores a number of rectangles in SOA form to allow vectorized
///        operations.
class Rectangles : public AlignedBase {

private:

  //   fCorner[0]             fCorner[1]
  //             p0---------p2
  //       |      |         |
  // (Normalized) | fNormal |  fP := distance to origin
  //    fSide     |   -o-   |        along normal
  //       |      |         |
  //       v      |         |
  //             p1----------        

  int fSize;
  // SOA3D<Precision> fCenter;
  SOA3D<Precision> fNormal;
  Precision *fP;
  SOA3D<Precision> fSide;
  SOA3D<Precision> fCorner[2];

public:

#ifndef VECGEOM_NVCC
  inline Rectangles(int size);
#endif

  inline ~Rectangles();

  // inline void Set(
  //     int index,
  //     Vector3D<Precision> const &normal,
  //     Vector3D<Precision> const &center,
  //     Vector2D<Precision> const &dimensions);

  inline void Set(
      int index,
      Vector3D<Precision> const &p0,
      Vector3D<Precision> const &p1,
      Vector3D<Precision> const &p2);

  template <bool treatSurfaceT>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename SurfaceTraits<treatSurfaceT>::Surface_t Inside(
      Vector3D<Precision> const &point,
      int begin,
      int end) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Inside_t Inside(
      Vector3D<Precision> const &point,
      int begin,
      int end) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Inside_t Inside(Vector3D<Precision> const &point) const;

  template <bool insideT>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Distance(
      Vector3D<Precision> const &point,
      Vector3D<Precision> const &direction,
      int i,
      int end) const;

  template <bool insideT>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Distance(
      Vector3D<Precision> const &point,
      Vector3D<Precision> const &direction) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision DistanceToIn(
      Vector3D<Precision> const &point,
      Vector3D<Precision> const &direction) const;

  inline friend std::ostream& operator<<(
      std::ostream &os,
      Rectangles const &rhs);

};

#ifndef VECGEOM_NVCC
Rectangles::Rectangles(int size)
  : fSize(size), fNormal(size), fSide(size), fCorner{size, size} {
  fP = AlignedAllocate<Precision>(size);
}
#endif

Rectangles::~Rectangles() {
  AlignedFree(fP);
}

void Rectangles::Set(
    int index,
    Vector3D<Precision> const &p0,
    Vector3D<Precision> const &p1,
    Vector3D<Precision> const &p2) {

  // Store corners and sides
  fCorner[0].set(index, p0);
  fCorner[1].set(index, p2);
  Vector3D<Precision> side = p1 - p0;
  side.Normalize();
  fSide.set(index, side);

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

  fNormal.set(index, normal);
  fP[index] = d;

  // // Center of the plane is just backwards distance p along the normal
  // fCenter.set(index, -d * normal);
}

template <bool insideT>
VECGEOM_CUDA_HEADER_BOTH
Precision Rectangles::Distance(
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction,
    int i,
    int end) const {
  Precision bestDistance = kInfinity;
  for (;i < end; ++i) {
    Vector3D<Precision> normal = fNormal[i];
    Vector3D<Precision> side = fSide[i];
    Precision t = (normal.Dot(point) + fP[i]) / normal.Dot(direction);
    if (!insideT) t = -t;
    Vector3D<Precision> intersection = point + t*direction;
    Vector3D<Precision> fromP0 = intersection - fCorner[0][i];
    Vector3D<Precision> fromP2 = intersection - fCorner[1][i];
    if (t >= 0 && t < bestDistance && side.Dot(fromP0) >= 0 &&
        (-side).Dot(fromP2) >= 0) {
      bestDistance = t;
    }
  }
  return (bestDistance == kMaximum) ? kInfinity : bestDistance;
}

VECGEOM_CUDA_HEADER_BOTH
Precision Rectangles::DistanceToIn(
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction) const {
  return Distance<false>(point, direction, 0, fSize);
}

template <bool insideT>
VECGEOM_CUDA_HEADER_BOTH
Precision Rectangles::Distance(
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction) const {
  return Distance<insideT>(point, direction, 0, fSize);
}

std::ostream& operator<<(std::ostream &os, Rectangles const &rhs) {
  for (int i = 0, iMax = rhs.fSize; i < iMax; ++i) {
    Vector3D<Precision> normal = rhs.fNormal[i];
    Vector3D<Precision> center = -rhs.fP[i] * normal;
    os << "{(" << normal[0] << ", " << normal[1] << ", " << normal[2] << ", "
       << rhs.fP[i] << ") at " << center << ", corners in " << rhs.fCorner[0][i]
       << " and " << rhs.fCorner[1][i] << ", side " << rhs.fSide[i] << "}\n";
  }
  return os;
}

} // End global namespace

#endif // VECGEOM_VOLUMES_RECTANGLES_H_