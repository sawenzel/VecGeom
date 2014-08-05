/// \file Rectangles.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_RECTANGLES_H_
#define VECGEOM_VOLUMES_RECTANGLES_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "base/SOA3D.h"
#include "base/Vector3D.h"
#include "volumes/Planes.h"

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
///
/// To allow efficient computation, two corners, one normalized side and
/// the plane equation in which the rectangle lies is stored.
/// If the set of rectangles are assumed to be convex, the convex methods
/// can be called for faster computation, falling back on the implementations
/// for planes.
#ifdef VECGEOM_STD_CXX11
template <int N = 0>
#else
template <int N>
#endif
class Rectangles : public AlignedBase {

  //   fCorner[0]
  //             p0--------------
  //       |      |             |
  // (Normalized) | fPlane[0-2] | fPlane[3] := distance to origin
  //    fSide     |     -o-     |              along normal
  //       |      |             |
  //       v      |             |
  //             p1-------------p2
  //                              fCorner[1]

private:

  Precision fPlane[4][N];
  Precision fCorner[2][3][N];
  Precision fSide[3][N];

public:

  Rectangles() {}

#ifdef VECGEOM_STD_CXX11
  constexpr int GetSize() const { return N; }
#else
  int GetSize() const { return N; }
#endif

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> GetNormal(int i) const;

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDistance(int i) const;

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> GetCenter(int i) const;

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> GetCorner(int i, int j) const;

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> GetSide(int i) const;

  /// \brief Define a 3D rectangle from three of its corners.
  /// \param p0 First corner in sequence around the center.
  /// \param p1 Second corner in sequence around the center.
  /// \param p2 Third corner in sequence around the center.
  void Set(
      int index,
      Vector3D<Precision> const &p0,
      Vector3D<Precision> const &p1,
      Vector3D<Precision> const &p2);

  /// \brief Assumes a convex set of rectangles.
  /// Falls back on the implementation for planes.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::bool_v ConvexContains(
      Vector3D<typename Backend::precision_v> const &point) const;

  /// \brief Assumes a convex set of rectangles.
  /// Falls back on the implementation for planes.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::inside_v ConvexInside(
      Vector3D<typename Backend::precision_v> const &point) const;

  /// \brief Assumes a convex set of rectangles.
  /// Falls back on the implementation for planes.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v ConvexDistanceToOut(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction) const;

  /// \brief Makes no assumptions, except that the point should be outside the
  ///        rectangles.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v DistanceToIn(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction) const;

  // Static versions to be called from a specialized volume

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::bool_v ConvexContainsKernel(
      Precision const plane[4][N],
      Vector3D<typename Backend::precision_v> const &point);

  /// \brief Assumes a convex set of rectangles.
  /// Falls back on the implementation for planes.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::inside_v ConvexInsideKernel(
      Precision const plane[4][N],
      Vector3D<typename Backend::precision_v> const &point);

  /// \brief Assumes a convex set of rectangles.
  /// Falls back on the implementation for planes.
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v ConvexDistanceToOutKernel(
      Precision const plane[4][N],
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v DistanceToInKernel(
      Precision const plane[4][N],
      Precision const corner[2][3][N],
      Precision const side[3][N],
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

};

template <int N>
VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Rectangles<N>::GetNormal(int i) const {
  return Vector3D<Precision>(fPlane[0][i], fPlane[1][i], fPlane[2][i]);
}

template <int N>
VECGEOM_CUDA_HEADER_BOTH
Precision Rectangles<N>::GetDistance(int i) const {
  return fPlane[3][i];
}

template <int N>
VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Rectangles<N>::GetCenter(int i) const {
  return -fPlane[3][i] * GetNormal(i);
}

template <int N>
VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Rectangles<N>::GetCorner(int i, int j) const {
  return Vector3D<Precision>(fCorner[i][0][j], fCorner[i][1][j],
                             fCorner[i][2][j]);
}

template <int N>
VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Rectangles<N>::GetSide(int i) const {
  return Vector3D<Precision>(fSide[0][i], fSide[1][i], fSide[2][i]);
}

template <int N>
void Rectangles<N>::Set(
    int index,
    Vector3D<Precision> const &p0,
    Vector3D<Precision> const &p1,
    Vector3D<Precision> const &p2) {

  // Store corners and sides
  fCorner[0][0][index] = p0[0];
  fCorner[0][1][index] = p0[1];
  fCorner[0][2][index] = p0[2];
  fCorner[1][0][index] = p2[0];
  fCorner[1][1][index] = p2[1];
  fCorner[1][2][index] = p2[2];
  Vector3D<Precision> side = (p1 - p0).Normalized();
  fSide[0][index] = side[0];
  fSide[1][index] = side[1];
  fSide[2][index] = side[2];

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

  fPlane[0][index] = normal[0];
  fPlane[1][index] = normal[1];
  fPlane[2][index] = normal[2];
  fPlane[3][index] = d;
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::inside_v Rectangles<N>::ConvexInside(
    Vector3D<typename Backend::precision_v> const &point) const {
  return ConvexInsideKernel<Backend>(fPlane, point);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::bool_v Rectangles<N>::ConvexContains(
    Vector3D<typename Backend::precision_v> const &point) const {
  return ConvexContainsKernel<Backend>(fPlane, point);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Rectangles<N>::ConvexDistanceToOut(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {
  return ConvexDistanceToOutKernel<Backend>(fPlane, point, direction);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Rectangles<N>::DistanceToIn(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {
  return DistanceToInKernel<Backend>(
      fPlane, fCorner, fSide, point, direction);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::inside_v Rectangles<N>::ConvexInsideKernel(
    Precision const plane[4][N],
    Vector3D<typename Backend::precision_v> const &point) {
  return Planes<N>::template InsideKernel<Backend>(plane, point);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::bool_v Rectangles<N>::ConvexContainsKernel(
    Precision const plane[4][N],
    Vector3D<typename Backend::precision_v> const &point) {
  return Planes<N>::template ContainsKernel<Backend>(plane, point);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Rectangles<N>::ConvexDistanceToOutKernel(
    Precision const plane[4][N],
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {
  return Planes<N>::template DistanceToOutKernel<Backend>(
      plane, point, direction);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Rectangles<N>::DistanceToInKernel(
    Precision const plane[4][N],
    Precision const corner[2][3][N],
    Precision const side[3][N],
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;

  Float_t bestDistance = kInfinity;
  Float_t distance[N];
  for (int i = 0; i < N; ++i) {
    distance[i] = -(plane[0][i]*point[0] + plane[1][i]*point[1] +
                    plane[2][i]*point[2] + plane[3][i])
                 / (plane[0][i]*direction[0] + plane[1][i]*direction[1] +
                    plane[2][i]*direction[2]);
    Vector3D<Float_t> intersection = point + distance[i]*direction;
    Vector3D<Float_t> fromP0(
      intersection[0] - corner[0][0][i],
      intersection[1] - corner[0][1][i],
      intersection[2] - corner[0][2][i]
    );
    Vector3D<Float_t> fromP2(
      intersection[0] - corner[1][0][i],
      intersection[1] - corner[1][1][i],
      intersection[2] - corner[1][2][i]
    );
    Vector3D<Float_t> side(
      side[0][i],
      side[1][i],
      side[2][i]
    );
    if (distance >= 0 && distance < bestDistance &&
        side.Dot(fromP0) >= 0 && (-side).Dot(fromP2) >= 0) {
      bestDistance = distance;
    }
  }
  return bestDistance;
}

/// \brief Specialization that allows the size to be specified at runtime.
///
/// Is not as likely to be autovectorized as the static version, so could
/// benefit from explicit vectorization. However, attempts at this has so far
/// shown worse results than the naive scalar implementation.
template <>
class Rectangles<0> : public AlignedBase {

  //   fCorner[0]
  //             p0----------
  //       |      |         |
  // (Normalized) | fNormal | fP := distance to origin
  //    fSide     |   -o-   |       along normal
  //       |      |    '    |
  //       v      |         |
  //             p1---------p2
  //                          fCorner[1]

private:

  int fSize;
  SOA3D<Precision> fNormal;
  Precision *fP;
  SOA3D<Precision> fSide;
  SOA3D<Precision> fCorner[2];

public:

#ifndef VECGEOM_NVCC
  inline Rectangles(int size);
#endif

  inline ~Rectangles();

  VECGEOM_CUDA_HEADER_BOTH
  int GetSize() const { return fSize; }

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> GetNormal(int i) const { return fNormal[i]; }

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> GetDistance(int i) const { return fP[i]; }

  VECGEOM_CUDA_HEADER_BOTH
  inline Vector3D<Precision> GetCenter(int i) const;

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> GetCorner(int i, int j) const;

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> GetSide(int i) const { return fSide[i]; }

  inline void Set(
      int index,
      Vector3D<Precision> const &p0,
      Vector3D<Precision> const &p1,
      Vector3D<Precision> const &p2);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision DistanceToIn(
      Vector3D<Precision> const &point,
      Vector3D<Precision> const &direction) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision DistanceToOut(
      Vector3D<Precision> const &point,
      Vector3D<Precision> const &direction) const;

private:

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

};

#ifndef VECGEOM_NVCC
Rectangles<0>::Rectangles(int size)
  : fSize(size), fNormal(size), fSide(size), fCorner{size, size} {
  fP = AlignedAllocate<Precision>(size);
}
#endif

Rectangles<0>::~Rectangles() {
  AlignedFree(fP);
}

VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Rectangles<0>::GetCenter(int i) const {
  return -fP[i] * GetNormal(i);
}

VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Rectangles<0>::GetCorner(int i, int j) const {
  return fCorner[i][j];
}

void Rectangles<0>::Set(
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
}

template <bool insideT>
VECGEOM_CUDA_HEADER_BOTH
Precision Rectangles<0>::Distance(
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
  return bestDistance;
}

VECGEOM_CUDA_HEADER_BOTH
Precision Rectangles<0>::DistanceToIn(
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction) const {
  return Distance<false>(point, direction, 0, fSize);
}

VECGEOM_CUDA_HEADER_BOTH
Precision Rectangles<0>::DistanceToOut(
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction) const {
  return Distance<true>(point, direction, 0, fSize);
}

template <int N>
std::ostream& operator<<(std::ostream &os, Rectangles<N> const &rhs) {
  for (int i = 0, iMax = rhs.GetSize(); i < iMax; ++i) {
    Vector3D<Precision> normal = rhs.GetNormal(i);
    os << "{(" << normal[0] << ", " << normal[1] << ", " << normal[2] << ", "
       << rhs.GetDistance(i) << ") at " << rhs.GetCenter(i) << ", corners in "
       << rhs.GetCorner(0, i) << " and " << rhs.GetCorner(1, i) << ", side "
       << rhs.GetSide(i) << "}\n";
  }
  return os;
}

} // End global namespace

#endif // VECGEOM_VOLUMES_RECTANGLES_H_