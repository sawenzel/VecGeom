/// \file Quadrilaterals.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_QUADRILATERALS_H_
#define VECGEOM_VOLUMES_QUADRILATERALS_H_

#include "base/Global.h"

#include "base/Array.h"
#include "base/SOA.h"
#include "base/SOA3D.h"
#include "base/Vector3D.h"
#include "volumes/Planes.h"

namespace VECGEOM_NAMESPACE {

/// \brief Fixed size convex, planar quadrilaterals.
template <int N>
class Quadrilaterals : public AlignedBase {

private:

  SOA<Precision, 4, N> fPlane;
  SOA<Precision, 3, N> fSides[4];
  SOA<Precision, 3, N> fCorners[4];

public:

  typedef Precision Array_t[N];

#ifdef VECGEOM_STD_CXX11
  Quadrilaterals(int size);
#endif

  VECGEOM_CUDA_HEADER_BOTH
#ifndef VECGEOM_NVCC
  constexpr int size() const { return N; }
#else
  int size() const { return N; }
#endif

  /// \param corner0 First corner in counterclockwise order.
  /// \param corner1 Second corner in counterclockwise order.
  /// \param corner2 Third corner in counterclockwise order.
  /// \param corner3 Fourth corner in counterclockwise order.
  void Set(
      int index,
      Vector3D<Precision> const &corner0,
      Vector3D<Precision> const &corner1,
      Vector3D<Precision> const &corner2,
      Vector3D<Precision> const &corner3);

  VECGEOM_CUDA_HEADER_BOTH
  static Array_t const& ToFixedSize(Precision const *array);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v ConvexDistanceToOut(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::distance_v> const &direction) const;

};

template <int N>
VECGEOM_CUDA_HEADER_BOTH
typename Quadrilaterals<N>::Array_t const&
Quadrilaterals<N>::ToFixedSize(Precision const *array) {
  return *reinterpret_cast<Array_t const*>(array);
}

template <int N>
void Quadrilaterals<N>::Set(
    int index,
    Vector3D<Precision> const &corner0,
    Vector3D<Precision> const &corner1,
    Vector3D<Precision> const &corner2,
    Vector3D<Precision> const &corner3) {

  Vector3D<Precision> sides[4];
  sides[0] = (corner1 - corner0).Normalized();
  sides[1] = (corner2 - corner1).Normalized();
  sides[2] = (corner3 - corner2).Normalized();
  sides[3] = (corner0 - corner3).Normalized();
  for (int i = 0; i < 3; ++i) {
    fCorners[0][i][index] = corner0[i];
    fCorners[1][i][index] = corner1[i];
    fCorners[2][i][index] = corner2[i];
    fCorners[3][i][index] = corner3[i];
    for (int j = 0; j < 4; ++i) fSides[j][i][index] = sides[j][i];
  }
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

  for (int i = 0; i < 3; ++i) {
    fPlane[i][index] = normal[i];
  }
  fPlane[3][index] = d;
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals<N>::ConvexDistanceToOut(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::distance_v> const &direction) const {
  return Planes<N>::template DistanceToOutKernel<Backend>(
      fPlane[0], fPlane[1], fPlane[2], fPlane[3], point, direction);
}

/// \brief Runtime sized convex, planar quadrilaterals.
template <>
class Quadrilaterals<0> {

private:

  SOA3D<Precision> fNormal;
  Array<Precision> fDistance;
  SOA3D<Precision> fSides[4];
  SOA3D<Precision> fCorners[4];

public:

  typedef SOA3D<Precision> Sides_t[4];
  typedef SOA3D<Precision> Corners_t[4];

#ifdef VECGEOM_STD_CXX11
  inline Quadrilaterals(int size);
  inline ~Quadrilaterals();
#endif

  template <int N>
  Quadrilaterals(Quadrilaterals<N> const &other);

  template <int N>
  Quadrilaterals<0>& operator=(Quadrilaterals<N> const &other);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int size() const { return fNormal.size(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  SOA3D<Precision> const& GetNormal() const { return fNormal; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Array<Precision> const& GetDistance() const { return fDistance; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Sides_t const& GetSides() const { return fSides; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Corners_t const& GetCorners() const { return fCorners; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static Precision const* ToFixedSize(Precision const *array) { return array; }

  /// \param corner0 First corner in counterclockwise order.
  /// \param corner1 Second corner in counterclockwise order.
  /// \param corner2 Third corner in counterclockwise order.
  /// \param corner3 Fourth corner in counterclockwise order.
  inline void Set(
      int index,
      Vector3D<Precision> const &corner0,
      Vector3D<Precision> const &corner1,
      Vector3D<Precision> const &corner2,
      Vector3D<Precision> const &corner3);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v DistanceToIn(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::bool_v Contains(
      Vector3D<typename Backend::precision_v> const &point) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::inside_v Inside(
      Vector3D<typename Backend::precision_v> const &point) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v ConvexDistanceToOut(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::distance_v> const &direction) const;

};

#ifdef VECGEOM_STD_CXX11

Quadrilaterals<0>::Quadrilaterals(int size)
    : fNormal(size), fDistance(size), fSides{size, size, size, size},
      fCorners{size, size, size, size} {}

// Quadrilaterals<0>::Quadrilaterals() : fNormal(0), fDistance(NULL) {}

Quadrilaterals<0>::~Quadrilaterals() {}

template <int N>
Quadrilaterals<0>& Quadrilaterals<0>::operator=(
    Quadrilaterals<N> const &other) {
  fNormal = other.fNormal;
  fDistance = other.fDistance;
  for (int i = 0; i < 4; ++i) {
    fSides[i] = other.fSides[i];
    fCorners[i] = other.fCorners[i];
  }
  return *this;
}

void Quadrilaterals<0>::Set(
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
  if (d >= 0) {
    // Ensure normal is pointing away from origin
    normal = -normal;
    d = -d;
  }

  fNormal.set(index, normal);
  fDistance[index] = d;
}

#endif

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals<0>::DistanceToIn(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  const int n = size();
  Float_t bestDistance = kInfinity;
  Float_t distance = AlignedAllocate<Float_t>(n);
  Bool_t inBounds = AlignedAllocate<Float_t>(n);
  // Compute intersection and pray for the loop to be unrolled and/or vectorized
  for (int i = 0; i < n; ++i) {
    distance[i] = -(fNormal[0][i]*point[0] + fNormal[1][i]*point[1] +
                    fNormal[2][i]*point[2] + fDistance[i])
                 / (fNormal[0][i]*direction[0] + fNormal[1][i]*direction[1] +
                    fNormal[2][i]*direction[2]);
    Vector3D<Float_t> intersection = point + direction*distance[i];
    Bool_t inside[4];
    for (int j = 0; j < 4; ++j) {
      Float_t dot = fSides[j][0][i] * (intersection[0] - fCorners[j][0][i]) +
                    fSides[j][1][i] * (intersection[1] - fCorners[j][1][i]) +
                    fSides[j][2][i] * (intersection[2] - fCorners[j][2][i]);
      inside[j] = dot >= 0;
    }
    inBounds[i] = inside[0] && inside[1] && inside[2] && inside[3];
  }
  // Aggregate result
  for (int i = 0; i < n; ++i) {
    if (inBounds[i] && distance[i] >= 0 && distance[i] < bestDistance) {
      bestDistance = distance;
    }
  }
  AlignedFree(distance);
  AlignedFree(inBounds);
  return bestDistance;
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals<0>::ConvexDistanceToOut(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::distance_v> const &direction) const {
  return Planes<0>::template DistanceToOutKernel<Backend>(
      fNormal.x(), fNormal.y(), fNormal.z(), fDistance, point, direction);
}

} // End global namespace

#endif // VECGEOM_VOLUMES_QUADRILATERALS_H_