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

  Quadrilaterals(int size);

  VECGEOM_CUDA_HEADER_BOTH
  constexpr int size() const { return N; }

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

  template <class Backend, class InputStruct>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v DistanceToInKernel(
      const int n,
      Precision const (&a)[N],
      Precision const (&b)[N],
      Precision const (&c)[N],
      Precision const (&d)[N],
      InputStruct const (&sides)[4],
      InputStruct const (&corners)[4],
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v DistanceToOutKernel(
      const int n,
      Precision const (&a)[N],
      Precision const (&b)[N],
      Precision const (&c)[N],
      Precision const (&d)[N],
      const Precision zMin,
      const Precision zMax,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

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
template <class Backend, class InputStruct>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals<N>::DistanceToInKernel(
    const int n,
    Precision const (&a)[N],
    Precision const (&b)[N],
    Precision const (&c)[N],
    Precision const (&d)[N],
    InputStruct const (&sides)[4],
    InputStruct const (&corners)[4],
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t bestDistance = kInfinity;

  Float_t distance[N];
  Bool_t valid[N];
  for (int i = 0; i < N; ++i) {
    distance[i] = -(a[i]*point[0] + b[i]*point[1] + c[i]*point[2] + d[i])
                 / (a[i]*direction[0] + b[i]*direction[1] +
                    c[i]*direction[2]);
    Vector3D<Float_t> intersection = point + direction*distance[i];
    Bool_t inBounds[4];
    for (int j = 0; j < 4; ++j) {
      Vector3D<Float_t> cross =
          Vector3D<Float_t>::Cross(sides[j][i], intersection - corners[j][i]);
      inBounds[j] = a[i]*cross[0] + b[i]*cross[1] + c[i]*cross[2] >= 0;
    }
    valid[i] = distance[i] >= 0 &&
               inBounds[0] && inBounds[1] && inBounds[2] && inBounds[3];
  }
  for (int i = 0; i < N; ++i) {
    MaskedAssign(valid[i] && distance[i] < bestDistance, distance[i],
                 &bestDistance);
  }

  return bestDistance;
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals<N>::DistanceToOutKernel(
    const int n,
    Precision const (&a)[N],
    Precision const (&b)[N],
    Precision const (&c)[N],
    Precision const (&d)[N],
    const Precision zMin,
    const Precision zMax,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t bestDistance = kInfinity;

  Float_t distance[N];
  Bool_t inBounds[N];
  for (int i = 0; i < N; ++i) {
    distance[i] = -(a[i]*point[0] + b[i]*point[1] +
                    c[i]*point[2] + d[i])
                 / (a[i]*direction[0] + b[i]*direction[1] +
                    c[i]*direction[2]);
    Float_t zProjection = point[2] + distance[i]*direction[2];
    inBounds[i] = distance[i] >= 0 &&
                  zProjection >= zMin && zProjection <= zMax;
  }
  for (int i = 0; i < N; ++i) {
    MaskedAssign(inBounds[i] && distance[i] < bestDistance, distance[i],
                 &bestDistance);
  }

  return bestDistance;
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

  /// Set the sign of the specified component of the normal to positive or
  /// negative by flipping the sign of all components to the desired sign.
  /// \param component Which component of the normal to fix, [0: x, 1: y, 2: z].
  /// \param positive Whether the component should be set to positive (true) or
  ///                 negative (false).
  inline void FixNormalSign(int component, bool positive);

  template <class Backend, class InputStruct>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v DistanceToInKernel(
      const int n,
      Precision const a[],
      Precision const b[],
      Precision const c[],
      Precision const d[],
      InputStruct const (&sides)[4],
      InputStruct const (&corners)[4],
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v DistanceToOutKernel(
      const int n,
      Precision const a[],
      Precision const b[],
      Precision const c[],
      Precision const d[],
      const Precision zMin,
      const Precision zMax,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

};

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

  fNormal.set(index, normal);
  fDistance[index] = d;
}

void Quadrilaterals<0>::FixNormalSign(
    int component,
    bool positive) {
  for (int i = 0, iMax = fNormal.size(); i < iMax; ++i) {
    Vector3D<Precision> normal = fNormal[i];
    if ((positive  && normal[component] < 0) ||
        (!positive && normal[component] > 0)) {
      fNormal.set(i, -normal);
      fDistance[i] = -fDistance[i];
    }
  }
}

template <class Backend, class InputStruct>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals<0>::DistanceToInKernel(
    const int n,
    Precision const a[],
    Precision const b[],
    Precision const c[],
    Precision const d[],
    InputStruct const (&sides)[4],
    InputStruct const (&corners)[4],
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t bestDistance = kInfinity;

  for (int i = 0; i < n; ++i) {
    Float_t distance = -(a[i]*point[0] + b[i]*point[1] + c[i]*point[2] + d[i])
                      / (a[i]*direction[0] + b[i]*direction[1] +
                         c[i]*direction[2]);
    Vector3D<Float_t> intersection = point + direction*distance;
    Bool_t inBounds[4];
    for (int j = 0; j < 4; ++j) {
      Vector3D<Float_t> cross =
          Vector3D<Float_t>::Cross(sides[j][i], intersection - corners[j][i]);
      inBounds[j] = a[i]*cross[0] + b[i]*cross[1] + c[i]*cross[2] >= 0;
    }
    MaskedAssign(distance >= 0 && distance < bestDistance &&
                 inBounds[0] && inBounds[1] && inBounds[2] && inBounds[3],
                 distance, &bestDistance);
  }

  return bestDistance;
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals<0>::DistanceToOutKernel(
    const int n,
    Precision const a[],
    Precision const b[],
    Precision const c[],
    Precision const d[],
    const Precision zMin,
    const Precision zMax,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;

  Float_t bestDistance = kInfinity;

  for (int i = 0; i < n; ++i) {
    Float_t distance = -(a[i]*point[0] + b[i]*point[1] +
                         c[i]*point[2] + d[i])
                      / (a[i]*direction[0] + b[i]*direction[1] +
                         c[i]*direction[2]);
    Float_t zProjection = point[2] + distance*direction[2];
    MaskedAssign(distance >= 0 && zProjection >= zMin && zProjection <= zMax &&
                 distance < bestDistance, distance, &bestDistance);
  }

  return bestDistance;
}

template <int N>
std::ostream& operator<<(std::ostream &os, Quadrilaterals<N> const &quads) {
  for (int i = 0, iMax = quads.size(); i < iMax; ++i) {
    os << "{" << quads.GetNormal()[i] << ", " << quads.GetDistance()[i]
       << ", {";
    for (int j = 0; j < 4; ++j) os << quads.GetCorners()[j][i] << ", ";
    os << ", ";
    for (int j = 0; j < 4; ++j) os << quads.GetSides()[j][i] << ", ";
    os << "}\n";
  }
  return os;
}

} // End global namespace

#endif // VECGEOM_VOLUMES_QUADRILATERALS_H_