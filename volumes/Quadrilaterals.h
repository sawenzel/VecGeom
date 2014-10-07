/// \file Quadrilaterals.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_QUADRILATERALS_H_
#define VECGEOM_VOLUMES_QUADRILATERALS_H_

#include "base/Global.h"

#include "base/Array.h"
#include "base/SOA3D.h"
#include "base/Vector3D.h"
#include "volumes/Planes.h"

namespace VECGEOM_NAMESPACE {

class Quadrilaterals {

private:

  Planes fPlanes;
  SOA3D<Precision> fSides[4];
  SOA3D<Precision> fCorners[4];

public:

  typedef SOA3D<Precision> Sides_t[4];
  typedef SOA3D<Precision> Corners_t[4];

  Quadrilaterals(int size);

  ~Quadrilaterals();

  Quadrilaterals(Quadrilaterals const &other);

  Quadrilaterals& operator=(Quadrilaterals const &other);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int size() const { return fPlanes.size(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  SOA3D<Precision> const& GetNormals() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> GetNormal(int i) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Array<Precision> const& GetDistances() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDistance(int i) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Sides_t const& GetSides() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Corners_t const& GetCorners() const;

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

  /// Set the sign of the specified component of the normal to positive or
  /// negative by flipping the sign of all components to the desired sign.
  /// \param component Which component of the normal to fix, [0: x, 1: y, 2: z].
  /// \param positive Whether the component should be set to positive (true) or
  ///                 negative (false).
  void FixNormalSign(int component, bool positive);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v DistanceToIn(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v DistanceToOut(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      Precision zMin = -kInfinity,
      Precision zMax = kInfinity) const;

  template <class Backend, class InputStruct>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v DistanceToInKernel(
      const int n,
      __restrict__ Precision const a[],
      __restrict__ Precision const b[],
      __restrict__ Precision const c[],
      __restrict__ Precision const d[],
      __restrict__ InputStruct const (&sides)[4],
      __restrict__ InputStruct const (&corners)[4],
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v DistanceToOutKernel(
      const int n,
      __restrict__ Precision const a[],
      __restrict__ Precision const b[],
      __restrict__ Precision const c[],
      __restrict__ Precision const d[],
      const Precision zMin,
      const Precision zMax,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

};

VECGEOM_CUDA_HEADER_BOTH
SOA3D<Precision> const& Quadrilaterals::GetNormals() const {
  return fPlanes.GetNormals();
}

VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Quadrilaterals::GetNormal(int i) const {
  return fPlanes.GetNormal(i);
}

VECGEOM_CUDA_HEADER_BOTH
Array<Precision> const& Quadrilaterals::GetDistances() const {
  return fPlanes.GetDistances();
}

VECGEOM_CUDA_HEADER_BOTH
Precision Quadrilaterals::GetDistance(int i) const {
  return fPlanes.GetDistance(i);
}

VECGEOM_CUDA_HEADER_BOTH
Quadrilaterals::Sides_t const& Quadrilaterals::GetSides() const {
  return fSides;
}

VECGEOM_CUDA_HEADER_BOTH
Quadrilaterals::Corners_t const& Quadrilaterals::GetCorners() const {
  return fCorners;
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToIn(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {
  return DistanceToInKernel<Backend>(
      size(), GetNormals().x(), GetNormals().y(), GetNormals().z(),
      &GetDistances()[0], fSides, fCorners, point, direction);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToOut(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    Precision zMin,
    Precision zMax) const {
  return DistanceToOutKernel<Backend>(
      size(), GetNormals().x(), GetNormals().y(), GetNormals().z(),
      &GetDistances()[0], zMin, zMax, point, direction);
}

template <class Backend, class InputStruct>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToInKernel(
    const int n,
    __restrict__ Precision const a[],
    __restrict__ Precision const b[],
    __restrict__ Precision const c[],
    __restrict__ Precision const d[],
    __restrict__ InputStruct const (&sides)[4],
    __restrict__ InputStruct const (&corners)[4],
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t bestDistance = kInfinity;

  for (int i = 0; i < n; ++i) {
    Float_t distance = 
        -(a[i]*point[0] + b[i]*point[1] + c[i]*point[2] + d[i])
       / (a[i]*direction[0] + b[i]*direction[1] + c[i]*direction[2]);
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
typename Backend::precision_v Quadrilaterals::DistanceToOutKernel(
    const int n,
    __restrict__ Precision const a[],
    __restrict__ Precision const b[],
    __restrict__ Precision const c[],
    __restrict__ Precision const d[],
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

std::ostream& operator<<(std::ostream &os, Quadrilaterals const &quads);

} // End global namespace

#endif // VECGEOM_VOLUMES_QUADRILATERALS_H_