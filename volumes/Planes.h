/// \file Planes.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLANES_H_
#define VECGEOM_VOLUMES_PLANES_H_

#include "base/Global.h"

#include "backend/Backend.h"
#include "base/AlignedBase.h"
#include "base/Array.h"
#include "base/SOA3D.h"
#include "volumes/kernel/GenericKernels.h"

#include <ostream>

// #define VECGEOM_PLANES_VC

namespace VECGEOM_NAMESPACE {

class Planes : public AlignedBase {

private:

  SOA3D<Precision> fNormals;
  Array<Precision> fDistances;

public:

  Planes(int size);

  ~Planes();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int size() const;

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

  void Set(
      int index,
      Vector3D<Precision> const &normal,
      Vector3D<Precision> const &origin);

  void Set(
      int index,
      Vector3D<Precision> const &normal,
      Precision distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::bool_v Contains(
      Vector3D<typename Backend::precision_v> const &point);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::inside_v Inside(
      Vector3D<typename Backend::precision_v> const &point);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v Distance(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v Distance(
      Vector3D<typename Backend::precision_v> const &point) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::bool_v ContainsKernel(
      const int size,
      __restrict__ Precision const a[],
      __restrict__ Precision const b[],
      __restrict__ Precision const c[],
      __restrict__ Precision const d[],
      Vector3D<typename Backend::precision_v> const &point);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::inside_v InsideKernel(
      const int size,
      __restrict__ Precision const a[],
      __restrict__ Precision const b[],
      __restrict__ Precision const c[],
      __restrict__ Precision const d[],
      Vector3D<typename Backend::precision_v> const &point);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::precision_v DistanceKernel(
      int size,
      __restrict__ Precision const a[],
      __restrict__ Precision const b[],
      __restrict__ Precision const c[],
      __restrict__ Precision const d[],
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

  template <bool pointInsideT, class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::precision_v DistanceKernel(
      const int n,
      __restrict__ Precision const a[],
      __restrict__ Precision const b[],
      __restrict__ Precision const c[],
      __restrict__ Precision const d[],
      Vector3D<typename Backend::precision_v> const &point);

};

VECGEOM_CUDA_HEADER_BOTH
int Planes::size() const {
  return fNormals.size();
}

VECGEOM_CUDA_HEADER_BOTH
SOA3D<Precision> const& Planes::GetNormals() const {
  return fNormals;
}

VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Planes::GetNormal(int i) const {
  return fNormals[i];
}

VECGEOM_CUDA_HEADER_BOTH
Array<Precision> const& Planes::GetDistances() const {
  return fDistances;
}

VECGEOM_CUDA_HEADER_BOTH
Precision Planes::GetDistance(int i) const {
  return fDistances[i];
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::bool_v Planes::Contains(
    Vector3D<typename Backend::precision_v> const &point) {
  return ContainsKernel<Backend>(
      size(), fNormals.x(), fNormals.y(), fNormals.z(), &fDistances[0], point);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::inside_v Planes::Inside(
    Vector3D<typename Backend::precision_v> const &point) {
  return InsideKernel<Backend>(
      size(), fNormals.x(), fNormals.y(), fNormals.z(), &fDistances[0], point);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Planes::Distance(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {
  return DistanceKernel<Backend>(
      size(), fNormals.x(), fNormals.y(), fNormals.z(), &fDistances[0],
      point, direction);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Planes::Distance(
    Vector3D<typename Backend::precision_v> const &point) const {
  return DistanceKernel<Backend>(
      size(), fNormals.x(), fNormals.y(), fNormals.z(), &fDistances[0], point);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::bool_v Planes::ContainsKernel(
    const int size,
    __restrict__ Precision const a[],
    __restrict__ Precision const b[],
    __restrict__ Precision const c[],
    __restrict__ Precision const d[],
    Vector3D<typename Backend::precision_v> const &point) {

  typename Backend::bool_v result(true);
  for (int i = 0; i < size; ++i) {
    result &= a[i]*point[0] + b[i]*point[1] + c[i]*point[2] + d[i] < 0;
  }
  return result;
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::inside_v Planes::InsideKernel(
    const int size,
    __restrict__ Precision const a[],
    __restrict__ Precision const b[],
    __restrict__ Precision const c[],
    __restrict__ Precision const d[],
    Vector3D<typename Backend::precision_v> const &point) {

  typename Backend::inside_v result(EInside::kInside);
  for (int i = 0; i < size; ++i) {
    typename Backend::precision_v distanceResult =
        a[i]*point[0] + b[i]*point[1] + c[i]*point[2] + d[i];
    typename Backend::bool_v notSurface = result != EInside::kSurface;
    MaskedAssign(notSurface && distanceResult > kTolerance, EInside::kOutside,
                 &result);
  }
  return result;
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Planes::DistanceKernel(
    const int size,
    __restrict__ Precision const a[],
    __restrict__ Precision const b[],
    __restrict__ Precision const c[],
    __restrict__ Precision const d[],
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;

  Float_t bestDistance = kInfinity;
  for (int i = 0; i < size; ++i) {
    Float_t distance = -(a[i]*point[0] + b[i]*point[1] + c[i]*point[2] + d[i])
                      / (a[i]*direction[0] + b[i]*direction[1] +
                         c[i]*direction[2]);
    MaskedAssign(distance >= 0 && distance < bestDistance, distance,
                 &bestDistance);
  }

  return bestDistance;
}

template <bool pointInsideT, class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v Planes::DistanceKernel(
    const int n,
    __restrict__ Precision const a[],
    __restrict__ Precision const b[],
    __restrict__ Precision const c[],
    __restrict__ Precision const d[],
    Vector3D<typename Backend::precision_v> const &point) {

  typedef typename Backend::precision_v Float_t;

  Float_t bestDistance = kInfinity;
  for (int i = 0; i < n; ++i) {
    Float_t distance = FlipSign<!pointInsideT>::Flip(
        a[i]*point[0] + b[i]*point[1] + c[i]*point[2] + d[i]);
    MaskedAssign(distance >= 0 && distance < bestDistance, distance,
                 &bestDistance);
  }
  return bestDistance;
}

std::ostream& operator<<(std::ostream &os, Planes const &planes);

} // End global namespace

#endif // VECGEOM_VOLUMES_PLANES_H_