/// \file Planes.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLANES_H_
#define VECGEOM_VOLUMES_PLANES_H_

#include "base/Global.h"

#include "backend/Backend.h"
#include "base/AlignedBase.h"
#include "base/SOA.h"
#include "base/SOA3D.h"
#include "volumes/kernel/GenericKernels.h"

#include <ostream>

// #define VECGEOM_PLANES_VC

namespace VECGEOM_NAMESPACE {

#ifdef VECGEOM_STD_CXX11
template <int N = 0>
#else
template <int N>
#endif
class Planes : public AlignedBase {

public:

  /// Parameters of normalized plane equation
  /// n0*x + n1*y + n2*z + p = 0
  typedef SOA<Precision, 4, N> Plane_t;

private:
  Plane_t fPlane;

public:

  Planes() {}

  ~Planes() {}

  VECGEOM_CUDA_HEADER_BOTH
  Plane_t& operator[](int index) { return fPlane[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  Plane_t const& operator[](int index) const { return fPlane[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> GetNormal(int index) const;

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDistance(int index) const;

  void Set(
      int index, Vector3D<Precision> const &normal,
      Vector3D<Precision> const &origin);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::inside_v Inside(
      Vector3D<typename Backend::precision_v> const &point) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::bool_v Contains(
      Vector3D<typename Backend::precision_v> const &point) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::precision_v DistanceToOut(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const;

  // Kernels containing the actual computations, which can be utilized from
  // other classes if necessary.

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::precision_v DistanceToOutKernel(
      int size,
      Precision const (&a)[N],
      Precision const (&b)[N],
      Precision const (&c)[N],
      Precision const (&d)[N],
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::inside_v InsideKernel(
      const int n,
      Precision const (&a)[N],
      Precision const (&b)[N],
      Precision const (&c)[N],
      Precision const (&d)[N],
      Vector3D<typename Backend::precision_v> const &point);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::bool_v ContainsKernel(
      const int n,
      Precision const (&a)[N],
      Precision const (&b)[N],
      Precision const (&c)[N],
      Precision const (&d)[N],
      Vector3D<typename Backend::precision_v> const &point);

  template <bool pointInsideT, class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::precision_v DistanceKernel(
      const int n,
      Precision const (&a)[N],
      Precision const (&b)[N],
      Precision const (&c)[N],
      Precision const (&d)[N],
      Vector3D<typename Backend::precision_v> const &point);

};

template <int N>
Vector3D<Precision> Planes<N>::GetNormal(int index) const {
  return Vector3D<Precision>(fPlane[0][index], fPlane[1][index],
                             fPlane[2][index]);
}

template <int N>
Precision Planes<N>::GetDistance(int index) const {
  return fPlane[3][index];
}

template <int N>
void Planes<N>::Set(
    int index,
    Vector3D<Precision> const &normal,
    Vector3D<Precision> const &origin) {
  // Normalize plane equation by length of normal
  Precision inverseLength = 1. / normal.Mag();
  fPlane[0][index] = inverseLength*normal[0];
  fPlane[1][index] = inverseLength*normal[1];
  fPlane[2][index] = inverseLength*normal[2];
  Precision d = -normal.Dot(origin);
  fPlane[3][index] = inverseLength * d;
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v Planes<N>::DistanceToOut(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {
  return DistanceToOutKernel<Backend>(
      N, fPlane[0], fPlane[1], fPlane[2], fPlane[3], point, direction);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::inside_v Planes<N>::Inside(
    Vector3D<typename Backend::precision_v> const &point) const {
  return InsideKernel<Backend>(
      N, fPlane[0], fPlane[1], fPlane[2], fPlane[3], point);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::bool_v Planes<N>::Contains(
    Vector3D<typename Backend::precision_v> const &point) const {
  return ContainsKernel<Backend>(
      N, fPlane[0], fPlane[1], fPlane[2], fPlane[3], point);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::bool_v Planes<N>::ContainsKernel(
    const int n,
    Precision const (&a)[N],
    Precision const (&b)[N],
    Precision const (&c)[N],
    Precision const (&d)[N],
    Vector3D<typename Backend::precision_v> const &point) {
  typedef typename Backend::bool_v Bool_t;
  Bool_t contains[N];
  for (int i = 0; i < N; ++i) {
    contains[i] = (a[i]*point[0] + b[i]*point[1] +
                   c[i]*point[2] + d[i]) <= 0;
  }
  return all_of(contains, contains+N);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::inside_v Planes<N>::InsideKernel(
    const int n,
    Precision const (&a)[N],
    Precision const (&b)[N],
    Precision const (&c)[N],
    Precision const (&d)[N],
    Vector3D<typename Backend::precision_v> const &point) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::inside_v Inside_t;

  Inside_t result = EInside::kInside;
  Inside_t inside[N];
  for (int i = 0; i < N; ++i) {
    Float_t distance = a[i]*point[0] + b[i]*point[1] + c[i]*point[2] + d[i];
    inside[i] = EInside::kSurface;
    MaskedAssign(distance < -kTolerance, EInside::kInside,  &inside[i]);
    MaskedAssign(distance >  kTolerance, EInside::kOutside, &inside[i]);
  }
  for (int i = 0; i < N; ++i) {
    MaskedAssign(inside[i] == EInside::kSurface, EInside::kSurface, &result);
    MaskedAssign(result != EInside::kSurface && inside[i] == EInside::kOutside,
                 EInside::kOutside, &result);
  }
  return result;
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Planes<N>::DistanceToOutKernel(
    const int size,
    Precision const (&a)[N],
    Precision const (&b)[N],
    Precision const (&c)[N],
    Precision const (&d)[N],
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t bestDistance = kInfinity;
  Float_t distance[N];
  Bool_t valid[N];
  for (int i = 0; i < N; ++i) {
    distance[i] = -(a[i]*point[0] + b[i]*point[1] +
                    c[i]*point[2] + d[i])
                 / (a[i]*direction[0] + b[i]*direction[1] +
                    c[i]*direction[2]);
    valid[i] = distance[i] >= 0;
  }
  for (int i = 0; i < N; ++i) {
    MaskedAssign(valid[i] && distance[i] < bestDistance, distance[i],
                 &bestDistance);
  }
  return bestDistance;
}

template <int N>
template <bool pointInsideT, class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v Planes<N>::DistanceKernel(
    const int n,
    Precision const (&a)[N],
    Precision const (&b)[N],
    Precision const (&c)[N],
    Precision const (&d)[N],
    Vector3D<typename Backend::precision_v> const &point) {

  typedef typename Backend::precision_v Float_t;

  Float_t distance[N];
  for (int i = 0; i < N; ++i) {
    distance[i] = a[i]*point[0] + b[i]*point[1] + c[i]*point[2] + d[i];
  }
  Float_t bestDistance = kInfinity;
  for (int i = 0; i < N; ++i) {
    MaskedAssign(
      PointInsideTraits<pointInsideT, Backend>::IsBetterDistance(
          distance[i], bestDistance),
      distance[i],
      &bestDistance
    );
  }
  return bestDistance;
}

/// \brief Specialization that allows the size to be specified at runtime.
template <>
class Planes<0> : public AlignedBase {

private:

  int fSize;
  Precision *fPlane[4];

public:

  inline Planes(int size);

  inline ~Planes();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int size() const { return fSize; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> GetNormal(int index) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDistance(int index) const;

  inline void Set(
      int index,
      Vector3D<Precision> const &normal,
      Vector3D<Precision> const &origin);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::bool_v ContainsKernel(
      const int size,
      Precision const a[],
      Precision const b[],
      Precision const c[],
      Precision const d[],
      Vector3D<typename Backend::precision_v> const &point);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::inside_v InsideKernel(
      const int size,
      Precision const a[],
      Precision const b[],
      Precision const c[],
      Precision const d[],
      Vector3D<typename Backend::precision_v> const &point);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::precision_v DistanceToOutKernel(
      int size,
      Precision const a[],
      Precision const b[],
      Precision const c[],
      Precision const d[],
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

  template <bool pointInsideT, class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::precision_v DistanceKernel(
      const int n,
      Precision const a[],
      Precision const b[],
      Precision const c[],
      Precision const d[],
      Vector3D<typename Backend::precision_v> const &point);

};

Planes<0>::Planes(int size) : fSize(size) {
  fPlane[0] = AlignedAllocate<Precision>(size);
  fPlane[1] = AlignedAllocate<Precision>(size);
  fPlane[2] = AlignedAllocate<Precision>(size);
  fPlane[3] = AlignedAllocate<Precision>(size);
}

Planes<0>::~Planes() {
  AlignedFree(fPlane[0]);
  AlignedFree(fPlane[1]);
  AlignedFree(fPlane[2]);
  AlignedFree(fPlane[3]);
}

Vector3D<Precision> Planes<0>::GetNormal(int index) const {
  return Vector3D<Precision>(fPlane[0][index], fPlane[1][index],
                             fPlane[2][index]);;
}

Precision Planes<0>::GetDistance(int index) const {
  return fPlane[3][index];
}

void Planes<0>::Set(
    int index,
    Vector3D<Precision> const &normal,
    Vector3D<Precision> const &x0) {
  Precision inverseLength = 1. / normal.Mag();
  fPlane[0][index] = inverseLength*normal[0];
  fPlane[1][index] = inverseLength*normal[1];
  fPlane[2][index] = inverseLength*normal[2];
  Precision d = -normal.Dot(x0);
  fPlane[3][index] = inverseLength * d;
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::bool_v Planes<0>::ContainsKernel(
    const int size,
    Precision const a[],
    Precision const b[],
    Precision const c[],
    Precision const d[],
    Vector3D<typename Backend::precision_v> const &point) {

  typename Backend::bool_v result(true);
  for (int i = 0; i < size; ++i) {
    typename Backend::precision_v distanceResult = 
        a[i]*point[0] + b[i]*point[1] + c[i]*point[2] + d[i];
    result &= distanceResult < -kTolerance;
  }
  return result;
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::inside_v Planes<0>::InsideKernel(
    const int size,
    Precision const a[],
    Precision const b[],
    Precision const c[],
    Precision const d[],
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
typename Backend::precision_v Planes<0>::DistanceToOutKernel(
    int size,
    Precision const a[],
    Precision const b[],
    Precision const c[],
    Precision const d[],
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t bestDistance(kInfinity);
  Float_t *distance = AlignedAllocate<Float_t>(size);
  Bool_t *valid = AlignedAllocate<Bool_t>(size);
  for (int i = 0; i < size; ++i) {
    distance[i] = -(a[i]*point[0] + b[i]*point[1] +
                    c[i]*point[2] + d[i])
                 / (a[i]*direction[0] + b[i]*direction[1] +
                    c[i]*direction[2]);
    valid[i] = distance[i] >= 0;
  }
  for (int i = 0; i < size; ++i) {
    MaskedAssign(valid[i] && distance[i] < bestDistance, distance[i],
                 &bestDistance);
  }
  AlignedFree(distance);
  AlignedFree(valid);
  return bestDistance;
}

template <bool pointInsideT, class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v Planes<0>::DistanceKernel(
    const int n,
    Precision const a[],
    Precision const b[],
    Precision const c[],
    Precision const d[],
    Vector3D<typename Backend::precision_v> const &point) {

  typedef typename Backend::precision_v Float_t;

  Float_t distance, bestDistance = kInfinity;
  for (int i = 0; i < n; ++i) {
    distance = a[i]*point[0] + b[i]*point[1] + c[i]*point[2] + d[i];
    MaskedAssign(
      PointInsideTraits<pointInsideT, Backend>::IsBetterDistance(
          distance, bestDistance),
      distance,
      &bestDistance
    );
  }
  return bestDistance;
}

template <int N>
std::ostream& operator<<(std::ostream &os, Planes<N> const &planes) {
  for (int i = 0; i < N; ++i) {
    os << "{" << planes.GetNormal(i) << ", " << planes.GetDistance(i) << "}\n";
  }
  return os;
}

} // End global namespace

#endif // VECGEOM_VOLUMES_PLANES_H_