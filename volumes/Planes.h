/// \file Planes.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLANES_H_
#define VECGEOM_VOLUMES_PLANES_H_

#include "base/Global.h"

#include "backend/Backend.h"
#include "base/AlignedBase.h"
#include "base/SOA3D.h"

#include <ostream>

// #define VECGEOM_PLANES_VC

namespace VECGEOM_NAMESPACE {

#ifdef VECGEOM_STD_CXX11
template <int N = 0>
#else
template <int N>
#endif
class Planes : public AlignedBase {

private:

  // Parameters of normalized plane equation
  // n0*x + n1*y + n2*z + p = 0
  Precision fA[N], fB[N], fC[N], fP[N];

public:

  Planes() {}

  Planes(Planes<N> const &rhs);

  ~Planes() {}

  Planes& operator=(Planes<N> const &rhs);

  Vector3D<Precision> GetNormal(size_t index) const;

  Precision GetDistance(size_t index) const;

  void Set(size_t index, Vector3D<Precision> const &normal,
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

  // Kernel containing the actual computations, which can be utilized from
  // other classes if necessary.

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::precision_v DistanceToOutKernel(
      Precision const *fA,
      Precision const *fB,
      Precision const *fC,
      Precision const *fP,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::inside_v InsideKernel(
      Precision const *fA,
      Precision const *fB,
      Precision const *fC,
      Precision const *fP,
      Vector3D<typename Backend::precision_v> const &point);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::bool_v ContainsKernel(
      Precision const *fA,
      Precision const *fB,
      Precision const *fC,
      Precision const *fP,
      Vector3D<typename Backend::precision_v> const &point);

};

template <int N>
Planes<N>::Planes(Planes<N> const &rhs) {
  *this = rhs;
}

template <int N>
Planes<N>& Planes<N>::operator=(Planes<N> const &rhs) {
  copy(rhs.fA, rhs.fA+N, fA);
  copy(rhs.fB, rhs.fB+N, fB);
  copy(rhs.fC, rhs.fC+N, fC);
  copy(rhs.fP, rhs.fP+N, fP);
  return *this;
}

template <int N>
Vector3D<Precision> Planes<N>::GetNormal(size_t index) const {
  return Vector3D<Precision>(fA[index], fB[index], fC[index]);
}

template <int N>
Precision Planes<N>::GetDistance(size_t index) const {
  return fP[index];
}

template <int N>
void Planes<N>::Set(
    size_t index,
    Vector3D<Precision> const &normal,
    Vector3D<Precision> const &origin) {
  // Normalize plane equation by length of normal
  Precision inverseLength = 1. / normal.Mag();
  fA[index] = inverseLength*normal[0];
  fB[index] = inverseLength*normal[1];
  fC[index] = inverseLength*normal[2];
  Precision d = -normal.Dot(origin);
  fP[index] = inverseLength * d;
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
typename Backend::precision_v Planes<N>::DistanceToOut(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {
  return DistanceToOutKernel<Backend>(fA, fB, fC, fP, point, direction);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::inside_v Planes<N>::Inside(
    Vector3D<typename Backend::precision_v> const &point) const {
  return InsideKernel<Backend>(fA, fB, fC, fP, point);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::bool_v Planes<N>::Contains(
    Vector3D<typename Backend::precision_v> const &point) const {
  return ContainsKernel<Backend>(fA, fB, fC, fP, point);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::bool_v Planes<N>::ContainsKernel(
    Precision const *fA,
    Precision const *fB,
    Precision const *fC,
    Precision const *fP,
    Vector3D<typename Backend::precision_v> const &point) {
  typedef typename Backend::bool_v Bool_t;
  Bool_t contains[N];
  for (int i = 0; i < N; ++i) {
    contains[i] = (fA[i]*point[0] + fB[i]*point[1] +
                   fC[i]*point[2] + fP[i]) <= 0;
  }
  return all_of(contains, contains+N);
}

template <int N>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::inside_v Planes<N>::InsideKernel(
    Precision const *fA,
    Precision const *fB,
    Precision const *fC,
    Precision const *fP,
    Vector3D<typename Backend::precision_v> const &point) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::inside_v Inside_t;

  Inside_t result = EInside::kInside;
  Inside_t inside[N];
  for (int i = 0; i < N; ++i) {
    Float_t distance = fA[i]*point[0] + fB[i]*point[1] + fC[i]*point[2] + fP[i];
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
VECGEOM_INLINE
typename Backend::precision_v Planes<N>::DistanceToOutKernel(
    Precision const *fA,
    Precision const *fB,
    Precision const *fC,
    Precision const *fP,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t bestDistance = kInfinity;
  Float_t distance[N];
  Bool_t valid[N];
  for (int i = 0; i < N; ++i) {
    distance[i] = -(fA[i]*point[0] + fB[i]*point[1] + fC[i]*point[2] + fP[i])
                 / (fA[i]*direction[0] + fB[i]*direction[1] +
                    fC[i]*direction[2]);
    valid[i] = distance[i] >= 0;
  }
  for (int i = 0; i < N; ++i) {
    MaskedAssign(valid[i] && distance[i] < bestDistance, distance[i],
                 &bestDistance);
  }
  return bestDistance;
}

/// \brief Specialization that allows the size to be specified at runtime.
template <>
class Planes<0> : public AlignedBase {

private:

  SOA3D<Precision> fNormal;
  Precision *fP;

public:

  Planes(size_t size);

  Planes(Planes const &rhs);

#ifdef VECGEOM_NVCC
  __device__ Planes(SOA3D<Precision> const &normal, Precision *p);
#endif

  ~Planes();

  Planes& operator=(Planes const &rhs);

  Vector3D<Precision> GetNormal(size_t index) const;

  Precision GetDistance(size_t index) const;

  VECGEOM_INLINE
  void Set(size_t index, Vector3D<Precision> const &normal,
           Vector3D<Precision> const &origin);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Inside(Vector3D<Precision> const &point, Inside_t *inside) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Contains(Vector3D<Precision> const &point, bool *inside) const;

private:

  template <bool treatSurfaceT>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void InsideKernel(
      Vector3D<Precision> const &point,
      typename TreatSurfaceTraits<treatSurfaceT, kScalar>::Surface_t
      *inside) const;

};

Planes<0>::Planes(size_t size) : fNormal(size), fP(NULL) {
  fP = AlignedAllocate<Precision>(size);
}

Planes<0>::Planes(Planes const &rhs) : fNormal(), fP(NULL) {
  *this = rhs;
}

#ifdef VECGEOM_NVCC
template <>
__device__
Planes<0>::Planes(SOA3D<Precision> const &normal, Precision *p)
    : fNormal(normal.x(), normal.y(), normal.z(), normal.size()), fP(p) {}
#endif

Planes<0>::~Planes() {
  AlignedFree(fP);
}

Planes<0>& Planes<0>::operator=(Planes<0> const &rhs) {
  fNormal = rhs.fNormal;
  copy(rhs.fP, rhs.fP+rhs.fNormal.size(), fP);
  return *this;
}

Vector3D<Precision> Planes<0>::GetNormal(size_t index) const {
  return fNormal[index];
}

Precision Planes<0>::GetDistance(size_t index) const {
  return fP[index];
}

void Planes<0>::Set(
    size_t index,
    Vector3D<Precision> const &normal,
    Vector3D<Precision> const &x0) {
  Precision inverseLength = 1. / normal.Mag();
  fNormal.set(index, inverseLength*normal[0], inverseLength*normal[1],
              inverseLength*normal[2]);
  Precision d = -normal.Dot(x0);
  fP[index] = inverseLength * d;
}

template <bool treatSurfaceT>
VECGEOM_CUDA_HEADER_BOTH
void Planes<0>::InsideKernel(
    Vector3D<Precision> const &point,
    typename TreatSurfaceTraits<treatSurfaceT, kScalar>::Surface_t
    *inside) const {
  int i = 0;
#ifdef VECGEOM_PLANES_VC
  for (int iMax = fNormal.size()-VcPrecision::Size; i <= iMax;
       i += VcPrecision::Size) {
    Vector3D<VcPrecision> normal(
      VcPrecision(fNormal.x()+i),
      VcPrecision(fNormal.y()+i),
      VcPrecision(fNormal.z()+i)
    );
    VcPrecision p(&fP[i]);
    VcPrecision distanceResult = normal.Dot(point) + p;
    VcInside insideResult = EInside::kOutside;
    MaskedAssign(distanceResult < 0., EInside::kInside, &insideResult);
    if (treatSurfaceT) {
      MaskedAssign(Abs(distanceResult) < kTolerance, EInside::kSurface,
                   &insideResult);
    }
    for (int j = 0; j < VcPrecision::Size; ++j) inside[i+j] = insideResult[j];
  }
#endif
  for (int iMax = fNormal.size(); i < iMax; ++i) {
    Vector3D<Precision> normal = fNormal[i];
    Precision distanceResult = normal.Dot(point) + fP[i];
    if (treatSurfaceT) {
      inside[i] = (distanceResult < -kTolerance) ? EInside::kInside :
                  (distanceResult >  kTolerance) ? EInside::kOutside
                                                 : EInside::kSurface;
    } else {
      inside[i] = (distanceResult <= 0.) ? EInside::kInside : EInside::kOutside;
    }
  }
}

VECGEOM_CUDA_HEADER_BOTH
void Planes<0>::Inside(
    Vector3D<Precision> const &point,
    Inside_t *inside) const {
  InsideKernel<true>(point, inside);
}

VECGEOM_CUDA_HEADER_BOTH
void Planes<0>::Contains(Vector3D<Precision> const &point, bool *inside) const {
  InsideKernel<false>(point, inside);
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