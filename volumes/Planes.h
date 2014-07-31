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

namespace {
template <bool treatSurfaceT, class Backend> struct TreatSurfaceTraits;
template <class Backend> struct TreatSurfaceTraits<true, Backend> {
  typedef typename Backend::inside_v Surface_t;
};
template <class Backend> struct TreatSurfaceTraits<false, Backend> {
  typedef typename Backend::bool_v Surface_t;
};
}

template <int N>
class Planes : public AlignedBase {

private:

  Precision fA[N], fB[N], fC[N], fP[N];

public:

  Planes() {}

  Planes(Planes<N> const &rhs);

  ~Planes() {}

  Planes& operator=(Planes<N> const &rhs);

  Vector3D<Precision> GetNormal(size_t index) const;

  Precision GetDistance(size_t index) const;

  VECGEOM_INLINE
  void Set(size_t index, Vector3D<Precision> const &normal,
           Vector3D<Precision> const &origin);

  template <bool signedT, class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename TreatSurfaceTraits<signedT, Backend>::Surface_t Distance(
      Vector3D<typename Backend::float_v> const &point) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::float_v SignedDistance(
      Vector3D<typename Backend::float_v> const &point) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::float_v UnsignedDistance(
      Vector3D<typename Backend::float_v> const &point) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::inside_v Inside(
      Vector3D<typename Backend::float_v> const &point) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  typename Backend::bool_v Contains(
      Vector3D<typename Backend::float_v> const &point) const;

};

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

  template <bool signedT>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Distance(Vector3D<Precision> const &point, Precision *distance) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SignedDistance(Vector3D<Precision> const &point,
                      Precision *distance) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void UnsignedDistance(Vector3D<Precision> const &point,
                        Precision *distance) const;

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

template <bool signedT>
VECGEOM_CUDA_HEADER_BOTH
void Planes<0>::Distance(
    Vector3D<Precision> const &point,
    Precision *distance) const {
  size_t i = 0;
#ifdef VECGEOM_PLANES_VC
  for (size_t iMax = fNormal.size()-VcPrecision::Size; i <= iMax;
       i += VcPrecision::Size) {
    Vector3D<VcPrecision> normal(
      VcPrecision(fNormal.x()+i),
      VcPrecision(fNormal.y()+i),
      VcPrecision(fNormal.z()+i)
    );
    VcPrecision p(&fP[i]);
    VcPrecision distanceResult = normal.Dot(point) + p;
    if (!signedT) {
      distanceResult = Abs(distanceResult);
    }
    distanceResult.store(&distance[i]);
  }
#endif
  for (size_t iMax = fNormal.size(); i < iMax; ++i) {
    Vector3D<Precision> normal = fNormal[i];
    distance[i] = normal.Dot(point) + fP[i];
  }
}

VECGEOM_CUDA_HEADER_BOTH
void Planes<0>::SignedDistance(
    Vector3D<Precision> const &point,
    Precision *distance) const {
  return Distance<true>(point, distance);
}

VECGEOM_CUDA_HEADER_BOTH
void Planes<0>::UnsignedDistance(
    Vector3D<Precision> const &point,
    Precision *distance) const {
  return Distance<false>(point, distance);
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