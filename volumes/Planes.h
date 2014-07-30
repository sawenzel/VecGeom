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
template <bool treatSurfaceT> struct TreatSurfaceTraits;
template <> struct TreatSurfaceTraits<true>  { typedef Inside_t Surface_t; };
template <> struct TreatSurfaceTraits<false> { typedef bool Surface_t; };
}

class Planes : public AlignedBase {

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

  friend std::ostream& operator<<(std::ostream &os, Planes const &planes);

private:

  template <bool treatSurfaceT>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void InsideKernel(
      Vector3D<Precision> const &point,
      typename TreatSurfaceTraits<treatSurfaceT>::Surface_t *inside) const;

};

Planes::Planes(size_t size) : fNormal(size), fP(NULL) {
  fP = AlignedAllocate<Precision>(size);
}

Planes::Planes(Planes const &rhs) : fNormal(), fP(NULL) {
  *this = rhs;
}

#ifdef VECGEOM_NVCC
__device__
Planes::Planes(SOA3D<Precision> const &normal, Precision *p)
    : fNormal(normal.x(), normal.y(), normal.z(), normal.size()), fP(p) {}
#endif

Planes::~Planes() {
  AlignedFree(fP);
}

Planes& Planes::operator=(Planes const &rhs) {
  fNormal = rhs.fNormal;
  copy(rhs.fP, rhs.fP+rhs.fNormal.size(), fP);
  return *this;
}

void Planes::Set(size_t index, Vector3D<Precision> const &normal,
                 Vector3D<Precision> const &x0) {
  Precision inverseLength = 1. / normal.Mag();
  fNormal.set(index, inverseLength*normal[0], inverseLength*normal[1],
              inverseLength*normal[2]);
  Precision d = -normal.Dot(x0);
  fP[index] = inverseLength * d;
}

template <bool signedT>
VECGEOM_CUDA_HEADER_BOTH
void Planes::Distance(Vector3D<Precision> const &point,
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
void Planes::SignedDistance(Vector3D<Precision> const &point,
                            Precision *distance) const {
  return Distance<true>(point, distance);
}

VECGEOM_CUDA_HEADER_BOTH
void Planes::UnsignedDistance(Vector3D<Precision> const &point,
                              Precision *distance) const {
  return Distance<false>(point, distance);
}

template <bool treatSurfaceT>
VECGEOM_CUDA_HEADER_BOTH
void Planes::InsideKernel(
    Vector3D<Precision> const &point,
    typename TreatSurfaceTraits<treatSurfaceT>::Surface_t *inside) const {
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
void Planes::Inside(Vector3D<Precision> const &point, Inside_t *inside) const {
  InsideKernel<true>(point, inside);
}

VECGEOM_CUDA_HEADER_BOTH
void Planes::Contains(Vector3D<Precision> const &point, bool *inside) const {
  InsideKernel<false>(point, inside);
}

std::ostream& operator<<(std::ostream &os, Planes const &planes) {
  for (int i = 0, iMax = planes.fNormal.size(); i < iMax; ++i) {
    os << "{" << planes.fNormal[i] << ", " << planes.fP[i] << "}\n";
  }
  return os;
}

} // End global namespace

#endif // VECGEOM_VOLUMES_PLANES_H_