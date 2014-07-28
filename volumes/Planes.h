/// \file Planes.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLANES_H_
#define VECGEOM_VOLUMES_PLANES_H_

#include "base/Global.h"

#include "backend/Backend.h"
#include "base/AlignedBase.h"
#include "base/SOA3D.h"

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
  void set(size_t index, Vector3D<Precision> const &normal,
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
  Precision MinimumDistance(Vector3D<Precision> const &point) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision MinimumDistance(Vector3D<Precision> const &point,
                            size_t &index) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Inside_t Inside(Vector3D<Precision> const &point) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Inside(Vector3D<Precision> const &point, Inside_t *inside) const;

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

void Planes::set(size_t index, Vector3D<Precision> const &normal,
                 Vector3D<Precision> const &x0) {
  Precision inverseLength = 1. / std::sqrt(normal.x()*normal.x() +
                                           normal.y()*normal.y() +
                                           normal.z()*normal.z());
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
#ifdef VECGEOM_NVC
  for (size_t iMax = fNormal.size()-VcPrecision::Size; i <= iMax;
       i += VcPrecision::Size) {
    Vector3D<VcPrecision> normal(
      VcPrecision(fNormal.x()+i),
      VcPrecision(fNormal.y()+i),
      VcPrecision(fNormal.z()+i)
    );
    VcPrecision p(&fP[i]);
    VcPrecision distanceResult = normal.Dot(point) + p;
    if (signedT) {
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

VECGEOM_CUDA_HEADER_BOTH
Precision Planes::MinimumDistance(Vector3D<Precision> const &point,
                                  size_t &index) const {
  Precision *distance = AlignedAllocate<Precision>(fNormal.size());
  UnsignedDistance(point, distance);
  Precision *minimum = min_element(distance, distance+fNormal.size());
  index = minimum - distance;
  AlignedFree(distance);
  return *minimum;
}

VECGEOM_CUDA_HEADER_BOTH
Precision Planes::MinimumDistance(Vector3D<Precision> const &point) const {
  Precision *distance = AlignedAllocate<Precision>(fNormal.size());
  UnsignedDistance(point, distance);
  Precision result = *min_element(distance, distance+fNormal.size());
  AlignedFree(distance);
  return result;
}

template <bool treatSurfaceT>
VECGEOM_CUDA_HEADER_BOTH
void Planes::InsideKernel(
    Vector3D<Precision> const &point,
    typename TreatSurfaceTraits<treatSurfaceT>::Surface_t *inside) const {
  size_t i = 0;
#ifdef VECGEOM_VC
  for (size_t iMax = fNormal.size()-VcPrecision::Size; i <= iMax;
       i += VcPrecision::Size) {
    Vector3D<VcPrecision> normal(
      VcPrecision(fNormal.x()+i),
      VcPrecision(fNormal.y()+i),
      VcPrecision(fNormal.z()+i)
    );
    VcPrecision p(&fP[i]);
    VcPrecision dotProduct = normal.Dot(point);
    VcPrecision distanceResult = dotProduct + p;
    VcInside insideResult = EInside::kOutside;
    MaskedAssign(dotProduct < distanceResult, EInside::kInside,
                 &insideResult);
    if (treatSurfaceT) {
      MaskedAssign(Abs(distanceResult) < kTolerance, EInside::kSurface,
                   &insideResult);
      insideResult.store(&inside[i]);
    } else {
      for (size_t i = 0; i < VcPrecision::Size; ++i) {
        inside[i] = insideResult[i];
      }
    }
  }
#endif
  for (size_t iMax = fNormal.size(); i < iMax; ++i) {
    Vector3D<Precision> normal = fNormal[i];
    Precision dotProduct = normal.Dot(point);
    Precision distanceResult = dotProduct + fP[i];
    if (treatSurfaceT) {
      inside[i] = (Abs(distanceResult) < kTolerance) ? EInside::kSurface :
                  (dotProduct < distanceResult)      ? EInside::kInside
                                                     : EInside::kOutside;
    } else {
      inside[i] = (dotProduct <= distanceResult)
                  ? EInside::kInside : EInside::kOutside;
    }
  }
}

VECGEOM_CUDA_HEADER_BOTH
void Planes::Inside(Vector3D<Precision> const &point, Inside_t *inside) const {
  InsideKernel<true>(point, inside);
}

VECGEOM_CUDA_HEADER_BOTH
Inside_t Planes::Inside(Vector3D<Precision> const &point) const {
  Inside_t *inside = AlignedAllocate<Inside_t>(fNormal.size());
  InsideKernel<true>(point, inside);
  Inside_t result = EInside::kInside;
  for (int i = 0, iMax = fNormal.size(); i < iMax; ++i) {
    if (inside[i] == EInside::kSurface || inside[i] == EInside::kOutside) {
      result = inside[i];
      break;
    }
  }
  AlignedFree(inside);
  return result;
}

} // End global namespace

#endif // VECGEOM_VOLUMES_PLANES_H_