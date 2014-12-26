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

//#include <ostream>

// #define VECGEOM_PLANES_VC

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class Planes; )
VECGEOM_DEVICE_DECLARE_CONV( Planes );

inline namespace VECGEOM_IMPL_NAMESPACE {

class Planes : public AlignedBase {

private:

  SOA3D<Precision> fNormals; ///< Normalized normals of the planes.
  Array<Precision> fDistances; ///< Distance from plane to origin (0, 0, 0).

public:

  VECGEOM_CUDA_HEADER_BOTH
  Planes(int size);

#ifdef VECGEOM_NVCC
  __device__
  Planes();
  __device__
  Planes(Precision *a, Precision *b, Precision *c, Precision *d, int size);
#endif

  VECGEOM_CUDA_HEADER_BOTH
  ~Planes();

  VECGEOM_CUDA_HEADER_BOTH
  Planes& operator=(Planes const &rhs);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision const* operator[](int index) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int size() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void reserve(size_t size) { fNormals.reserve(size); fDistances.Allocate(size); }

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
  void Set(
      int index,
      Vector3D<Precision> const &normal,
      Vector3D<Precision> const &origin);

  VECGEOM_CUDA_HEADER_BOTH
  void Set(
      int index,
      Vector3D<Precision> const &normal,
      Precision distance);

  /// Flip the sign of the normal and distance at the specified index
  VECGEOM_CUDA_HEADER_BOTH
  void FlipSign(int index);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::bool_v Contains(
      Vector3D<typename Backend::precision_v> const &point) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::inside_v Inside(
      Vector3D<typename Backend::precision_v> const &point) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v Distance(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction) const;

  template <bool pointInsideT, class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v Distance(
      Vector3D<typename Backend::precision_v> const &point) const;

};

std::ostream& operator<<(std::ostream &os, Planes const &planes);

VECGEOM_CUDA_HEADER_BOTH
Precision const* Planes::operator[](int i) const {
  if (i == 0) return fNormals.x();
  if (i == 1) return fNormals.y();
  if (i == 2) return fNormals.z();
  if (i == 3) return &fDistances[0];
  return NULL;
}

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

namespace {

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void AcceleratedContains(
    int &i,
    const int n,
    SOA3D<Precision> const &normals,
    Array<Precision> const &distances,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::bool_v &result) {
  return;
}

template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void AcceleratedContains<kScalar>(
    int &i,
    const int n,
    SOA3D<Precision> const &normals,
    Array<Precision> const &distances,
    Vector3D<Precision> const &point,
    bool &result) {
#ifdef VECGEOM_VC
  for (; i < n-kVectorSize; i += kVectorSize) {
    VcBool valid = VcPrecision(normals.x()+i)*point[0] +
                   VcPrecision(normals.y()+i)*point[1] +
                   VcPrecision(normals.z()+i)*point[2] +
                   VcPrecision(&distances[0]+i) < 0;
    result = IsFull(valid);
    if (!result) {
      i = n;
      break;
    }
  }
#endif
}

} // End anonymous namespace


template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::bool_v Planes::Contains(
    Vector3D<typename Backend::precision_v> const &point) const {

  typename Backend::bool_v result(true);

  int i = 0;
  const int n = size();
  // AcceleratedContains<Backend>(i, n, fNormals, fDistances, point, result);

  for (; i < n; ++i) {
    result &= point.Dot(fNormals[i]) + fDistances[i] < 0;
  }

  return result;
}

namespace {

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void AcceleratedInside(
    int &i,
    const int n,
    SOA3D<Precision> const &normals,
    Array<Precision> const &distances,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &result) {
  return;
}

template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void AcceleratedInside<kScalar>(
    int &i,
    const int n,
    SOA3D<Precision> const &normals,
    Array<Precision> const &distances,
    Vector3D<Precision> const &point,
    Inside_t &result) {
#ifdef VECGEOM_VC
  for (; i < n-kVectorSize; i += kVectorSize) {
    VcPrecision distance = VcPrecision(normals.x()+i)*point[0] +
                           VcPrecision(normals.y()+i)*point[1] +
                           VcPrecision(normals.z()+i)*point[2] +
                           VcPrecision(&distances[0]+i);
    // If point is outside tolerance of any plane, it is safe to return
    if (Any(distance > kTolerance)) {
      result = EInside::kOutside;
      i = n;
      break;
    }
    // If point is inside tolerance of all planes, keep looking
    if (IsFull(distance < -kTolerance)) continue;
    // Otherwise point must be on a surface, but could still be outside
    result = EInside::kSurface;
  }
#endif
}

} // End anonymous namespace

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::inside_v Planes::Inside(
    Vector3D<typename Backend::precision_v> const &point) const {

  typename Backend::inside_v result(EInside::kInside);

  int i = 0;
  const int n = size();
  AcceleratedInside<Backend>(i, n, fNormals, fDistances, point, result);

  for (; i < n; ++i) {
    typename Backend::precision_v distanceResult =
        fNormals.x(i)*point[0] + fNormals.y(i)*point[1] +
        fNormals.z(i)*point[2] + fDistances[i];
    MaskedAssign(distanceResult > kTolerance, EInside::kOutside,
                 &result);
    MaskedAssign(result == EInside::kInside && distanceResult > -kTolerance,
                 EInside::kSurface, &result);
    if (IsFull(result) == EInside::kOutside) break;
  }

  return result;
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Planes::Distance(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {

  typedef typename Backend::precision_v Float_t;

  Float_t bestDistance = kInfinity;
  for (int i = 0, iMax = size(); i < iMax; ++i) {
    Vector3D<Precision> normal = fNormals[i];
    Float_t distance = -(point.Dot(normal) + fDistances[i]) /
                       direction.Dot(normal);
    MaskedAssign(distance >= 0 && distance < bestDistance, distance,
                 &bestDistance);
  }

  return bestDistance;
}

template <bool pointInsideT, class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Planes::Distance(
    Vector3D<typename Backend::precision_v> const &point) const {

  typedef typename Backend::precision_v Float_t;

  Float_t bestDistance = kInfinity;
  for (int i = 0, iMax = size(); i < iMax; ++i) {
    Float_t distance = Flip<!pointInsideT>::FlipSign(
        point.Dot(fNormals[i]) + fDistances[i]);
    MaskedAssign(distance >= 0 && distance < bestDistance, distance,
                 &bestDistance);
  }
  return bestDistance;
}

} // End inline namespace

} // End global namespace

#endif // VECGEOM_VOLUMES_PLANES_H_
