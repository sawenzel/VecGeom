/// \file SpecializedPolyhedron.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_

#include "base/Global.h"

#include "volumes/kernel/PolyhedronImplementation.h"
#include "volumes/PlacedPolyhedron.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
class SpecializedPolyhedron
    : public ShapeImplementationHelper<
          PlacedPolyhedron,
          PolyhedronImplementation<innerRadiiT, phiCutoutT> > {

  typedef ShapeImplementationHelper<
      PlacedPolyhedron,
      PolyhedronImplementation<innerRadiiT, phiCutoutT> > Helper;

public:

#ifndef VECGEOM_NVCC

  SpecializedPolyhedron(
      char const *const label,
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transformation);

  SpecializedPolyhedron(
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transformation);

  SpecializedPolyhedron(
    char const *const label,
    const int sideCount,
    const int zPlaneCount,
    Precision zPlanes[],
    Precision rMin[],
    Precision rMax[]);

  SpecializedPolyhedron(
    char const *const label,
    Precision phiStart,
    Precision phiDelta,
    const int sideCount,
    const int zPlaneCount,
    Precision zPlanes[],
    Precision rMin[],
    Precision rMax[]);

#else

  __device__
  SpecializedPolyhedron(LogicalVolume const *const logical_volume,
                        Transformation3D const *const transformation,
                        const int id)
      : Helper(logical_volume, transformation, NULL, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual void PrintType() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual bool Contains(Vector3D<Precision> const &point) const;

  VECGEOM_INLINE
  virtual void Contains(SOA3D<Precision> const &point,
                        bool *const output) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual bool Contains(Vector3D<Precision> const &point,
                        Vector3D<Precision> &localPoint) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual bool UnplacedContains(Vector3D<Precision> const &localPoint) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Inside_t Inside(Vector3D<Precision> const &point) const;

  VECGEOM_INLINE
  virtual void Inside(SOA3D<Precision> const &point,
                      Inside_t *const output) const;

  using Helper::DistanceToIn;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToIn(Vector3D<Precision> const &point,
                                 Vector3D<Precision> const &direction,
                                 const Precision stepMax = kInfinity) const;

  virtual void DistanceToIn(SOA3D<Precision> const &points,
                            SOA3D<Precision> const &directions,
                            Precision const *const stepMax,
                            Precision *const output) const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToIn(Vector3D<Precision> const &point) const;

  virtual void SafetyToIn(SOA3D<Precision> const &point,
                          Precision *const safeties) const;

  virtual void SafetyToInMinimize(SOA3D<Precision> const &points,
                                  Precision *const safeties) const;

  using Helper::DistanceToOut;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &points,
                                  Vector3D<Precision> const &directions,
                                  const Precision stepMax = kInfinity) const;

  virtual void DistanceToOut(SOA3D<Precision> const &points,
                             SOA3D<Precision> const &directions,
                             Precision const *const stepMax,
                             Precision *const output) const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToOut(Vector3D<Precision> const &point) const;

  virtual void SafetyToOut(SOA3D<Precision> const &point,
                           Precision *const safeties) const;

  virtual void SafetyToOutMinimize(SOA3D<Precision> const &points,
                                   Precision *const safeties) const;

};

typedef SpecializedPolyhedron<UnplacedPolyhedron::kInnerRadiiGeneric,
                              UnplacedPolyhedron::kPhiCutoutGeneric>
    SimplePolyhedron;

#ifndef VECGEOM_NVCC

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
SpecializedPolyhedron<innerRadiiT, phiCutoutT>::SpecializedPolyhedron(
    char const *const label,
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation)
    : Helper(label, logical_volume, transformation, NULL) {}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
SpecializedPolyhedron<innerRadiiT, phiCutoutT>::SpecializedPolyhedron(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation)
    : SpecializedPolyhedron("", logical_volume, transformation) {}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
SpecializedPolyhedron<innerRadiiT, phiCutoutT>::SpecializedPolyhedron(
    char const *const label,
    const int sideCount,
    const int zPlaneCount,
    Precision zPlanes[],
    Precision rMin[],
    Precision rMax[])
    : Helper(label, new LogicalVolume(
          new UnplacedPolyhedron(sideCount, zPlaneCount, zPlanes, rMin, rMax)),
          &Transformation3D::kIdentity, NULL) {}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
SpecializedPolyhedron<innerRadiiT, phiCutoutT>::SpecializedPolyhedron(
    char const *const label,
    Precision phiStart,
    Precision phiDelta,
    const int sideCount,
    const int zPlaneCount,
    Precision zPlanes[],
    Precision rMin[],
    Precision rMax[])
    : Helper(label, new LogicalVolume(
          new UnplacedPolyhedron(
              phiStart, phiDelta, sideCount, zPlaneCount, zPlanes, rMin, rMax)),
          &Transformation3D::kIdentity, NULL) {}

#endif

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
void SpecializedPolyhedron<innerRadiiT, phiCutoutT>::PrintType() const {
  printf("SpecializedPolyhedron<%i>", innerRadiiT);
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
bool SpecializedPolyhedron<innerRadiiT, phiCutoutT>::Contains(
    Vector3D<Precision> const &point) const {
  Vector3D<Precision> localPoint;
  return Contains(point, localPoint);
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
void SpecializedPolyhedron<innerRadiiT, phiCutoutT>::Contains(
    SOA3D<Precision> const &points,
    bool *const output) const {
  for (int i = 0, iMax = points.size(); i < iMax; ++i) {
    Vector3D<Precision> localPoint =
        VPlacedVolume::transformation()->Transform(points[i]);
    output[i] =
        PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarContainsKernel(
            *PlacedPolyhedron::GetUnplacedVolume(), localPoint);
  }
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
bool SpecializedPolyhedron<innerRadiiT, phiCutoutT>::Contains(
    Vector3D<Precision> const &point,
    Vector3D<Precision> &localPoint) const {
  localPoint = VPlacedVolume::transformation()->Transform(point);
  return UnplacedContains(localPoint);
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
bool SpecializedPolyhedron<innerRadiiT, phiCutoutT>::UnplacedContains(
    Vector3D<Precision> const &localPoint) const {
  return
      PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarContainsKernel(
          *PlacedPolyhedron::GetUnplacedVolume(), localPoint);
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
Inside_t SpecializedPolyhedron<innerRadiiT, phiCutoutT>::Inside(
    Vector3D<Precision> const &point) const {
  Vector3D<Precision> localPoint =
      VPlacedVolume::transformation()->Transform(point);
  return PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarInsideKernel(
      *PlacedPolyhedron::GetUnplacedVolume(), localPoint);
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
void SpecializedPolyhedron<innerRadiiT, phiCutoutT>::Inside(
    SOA3D<Precision> const &points,
    Inside_t *const output) const {
  for (int i = 0, iMax = points.size(); i < iMax; ++i) {
    Vector3D<Precision> localPoint =
        VPlacedVolume::transformation()->Transform(points[i]);
    output[i] =
        PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarInsideKernel(
            *PlacedPolyhedron::GetUnplacedVolume(), localPoint);
  }
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
Precision SpecializedPolyhedron<innerRadiiT, phiCutoutT>::DistanceToIn(
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction,
    const Precision stepMax) const {
  Precision temp =
      PolyhedronImplementation<innerRadiiT,
                               phiCutoutT>::ScalarDistanceToInKernel(
          *PlacedPolyhedron::GetUnplacedVolume(),
          *VPlacedVolume::transformation(), point, direction, stepMax);
  return temp;
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
void SpecializedPolyhedron<innerRadiiT, phiCutoutT>::DistanceToIn(
    SOA3D<Precision> const &points,
    SOA3D<Precision> const &directions,
    Precision const *const stepMax,
    Precision *const output) const {
  for (int i = 0, iMax = points.size(); i < iMax; ++i) {
    Vector3D<Precision> localPoint =
        VPlacedVolume::transformation()->Transform(points[i]);
    Vector3D<Precision> localDirection =
        VPlacedVolume::transformation()->Transform(directions[i]);
    output[i] = PolyhedronImplementation<innerRadiiT,
                                         phiCutoutT>::ScalarDistanceToInKernel(
        *PlacedPolyhedron::GetUnplacedVolume(),
        *VPlacedVolume::transformation(),
        localPoint, localDirection, stepMax[i]);
  }
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
Precision SpecializedPolyhedron<innerRadiiT, phiCutoutT>::SafetyToIn(
    Vector3D<Precision> const &point) const {

  Vector3D<Precision> localPoint =
      VPlacedVolume::transformation()->Transform(point);

  return PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarSafetyKernel(
      *PlacedPolyhedron::GetUnplacedVolume(), localPoint);
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
void SpecializedPolyhedron<innerRadiiT, phiCutoutT>::SafetyToIn(
    SOA3D<Precision> const &points,
    Precision *const safeties) const {

  for (int i = 0, iMax = points.size(); i < iMax; ++i) {

    Vector3D<Precision> localPoint =
        VPlacedVolume::transformation()->Transform(points[i]);

    safeties[i] =
        PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarSafetyKernel(
            *PlacedPolyhedron::GetUnplacedVolume(), localPoint);
  }
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
void SpecializedPolyhedron<innerRadiiT, phiCutoutT>::SafetyToInMinimize(
    SOA3D<Precision> const &points,
    Precision *const safeties) const  {

  for (int i = 0, iMax = points.size(); i < iMax; ++i) {

    Vector3D<Precision> localPoint =
        VPlacedVolume::transformation()->Transform(points[i]);

    Precision candidate =
        PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarSafetyKernel(
            *PlacedPolyhedron::GetUnplacedVolume(), localPoint);

    safeties[i] = (candidate < safeties[i]) ? candidate : safeties[i];
  }
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
Precision SpecializedPolyhedron<innerRadiiT, phiCutoutT>::DistanceToOut(
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction,
    const Precision stepMax) const {
  return PolyhedronImplementation<innerRadiiT,
                                  phiCutoutT>::ScalarDistanceToOutKernel(
      *PlacedPolyhedron::GetUnplacedVolume(), point, direction, stepMax);
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
void SpecializedPolyhedron<innerRadiiT, phiCutoutT>::DistanceToOut(
    SOA3D<Precision> const &points,
    SOA3D<Precision> const &directions,
    Precision const *const stepMax,
    Precision *const output) const {
  for (int i = 0, iMax = points.size(); i < iMax; ++i) {
    output[i] =
        PolyhedronImplementation<innerRadiiT,
                                 phiCutoutT>::ScalarDistanceToOutKernel(
            *PlacedPolyhedron::GetUnplacedVolume(), points[i], directions[i],
            stepMax[i]);
  }
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
VECGEOM_CUDA_HEADER_BOTH
Precision SpecializedPolyhedron<innerRadiiT, phiCutoutT>::SafetyToOut(
    Vector3D<Precision> const &point) const {
  return PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarSafetyKernel(
      *PlacedPolyhedron::GetUnplacedVolume(), point);
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
void SpecializedPolyhedron<innerRadiiT, phiCutoutT>::SafetyToOut(
    SOA3D<Precision> const &points,
    Precision *const safeties) const {
  for (int i = 0, iMax = points.size(); i < iMax; ++i) {
    safeties[i] =
        PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarSafetyKernel(
            *PlacedPolyhedron::GetUnplacedVolume(), points[i]);
  }
}

template <UnplacedPolyhedron::EInnerRadii innerRadiiT,
          UnplacedPolyhedron::EPhiCutout phiCutoutT>
void SpecializedPolyhedron<innerRadiiT, phiCutoutT>::SafetyToOutMinimize(
    SOA3D<Precision> const &points,
    Precision *const safeties) const {
  for (int i = 0, iMax = points.size(); i < iMax; ++i) {
     safeties[i] =
         PolyhedronImplementation<innerRadiiT, phiCutoutT>::ScalarSafetyKernel(
            *PlacedPolyhedron::GetUnplacedVolume(), points[i]);
  }
}

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_
