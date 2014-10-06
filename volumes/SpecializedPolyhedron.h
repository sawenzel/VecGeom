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

template <bool treatInnerT, unsigned sideCountT>
struct PolyhedronSpecialization {
  static const bool treatInner = treatInnerT;
  static const int sideCount = sideCountT;
};

typedef PolyhedronSpecialization<true, 0> GenericPolyhedron;

template <bool treatInnerT, unsigned sideCountT>
class SpecializedPolyhedron
    : public ShapeImplementationHelper<
          PlacedPolyhedron,
          PolyhedronImplementation<treatInnerT, sideCountT> > {

  typedef ShapeImplementationHelper<
      PlacedPolyhedron,
      PolyhedronImplementation<treatInnerT, sideCountT> > Helper;

public:

#ifndef VECGEOM_NVCC

  SpecializedPolyhedron(char const *const label,
                        LogicalVolume const *const logical_volume,
                        Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL) {}

  SpecializedPolyhedron(LogicalVolume const *const logical_volume,
                        Transformation3D const *const transformation)
      : SpecializedPolyhedron("", logical_volume, transformation) {}

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

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToIn(Vector3D<Precision> const &point,
                                 Vector3D<Precision> const &direction,
                                 const Precision stepMax = kInfinity) const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToIn(Vector3D<Precision> const &point) const;

  virtual void SafetyToIn(SOA3D<Precision> const &point,
                          Precision *const safeties) const;

  virtual void SafetyToInMinimize(SOA3D<Precision> const &points,
                                  Precision *const safeties) const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &point,
                                  Vector3D<Precision> const &direction,
                                  const Precision stepMax = kInfinity) const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToOut(Vector3D<Precision> const &point) const;

  virtual void SafetyToOut(SOA3D<Precision> const &point,
                           Precision *const safeties) const;

  virtual void SafetyToOutMinimize(SOA3D<Precision> const &points,
                                   Precision *const safeties) const;

};

typedef SpecializedPolyhedron<true, 0> SimplePolyhedron;

template <bool treatInnerT, unsigned sideCountT>
void SpecializedPolyhedron<treatInnerT, sideCountT>::PrintType() const {
  printf("SpecializedPolyhedron<%i, %i>", treatInnerT, sideCountT);
}

template <bool treatInnerT, unsigned sideCountT>
VECGEOM_CUDA_HEADER_BOTH
bool SpecializedPolyhedron<treatInnerT, sideCountT>::Contains(
    Vector3D<Precision> const &point) const {
  Vector3D<Precision> localPoint;
  return Contains(point, localPoint);
}

template <bool treatInnerT, unsigned sideCountT>
void SpecializedPolyhedron<treatInnerT, sideCountT>::Contains(
    SOA3D<Precision> const &points,
    bool *const output) const {
  for (int i = 0, iMax = points.size(); i < iMax; ++i) {
    Vector3D<Precision> localPoint =
        VPlacedVolume::transformation()->Transform(points[i]);
    PolyhedronImplementation<treatInnerT, sideCountT>::template
    ScalarInsideKernel<false>(*PlacedPolyhedron::GetUnplacedVolume(),
                              localPoint, output[i]);
  }
}

template <bool treatInnerT, unsigned sideCountT>
VECGEOM_CUDA_HEADER_BOTH
bool SpecializedPolyhedron<treatInnerT, sideCountT>::Contains(
    Vector3D<Precision> const &point,
    Vector3D<Precision> &localPoint) const {
  localPoint = VPlacedVolume::transformation()->Transform(point);
  return UnplacedContains(localPoint);
}

template <bool treatInnerT, unsigned sideCountT>
VECGEOM_CUDA_HEADER_BOTH
bool SpecializedPolyhedron<treatInnerT, sideCountT>::UnplacedContains(
    Vector3D<Precision> const &localPoint) const {
  bool result = false;
  PolyhedronImplementation<treatInnerT, sideCountT>::template
      ScalarInsideKernel<false>(*PlacedPolyhedron::GetUnplacedVolume(),
                                localPoint, result);
  return result;
}

template <bool treatInnerT, unsigned sideCountT>
VECGEOM_CUDA_HEADER_BOTH
Inside_t SpecializedPolyhedron<treatInnerT, sideCountT>::Inside(
    Vector3D<Precision> const &point) const {
  Vector3D<Precision> localPoint =
      VPlacedVolume::transformation()->Transform(point);
  Inside_t result = EInside::kOutside;
  PolyhedronImplementation<treatInnerT, sideCountT>::template
      ScalarInsideKernel<true>(*PlacedPolyhedron::GetUnplacedVolume(),
                               localPoint, result);
  return result;
}

template <bool treatInnerT, unsigned sideCountT>
void SpecializedPolyhedron<treatInnerT, sideCountT>::Inside(
    SOA3D<Precision> const &points,
    Inside_t *const output) const {
  for (int i = 0, iMax = points.size(); i < iMax; ++i) {
    Vector3D<Precision> localPoint =
        VPlacedVolume::transformation()->Transform(points[i]);
    PolyhedronImplementation<treatInnerT, sideCountT>::template
        ScalarInsideKernel<true>(
            *PlacedPolyhedron::GetUnplacedVolume(), localPoint, output[i]);
  }
}

template <bool treatInnerT, unsigned sideCountT>
VECGEOM_CUDA_HEADER_BOTH
Precision SpecializedPolyhedron<treatInnerT, sideCountT>::DistanceToIn(
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction,
    const Precision stepMax) const {
  Vector3D<Precision> localPoint =
      VPlacedVolume::transformation()->Transform(point);
  Vector3D<Precision> localDirection =
      VPlacedVolume::transformation()->TransformDirection(direction);
  return PolyhedronImplementation<treatInnerT,
      sideCountT>::ScalarDistanceToInKernel(
          *PlacedPolyhedron::GetUnplacedVolume(), localPoint, localDirection,
          stepMax);
}

template <bool treatInnerT, unsigned sideCountT>
VECGEOM_CUDA_HEADER_BOTH
Precision SpecializedPolyhedron<treatInnerT, sideCountT>::SafetyToIn(
    Vector3D<Precision> const &point) const {

  Vector3D<Precision> localPoint =
      VPlacedVolume::transformation()->Transform(point);

  Precision result = kInfinity;
  PolyhedronImplementation<treatInnerT, sideCountT>::ScalarSafetyToInKernel(
      *PlacedPolyhedron::GetUnplacedVolume(), localPoint, result);
  return result;
}

template <bool treatInnerT, unsigned sideCountT>
void SpecializedPolyhedron<treatInnerT, sideCountT>::SafetyToIn(
    SOA3D<Precision> const &points,
    Precision *const safeties) const {

  for (int i = 0, iMax = points.size(); i < iMax; ++i) {

    Vector3D<Precision> localPoint =
        VPlacedVolume::transformation()->Transform(points[i]);

    PolyhedronImplementation<treatInnerT, sideCountT>::ScalarSafetyToInKernel(
        *PlacedPolyhedron::GetUnplacedVolume(), localPoint, safeties[i]);
  }
}

template <bool treatInnerT, unsigned sideCountT>
void SpecializedPolyhedron<treatInnerT, sideCountT>::SafetyToInMinimize(
    SOA3D<Precision> const &points,
    Precision *const safeties) const  {

  for (int i = 0, iMax = points.size(); i < iMax; ++i) {

    Vector3D<Precision> localPoint =
        VPlacedVolume::transformation()->Transform(points[i]);

    Precision candidate = kInfinity;
    PolyhedronImplementation<treatInnerT, sideCountT>::ScalarSafetyToInKernel(
        *PlacedPolyhedron::GetUnplacedVolume(), localPoint, candidate);

    safeties[i] = (candidate < safeties[i]) ? candidate : safeties[i];
  }
}

template <bool treatInnerT, unsigned sideCountT>
VECGEOM_CUDA_HEADER_BOTH
Precision SpecializedPolyhedron<treatInnerT, sideCountT>::DistanceToOut(
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction,
    const Precision stepMax) const {
  return PolyhedronImplementation<treatInnerT,
      sideCountT>::ScalarDistanceToOutKernel(
          *PlacedPolyhedron::GetUnplacedVolume(), point, direction, stepMax);
}

template <bool treatInnerT, unsigned sideCountT>
VECGEOM_CUDA_HEADER_BOTH
Precision SpecializedPolyhedron<treatInnerT, sideCountT>::SafetyToOut(
    Vector3D<Precision> const &point) const {
  Precision result = kInfinity;
  PolyhedronImplementation<treatInnerT, sideCountT>::ScalarSafetyToOutKernel(
      *PlacedPolyhedron::GetUnplacedVolume(), point, result);
  return result;
}

template <bool treatInnerT, unsigned sideCountT>
void SpecializedPolyhedron<treatInnerT, sideCountT>::SafetyToOut(
    SOA3D<Precision> const &points,
    Precision *const safeties) const {

  for (int i = 0, iMax = points.size(); i < iMax; ++i) {

    PolyhedronImplementation<treatInnerT, sideCountT>::ScalarSafetyToOutKernel(
        *PlacedPolyhedron::GetUnplacedVolume(), points[i], safeties[i]);
  }
}

template <bool treatInnerT, unsigned sideCountT>
void SpecializedPolyhedron<treatInnerT, sideCountT>::SafetyToOutMinimize(
    SOA3D<Precision> const &points,
    Precision *const safeties) const {

  for (int i = 0, iMax = points.size(); i < iMax; ++i) {

    Precision candidate = kInfinity;
    PolyhedronImplementation<treatInnerT, sideCountT>::ScalarSafetyToOutKernel(
        *PlacedPolyhedron::GetUnplacedVolume(), points[i], candidate);

    safeties[i] = (candidate < safeties[i]) ? candidate : safeties[i];
  }
}

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_