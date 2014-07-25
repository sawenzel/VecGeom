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

template <bool phiTreatmentT>
struct PolyhedronSpecialization {
  static const bool phiTreatment = phiTreatmentT;
};

typedef PolyhedronSpecialization<true> GenericPolyhedron;

template <class PolyhedronType>
class SpecializedPolyhedron
    : public ShapeImplementationHelper<
          PlacedPolyhedron, PolyhedronImplementation<PolyhedronType> > {

  typedef ShapeImplementationHelper<
      PlacedPolyhedron, PolyhedronImplementation<PolyhedronType> > Helper;

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

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual Inside_t Inside(Vector3D<Precision> const &point) const;

  VECGEOM_INLINE
  virtual void Inside(SOA3D<Precision> const &points,
                      Inside_t *const output) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual bool UnplacedContains(Vector3D<Precision> const &point) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual bool Contains(Vector3D<Precision> const &point) const;

  VECGEOM_INLINE
  virtual void Contains(SOA3D<Precision> const &points,
                        bool *const output) const;

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual void PrintType() const;

};

typedef SpecializedPolyhedron<GenericPolyhedron> SimplePolyhedron;

template <class PolyhedronType>
void SpecializedPolyhedron<PolyhedronType>::PrintType() const {
  printf("SpecializedPolyhedron<%i>", PolyhedronType::phiTreatment);
}

template <class PolyhedronType>
VECGEOM_CUDA_HEADER_BOTH
Inside_t SpecializedPolyhedron<PolyhedronType>::Inside(
    Vector3D<Precision> const &point) const {
  return PolyhedronImplementation<PolyhedronType>::InsideScalar(
           *PlacedPolyhedron::GetUnplacedVolume(),
           *VPlacedVolume::transformation(),
           point
         );
}

template <class PolyhedronType>
void SpecializedPolyhedron<PolyhedronType>::Inside(
    SOA3D<Precision> const &points,
    Inside_t *const output) const {
  for (int i = 0, iMax = points.size(); i < iMax; ++i) {
    output[i] = Inside(points[i]);
  }
}

template <class PolyhedronType>
VECGEOM_CUDA_HEADER_BOTH
bool SpecializedPolyhedron<PolyhedronType>::Contains(
    Vector3D<Precision> const &point) const {
  Vector3D<Precision> localPoint;
  return PolyhedronImplementation<PolyhedronType>::ContainsScalar(
           *PlacedPolyhedron::GetUnplacedVolume(),
           *VPlacedVolume::transformation(),
           point,
           localPoint
         );
}

template <class PolyhedronType>
VECGEOM_CUDA_HEADER_BOTH
bool SpecializedPolyhedron<PolyhedronType>::UnplacedContains(
    Vector3D<Precision> const &point) const {
  return PolyhedronImplementation<PolyhedronType>::UnplacedContainsScalar(
           *PlacedPolyhedron::GetUnplacedVolume(),
           point
         );
}

template <class PolyhedronType>
void SpecializedPolyhedron<PolyhedronType>::Contains(
    SOA3D<Precision> const &points,
    bool *const output) const {
  for (int i = 0, iMax = points.size(); i < iMax; ++i) {
    output[i] = Contains(points[i]);
  }
}

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_