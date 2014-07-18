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

  Inside_t output = EInside::kOutside;
  Precision bestDistance = kInfinity;

  Vector3D<Precision> localPoint =
      VPlacedVolume::transformation()->Transform(point);

  Array<UnplacedPolyhedron::PolyhedronSegment> const &segments =
      PlacedPolyhedron::GetUnplacedVolume()->GetSegments();
  for (Array<UnplacedPolyhedron::PolyhedronSegment>::const_iterator s =
       segments.cbegin(), sEnd = segments.cend(); s != sEnd; ++s) {
    Inside_t insideResult;
    Precision distanceResult;
    PolyhedronImplementation<PolyhedronType>::InsideSegment(
      *PlacedPolyhedron::GetUnplacedVolume(), *s, localPoint,
      insideResult, distanceResult
    );
    if (insideResult == EInside::kSurface) return EInside::kSurface;
    if (distanceResult < bestDistance) {
      output = insideResult;
      bestDistance = distanceResult;
    }
  }
  return output;
}

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_