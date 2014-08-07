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

template <bool treatInnerT, int sideCountT>
struct PolyhedronSpecialization {
  static const bool treatInner = treatInnerT;
  static const int sideCount = sideCountT;
};

typedef PolyhedronSpecialization<true, 0> GenericPolyhedron;

template <bool treatInnerT, int sideCountT>
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

};

typedef SpecializedPolyhedron<true, 0> SimplePolyhedron;

template <bool treatInnerT, int sideCountT>
void SpecializedPolyhedron<treatInnerT, sideCountT>::PrintType() const {
  printf("SpecializedPolyhedron<%i, %i>", treatInnerT, sideCountT);
}

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_