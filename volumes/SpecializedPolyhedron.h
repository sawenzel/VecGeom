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

class GenericPolyhedron {};

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
      : Helper(label, logical_volume, transformation, this) {}

  SpecializedPolyhedron(LogicalVolume const *const logical_volume,
                 Transformation3D const *const transformation)
      : SpecializedPolyhedron("", logical_volume, transformation) {}

#else

  __device__
  SpecializedPolyhedron(LogicalVolume const *const logical_volume,
                        Transformation3D const *const transformation,
                        const int id)
      : Helper(logical_volume, transformation, this, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;

};

typedef SpecializedPolyhedron<GenericPolyhedron> SimplePolyhedron;

template <class PolyhedronType>
void SpecializedPolyhedron<PolyhedronType>::PrintType() const {
  printf("SpecializedPolyhedron");
}

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPOLYHEDRON_H_