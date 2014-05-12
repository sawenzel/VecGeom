/// @file SpecializedTrapezoid.h

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTRAPEZOID_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTRAPEZOID_H_

#include "base/global.h"

#include "volumes/kernel/TrapezoidImplementation.h"
#include "volumes/PlacedTrapezoid.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
class SpecializedTrapezoid
    : public ShapeImplementationHelper<PlacedTrapezoid,
                                       TrapezoidImplementation<
                                           transCodeT, rotCodeT> > {

  typedef ShapeImplementationHelper<PlacedTrapezoid,
                                    TrapezoidImplementation<
                                        transCodeT, rotCodeT> > Helper;

public:

#ifndef VECGEOM_NVCC

  SpecializedTrapezoid(char const *const label,
                        LogicalVolume const *const logical_volume,
                        Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL) {}

  SpecializedTrapezoid(LogicalVolume const *const logical_volume,
                        Transformation3D const *const transformation)
      : SpecializedTrapezoid("", logical_volume, transformation) {}

#else

  __device__
  SpecializedTrapezoid(LogicalVolume const *const logical_volume,
                        Transformation3D const *const transformation,
                        PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;
  

};

typedef SpecializedTrapezoid<translation::kGeneric, rotation::kGeneric>
    SimpleTrapezoid;

template <TranslationCode transCodeT, RotationCode rotCodeT>
void SpecializedTrapezoid<transCodeT, rotCodeT>::PrintType() const {
  printf("SpecializedTrapezoid<%i, %i>", transCodeT, rotCodeT);
}

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDTRAPEZOID_H_