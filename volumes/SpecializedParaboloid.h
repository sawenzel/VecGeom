/// @file SpecializedParaboloid.h

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPARABOLOID_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPARABOLOID_H_

#include "base/Global.h"

#include "volumes/kernel/ParaboloidImplementation.h"
#include "volumes/PlacedParaboloid.h"
#include "volumes/ShapeImplementationHelper.h"
#include "base/Transformation3D.h"
#include <stdio.h>

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
class SpecializedParaboloid
    : public ShapeImplementationHelper<PlacedParaboloid,
                                       ParaboloidImplementation<
                                           transCodeT, rotCodeT> > {

  typedef ShapeImplementationHelper<PlacedParaboloid,
                                    ParaboloidImplementation<
                                        transCodeT, rotCodeT> > Helper;

public:

#ifndef VECGEOM_NVCC

  SpecializedParaboloid(char const *const label,
                        LogicalVolume const *const logical_volume,
                        Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL) {}

  SpecializedParaboloid(LogicalVolume const *const logical_volume,
                        Transformation3D const *const transformation)
      : SpecializedParaboloid("", logical_volume, transformation) {}

  SpecializedParaboloid(char const *const label, const Precision rlo, const Precision rhi, const Precision dz)
      : SpecializedParaboloid(label, new LogicalVolume(new UnplacedParaboloid(rlo, rhi, dz)), &Transformation3D::kIdentity) {}

 
#else

  __device__
  SpecializedParaboloid(LogicalVolume const *const logical_volume,
                        Transformation3D const *const transformation,
                        PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;
  

};

typedef SpecializedParaboloid<translation::kGeneric, rotation::kGeneric>
    SimpleParaboloid;

template <TranslationCode transCodeT, RotationCode rotCodeT>
void SpecializedParaboloid<transCodeT, rotCodeT>::PrintType() const {
  printf("SpecializedParaboloid<%i, %i>", transCodeT, rotCodeT);
}

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARABOLOID_H_
