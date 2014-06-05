/// @file SpecializedTrd.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTRD_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTRD_H_

#include "base/Global.h"

#include "volumes/kernel/TrdImplementation.h"
#include "volumes/PlacedTrd.h"
#include "volumes/ShapeImplementationHelper.h"
#include <stdio.h>

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT, typename trdTypeT>
class SpecializedTrd
    : public ShapeImplementationHelper<PlacedTrd,
                                       TrdImplementation<
                                           transCodeT, rotCodeT, trdTypeT> > {

  typedef ShapeImplementationHelper<PlacedTrd,
                                    TrdImplementation<
                                        transCodeT, rotCodeT, trdTypeT> > Helper;

public:

#ifndef VECGEOM_NVCC

  SpecializedTrd(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL) {}

  SpecializedTrd(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : SpecializedTrd("", logical_volume, transformation) {}

#else

  __device__
  SpecializedTrd(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;
  

};

typedef SpecializedTrd<translation::kGeneric, rotation::kGeneric, TrdTypes::UniversalTrd>
    SimpleTrd;

template <TranslationCode transCodeT, RotationCode rotCodeT, typename trdTypeT>
void SpecializedTrd<transCodeT, rotCodeT, trdTypeT>::PrintType() const {
  printf("SpecializedTrd<%i, %i, %s>", transCodeT, rotCodeT, trdTypeT::toString());
}

} // End global namespace



#endif // VECGEOM_VOLUMES_SPECIALIZEDTRD_H_