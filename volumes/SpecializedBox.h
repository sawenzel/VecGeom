/// @file SpecializedBox.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDBOX_H_
#define VECGEOM_VOLUMES_SPECIALIZEDBOX_H_

#include "base/Global.h"

#include "volumes/kernel/BoxImplementation.h"
#include "volumes/PlacedBox.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
class SpecializedBox
    : public ShapeImplementationHelper<PlacedBox,
                                       BoxImplementation<
                                           transCodeT, rotCodeT> > {

  typedef ShapeImplementationHelper<PlacedBox,
                                    BoxImplementation<
                                        transCodeT, rotCodeT> > Helper;

public:

#ifndef VECGEOM_NVCC

  SpecializedBox(char const *const label,
                 LogicalVolume const *const logical_volume,
                 Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, this) {}

  SpecializedBox(LogicalVolume const *const logical_volume,
                 Transformation3D const *const transformation)
      : SpecializedBox("", logical_volume, transformation) {}

#else

  __device__
  SpecializedBox(LogicalVolume const *const logical_volume,
                 Transformation3D const *const transformation,
                 const int id)
      : Helper(logical_volume, transformation, this, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;
  

};

typedef SpecializedBox<translation::kGeneric, rotation::kGeneric> SimpleBox;

template <TranslationCode transCodeT, RotationCode rotCodeT>
void SpecializedBox<transCodeT, rotCodeT>::PrintType() const {
  printf("SpecializedBox<%i, %i>", transCodeT, rotCodeT);
}

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDBOX_H_