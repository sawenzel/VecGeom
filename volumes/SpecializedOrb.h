/// @file SpecializedOrb.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDORB_H_
#define VECGEOM_VOLUMES_SPECIALIZEDORB_H_

#include "base/Global.h"

#include "volumes/kernel/OrbImplementation.h"
#include "volumes/PlacedOrb.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
class SpecializedOrb
    : public ShapeImplementationHelper<PlacedOrb,
                                       OrbImplementation<
                                           transCodeT, rotCodeT> > {

  typedef ShapeImplementationHelper<PlacedOrb,
                                    OrbImplementation<
                                        transCodeT, rotCodeT> > Helper;

public:

#ifndef VECGEOM_NVCC

  SpecializedOrb(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL) {}

  SpecializedOrb(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : SpecializedOrb("", logical_volume, transformation) {}
  
  SpecializedOrb(char const *const label,
                 const Precision fR)
      : SpecializedOrb(label, new LogicalVolume(new UnplacedOrb(fR)),
                       &Transformation3D::kIdentity) {}

  

#else
  __device__
  SpecializedOrb(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}
#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;
  

};

typedef SpecializedOrb<translation::kGeneric, rotation::kGeneric>
    SimpleOrb;

template <TranslationCode transCodeT, RotationCode rotCodeT>
void SpecializedOrb<transCodeT, rotCodeT>::PrintType() const {
  printf("SpecializedOrb<%i, %i>", transCodeT, rotCodeT);
}

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDORB_H_
