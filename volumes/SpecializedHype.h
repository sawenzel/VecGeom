/// @file SpecializedHype.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDHYPE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDHYPE_H_

#include "base/Global.h"
#include "backend/Backend.h"
#include "volumes/kernel/HypeImplementation.h"
#include "volumes/PlacedHype.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
class SpecializedHype
    : public ShapeImplementationHelper<PlacedHype,
                                       HypeImplementation<
                                           transCodeT, rotCodeT> > {

  typedef ShapeImplementationHelper<PlacedHype,
                                    HypeImplementation<
                                        transCodeT, rotCodeT> > Helper;

public:

#ifndef VECGEOM_NVCC

  SpecializedHype(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL) {}

  SpecializedHype(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : SpecializedHype("", logical_volume, transformation) {}
  
  SpecializedHype(char const *const label,
                 const Precision fRmin, const Precision stIn, 
                 const Precision fRmax, const Precision stOut,
                 const Precision fDz)
      : SpecializedHype(label, new LogicalVolume(new UnplacedHype(fRmin, stIn, fRmax, stOut, fDz)),  &Transformation3D::kIdentity) {}


#else

  __device__
  SpecializedHype(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;
  

};

typedef SpecializedHype<translation::kGeneric, rotation::kGeneric>
    SimpleHype;

template <TranslationCode transCodeT, RotationCode rotCodeT>
void SpecializedHype<transCodeT, rotCodeT>::PrintType() const {
  printf("SpecializedHype<%i, %i>", transCodeT, rotCodeT);
}

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDHYPE_H_
