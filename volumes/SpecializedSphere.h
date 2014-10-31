/// @file SpecializedSphere.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDSPHERE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDSPHERE_H_

#include "base/Global.h"
#include "backend/Backend.h"
#include "volumes/kernel/SphereImplementation.h"
#include "volumes/PlacedSphere.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
class SpecializedSphere
    : public ShapeImplementationHelper<PlacedSphere,
                                       SphereImplementation<
                                           transCodeT, rotCodeT> > {

  typedef ShapeImplementationHelper<PlacedSphere,
                                    SphereImplementation<
                                        transCodeT, rotCodeT> > Helper;

public:

#ifndef VECGEOM_NVCC

  SpecializedSphere(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL) {}

  SpecializedSphere(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : SpecializedSphere("", logical_volume, transformation) {}
  
  SpecializedSphere(char const *const label,
                 const Precision fRmin, const Precision fRmax, 
                 const Precision fSPhi, const Precision fDPhi,
                 const Precision fSTheta, const Precision fDTheta)
      : SpecializedSphere(label, new LogicalVolume(new UnplacedSphere(fRmin, fRmax, fSPhi, fDPhi, fSTheta, fDTheta)),
                       &Transformation3D::kIdentity) {}


#else

  __device__
  SpecializedSphere(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;
  

};

typedef SpecializedSphere<translation::kGeneric, rotation::kGeneric>
    SimpleSphere;

template <TranslationCode transCodeT, RotationCode rotCodeT>
void SpecializedSphere<transCodeT, rotCodeT>::PrintType() const {
  printf("SpecializedSphere<%i, %i>", transCodeT, rotCodeT);
}

} } // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDSPHERE_H_
