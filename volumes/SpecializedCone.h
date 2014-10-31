/*
 * SpecializedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */


#ifndef VECGEOM_VOLUMES_SPECIALIZEDCONE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDCONE_H_

#include "base/Global.h"

#include "volumes/kernel/ConeImplementation.h"
#include "volumes/PlacedCone.h"
#include "volumes/ScalarShapeImplementationHelper.h"
#include "base/SOA3D.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT,
          typename ConeType>
class SpecializedCone : public ScalarShapeImplementationHelper<PlacedCone,
                                       ConeImplementation<transCodeT, rotCodeT, ConeType> >
{

  typedef ScalarShapeImplementationHelper<PlacedCone,
                                    ConeImplementation<
                                        transCodeT, rotCodeT, ConeType> > Helper;
  typedef ConeImplementation<transCodeT, rotCodeT, ConeType> Specialization;

public:

#ifndef VECGEOM_NVCC

  SpecializedCone(char const *const label,
                  LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL) {}

  SpecializedCone(LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation)
      : SpecializedCone("", logical_volume, transformation) {}

  SpecializedCone(char const *const label,
                   const Precision rmin1,
                   const Precision rmax1,
                   const Precision rmin2,
                   const Precision rmax2,
                   const Precision dz,
                   const Precision sphi=0,
                   const Precision dphi=kTwoPi)
        : SpecializedCone(label,
                new LogicalVolume(new UnplacedCone(rmin1, rmax1, rmin2, rmax2, dz, sphi, dphi )),
                &Transformation3D::kIdentity) {}

#else

  __device__
  SpecializedCone(LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation,
                  PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual ~SpecializedCone() {}

};

typedef SpecializedCone<translation::kGeneric, rotation::kGeneric, ConeTypes::UniversalCone>
    SimpleCone;
typedef SpecializedCone<translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>
    SimpleUnplacedCone;


template <TranslationCode transCodeT, RotationCode rotCodeT, typename ConeType>
void SpecializedCone<transCodeT, rotCodeT, ConeType>::PrintType() const {
  printf("SpecializedCone<%i, %i>", transCodeT, rotCodeT);
}

} } // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
