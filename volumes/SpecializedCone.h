/*
 * SpecializedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */


#ifndef VECGEOM_VOLUMES_SPECIALIZEDCONE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDCONE_H_

#include "base/global.h"

#include "volumes/kernel/ConeImplementation.h"
#include "volumes/PlacedCone.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT,
          typename ConeType>
class SpecializedCone
    : public ShapeImplementationHelper<PlacedCone,
                                       ConeImplementation<
                                           transCodeT, rotCodeT, ConeType> >
{

  typedef ShapeImplementationHelper<PlacedCone,
                                    ConeImplementation<
                                        transCodeT, rotCodeT, ConeType> > Helper;

public:

#ifndef VECGEOM_NVCC

  SpecializedCone(char const *const label,
                  LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL) {}

  SpecializedCone(LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation)
      : SpecializedCone("", logical_volume, transformation) {}

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


};

//typedef SpecializedCone<translation::kGeneric, rotation::kGeneric>
//    SimpleParallelepiped;

template <TranslationCode transCodeT, RotationCode rotCodeT, typename ConeType>
void SpecializedCone<transCodeT, rotCodeT, ConeType>::PrintType() const {
  printf("SpecializedCone<%i, %i>", transCodeT, rotCodeT);
}

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
