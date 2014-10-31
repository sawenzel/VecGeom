/// \file SpecializedTube.h
/// \author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTUBE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTUBE_H_

#include "base/Global.h"

#include "volumes/kernel/TubeImplementation.h"
#include "volumes/PlacedTube.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT, typename tubeTypeT>
class SpecializedTube
    : public ShapeImplementationHelper<PlacedTube,
                                       TubeImplementation<
                                           transCodeT, rotCodeT, tubeTypeT> > {

  typedef ShapeImplementationHelper<PlacedTube,
                                    TubeImplementation<
                                        transCodeT, rotCodeT, tubeTypeT> > Helper;

public:

  typedef TubeImplementation<transCodeT, rotCodeT, tubeTypeT> Implementation;

#ifndef VECGEOM_NVCC

  SpecializedTube(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL) {}

  SpecializedTube(LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation)
      : SpecializedTube("", logical_volume, transformation) {}

  SpecializedTube(char const *const label,
                  const Precision rMin, const Precision rMax, const Precision z,
                  const Precision sPhi, const Precision dPhi)
      : SpecializedTube(label, new LogicalVolume(
                                   new UnplacedTube(rMin, rMax, z, sPhi, dPhi)),
                        &Transformation3D::kIdentity) {}

#else

  __device__
  SpecializedTube(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;
  

};

typedef SpecializedTube<translation::kGeneric, rotation::kGeneric, TubeTypes::UniversalTube>
    SimpleTube;

template <TranslationCode transCodeT, RotationCode rotCodeT, typename tubeTypeT>
void SpecializedTube<transCodeT, rotCodeT, tubeTypeT>::PrintType() const {
  printf("SpecializedTube<%i, %i, %s>", transCodeT, rotCodeT, tubeTypeT::toString());
}

} } // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
