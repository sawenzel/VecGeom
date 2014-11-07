#ifndef VECGEOM_VOLUMES_SPECIALIZEDBOOLEANMINUS_H
#define VECGEOM_VOLUMES_SPECIALIZEDBOOLEANMINUS_H

#include "base/Global.h"

#include "volumes/kernel/BooleanMinusImplementation.h"
#include "volumes/UnplacedBooleanMinusVolume.h"
#include "volumes/PlacedBooleanMinusVolume.h"
#include "volumes/ScalarShapeImplementationHelper.h"

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
class SpecializedBooleanMinusVolume
    : public ScalarShapeImplementationHelper<PlacedBooleanMinusVolume,
                                       BooleanMinusImplementation<transCodeT, rotCodeT> >
{

  typedef ScalarShapeImplementationHelper<PlacedBooleanMinusVolume,
                                    BooleanMinusImplementation<transCodeT, rotCodeT> > Helper;

 // typedef TUnplacedBooleanMinusVolume<LeftUnplacedVolume_t, RightPlacedVolume_t> UnplacedVol_t;
 typedef UnplacedBooleanMinusVolume UnplacedVol_t;

public:

#ifndef VECGEOM_NVCC

  SpecializedBooleanMinusVolume(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL) {}

  SpecializedBooleanMinusVolume(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : SpecializedBooleanMinusVolume("", logical_volume, transformation) {}

  virtual ~SpecializedBooleanMinusVolume() {}
#else

  __device__
  SpecializedBooleanMinusVolume(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const { printf("NOT IMPLEMENTED"); };
  

}; // endofclassdefinition

typedef SpecializedBooleanMinusVolume<translation::kGeneric, rotation::kGeneric> GenericPlacedBooleanMinusVolume;

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDBOOLEANMINUS_H
