#ifndef VECGEOM_VOLUMES_TSPECIALIZEDBOOLEANMINUS_H
#define VECGEOM_VOLUMES_TSPECIALIZEDBOOLEANMINUS_H

#include "base/Global.h"

#include "volumes/kernel/TBooleanMinusImplementation.h"
#include "volumes/TUnplacedBooleanMinusVolume.h"
#include "volumes/TPlacedBooleanMinusVolume.h"
#include "volumes/ShapeImplementationHelper.h"

namespace VECGEOM_NAMESPACE {

template <typename LeftUnplacedVolume_t, typename RightPlacedVolume_t, TranslationCode transCodeT, RotationCode rotCodeT>
class TSpecializedBooleanMinusVolume
    : public ShapeImplementationHelper<TPlacedBooleanMinusVolume,
                                       TBooleanMinusImplementation<LeftUnplacedVolume_t, RightPlacedVolume_t, transCodeT, rotCodeT> >
{

  typedef ShapeImplementationHelper<TPlacedBooleanMinusVolume,
                                    TBooleanMinusImplementation< LeftUnplacedVolume_t, RightPlacedVolume_t, transCodeT, rotCodeT> > Helper;

 // typedef TUnplacedBooleanMinusVolume<LeftUnplacedVolume_t, RightPlacedVolume_t> UnplacedVol_t;
 typedef TUnplacedBooleanMinusVolume UnplacedVol_t;

public:

#ifndef VECGEOM_NVCC

  TSpecializedBooleanMinusVolume(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL) {}

  TSpecializedBooleanMinusVolume(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : TSpecializedBooleanMinusVolume("", logical_volume, transformation) {}

  virtual ~TSpecializedBooleanMinusVolume() {}

#else

  __device__
  TSpecializedBooleanMinusVolume(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const { printf("NOT IMPLEMENTED"); };
  

}; // endofclassdefinition


} // End global namespace

#endif // VECGEOM_VOLUMES_TSPECIALIZEDBOOLEANMINUS_H 
