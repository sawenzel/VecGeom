#ifndef VECGEOM_VOLUMES_SPECIALIZEDBOOLEAN_H
#define VECGEOM_VOLUMES_SPECIALIZEDBOOLEAN_H

#include "base/Global.h"

#include "volumes/kernel/BooleanImplementation.h"
#include "volumes/UnplacedBooleanVolume.h"
#include "volumes/PlacedBooleanVolume.h"
#include "volumes/ScalarShapeImplementationHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_3v(SpecializedBooleanVolume, BooleanOperation,boolOp, TranslationCode,transCodeT, RotationCode,rotCodeT)

inline namespace VECGEOM_IMPL_NAMESPACE {

template <BooleanOperation boolOp, TranslationCode transCodeT, RotationCode rotCodeT>
class SpecializedBooleanVolume
    : public ScalarShapeImplementationHelper<BooleanImplementation<boolOp, transCodeT, rotCodeT> >
{

  typedef ScalarShapeImplementationHelper<BooleanImplementation<boolOp, transCodeT, rotCodeT> > Helper;

 // typedef TUnplacedBooleanVolume<LeftUnplacedVolume_t, RightPlacedVolume_t> UnplacedVol_t;
 typedef UnplacedBooleanVolume UnplacedVol_t;

public:

#ifndef VECGEOM_NVCC

  SpecializedBooleanVolume(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, (PlacedBox const *const)nullptr) {}

  SpecializedBooleanVolume(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : SpecializedBooleanVolume("", logical_volume, transformation) {}

  virtual ~SpecializedBooleanVolume() {}
#else

  __device__
  SpecializedBooleanVolume(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}

#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const { printf("NOT IMPLEMENTED"); };

}; // endofclassdefinition

typedef SpecializedBooleanVolume<kUnion, translation::kGeneric, rotation::kGeneric> GenericPlacedUnionVolume;
typedef SpecializedBooleanVolume<kIntersection, translation::kGeneric, rotation::kGeneric> GenericPlacedIntersectionVolume;
typedef SpecializedBooleanVolume<kSubtraction, translation::kGeneric, rotation::kGeneric> GenericPlacedSubtractionVolume;

typedef SpecializedBooleanVolume<kUnion, translation::kIdentity, rotation::kIdentity> GenericUnionVolume;
typedef SpecializedBooleanVolume<kIntersection, translation::kIdentity, rotation::kIdentity> GenericIntersectionVolume;
typedef SpecializedBooleanVolume<kSubtraction, translation::kIdentity, rotation::kIdentity> GenericSubtractionVolume;



} // End impl namespace

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDBOOLEAN_H
