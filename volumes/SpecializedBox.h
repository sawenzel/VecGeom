/// \file SpecializedBox.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDBOX_H_
#define VECGEOM_VOLUMES_SPECIALIZEDBOX_H_

#include "base/Global.h"

#include "volumes/kernel/BoxImplementation.h"
#include "volumes/PlacedBox.h"
#include "volumes/ShapeImplementationHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
class SpecializedBox
    : public ShapeImplementationHelper<BoxImplementation<
                                           transCodeT, rotCodeT> > {

  typedef ShapeImplementationHelper<BoxImplementation<
                                        transCodeT, rotCodeT> > Helper;

  // with this we'll be able to access the implementation struct of this shape

public:

  typedef BoxImplementation<transCodeT, rotCodeT> Implementation;

#ifndef VECGEOM_NVCC

  SpecializedBox(char const *const label,
                 LogicalVolume const *const logical_volume,
                 Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, (PlacedBox const *const)this) {}

  SpecializedBox(LogicalVolume const *const logical_volume,
                 Transformation3D const *const transformation)
      : SpecializedBox("", logical_volume, transformation) {}

  SpecializedBox(char const *const label,
                 Vector3D<Precision> const &dim)
      : SpecializedBox(label, new LogicalVolume(new UnplacedBox(dim)),
                       &Transformation3D::kIdentity) {}

  SpecializedBox(char const *const label,
                 const Precision dX, const Precision dY, const Precision dZ)
     : SpecializedBox(label, new LogicalVolume(new UnplacedBox(dX, dY, dZ)),
                      &Transformation3D::kIdentity) {}

#else

  __device__
  SpecializedBox(LogicalVolume const *const logical_volume,
                 Transformation3D const *const transformation,
                 const int id)
      : Helper(logical_volume, transformation, this, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

};

typedef SpecializedBox<translation::kGeneric, rotation::kGeneric> SimpleBox;

} } // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDBOX_H_
