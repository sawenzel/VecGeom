/// \file SpecializedTorus.h

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTUBE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTUBE_H_

#include "base/Global.h"

#include "volumes/kernel/TorusImplementation.h"
#include "volumes/PlacedTorus.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

  // NOTE: we may want to specialize the torus like we do for the tube
  // at the moment this is not done
template <TranslationCode transCodeT, RotationCode rotCodeT>
class SpecializedTorus
    : public ShapeImplementationHelper<TorusImplementation<
                                           transCodeT, rotCodeT> > {

  typedef ShapeImplementationHelper<TorusImplementation<
                                        transCodeT, rotCodeT> > Helper;

public:

#ifndef VECGEOM_NVCC
  SpecializedTorus(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, (PlacedBox const *const)nullptr) {}

  SpecializedTorus(LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation)
      : SpecializedTorus("", logical_volume, transformation) {}

  SpecializedTorus(char const *const label,
                  const Precision rMin, const Precision rMax, const Precision rTor,
                  const Precision sPhi, const Precision dPhi)
      : SpecializedTorus(label, new LogicalVolume(
                                   new UnplacedTorus(rMin, rMax, rTor, sPhi, dPhi)),
                        &Transformation3D::kIdentity) {}
#else
  __device__
  SpecializedTorus(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}
#endif

  virtual int memory_size() const { return sizeof(*this); }

};

typedef SpecializedTorus<translation::kGeneric, rotation::kGeneric> SimpleTorus;

} } // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
