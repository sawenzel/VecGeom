/// \file   SpecializedTrapezoid.h
/// \author Guilherme Lima (lima 'at' fnal 'dot' gov)
/*
 * 2014-05-01 - Created, based on the Parallelepiped draft
 */

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTRAPEZOID_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTRAPEZOID_H_

#include "base/Global.h"

#include "volumes/kernel/TrapezoidImplementation.h"
#include "volumes/PlacedTrapezoid.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
class SpecializedTrapezoid
    : public ShapeImplementationHelper<TrapezoidImplementation<
                                         transCodeT, rotCodeT> > {

  typedef ShapeImplementationHelper<TrapezoidImplementation<
                                      transCodeT, rotCodeT> > Helper;

public:

#ifndef VECGEOM_NVCC

  SpecializedTrapezoid(char const *const label,
                       LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, (PlacedBox const *const)nullptr) {}

  SpecializedTrapezoid(LogicalVolume const *const logical_volume,
                        Transformation3D const *const transformation)
      : SpecializedTrapezoid("", logical_volume, transformation) {}

  SpecializedTrapezoid(char const *const label,
                       Precision pDz,
                       Precision pTheta,
                       Precision pPhi,
                       Precision pDy1,
                       Precision pDx1,
                       Precision pDx2,
                       Precision pTanAlpha1,
                       Precision pDy2,
                       Precision pDx3,
                       Precision pDx4,
                       Precision pTanAlpha2)
    : SpecializedTrapezoid(label,
                           new LogicalVolume(
                             new UnplacedTrapezoid(pDz,pTheta,pPhi,
                                                   pDy1,pDx1,pDx2,pTanAlpha1,
                                                   pDy2,pDx3,pDx4,pTanAlpha2)),
                             &Transformation3D::kIdentity) {}

  SpecializedTrapezoid(char const *const label, Precision const* params )
    : SpecializedTrapezoid(label,
                           new LogicalVolume(new UnplacedTrapezoid(params)),
                           &Transformation3D::kIdentity) {}

  SpecializedTrapezoid(char const *const label, TrapCorners_t const& corners)
    : SpecializedTrapezoid(label,
                           new LogicalVolume(new UnplacedTrapezoid(corners)),
                           &Transformation3D::kIdentity) {}

                           SpecializedTrapezoid(char const *const label)
    : SpecializedTrapezoid(label, new LogicalVolume(new UnplacedTrapezoid()),
                           &Transformation3D::kIdentity) {}

  SpecializedTrapezoid(char const *const label,
                       const Precision dX, const Precision dY, const Precision dZ)
                           : SpecializedTrapezoid(label, new LogicalVolume(new UnplacedTrapezoid(dX, dY, dZ)),
                                                  &Transformation3D::kIdentity) {}

#else
  
  __device__
  SpecializedTrapezoid(LogicalVolume const *const logical_volume,
                        Transformation3D const *const transformation,
                        PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); } 
};

typedef SpecializedTrapezoid<translation::kGeneric, rotation::kGeneric>
    SimpleTrapezoid;


} } // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDTRAPEZOID_H_
