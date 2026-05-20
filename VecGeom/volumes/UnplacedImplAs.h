//@author: initial implementation Sandro Wenzel (July 2018)
#ifndef VOLUMES_SUNPLACEDVOLUME_IMPLAS_H_
#define VOLUMES_SUNPLACEDVOLUME_IMPLAS_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/kernel/ImplAsImplementation.h"
#include "VecGeom/base/Assert.h"
#include <type_traits>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2t(class, SUnplacedImplAs, typename, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

// A wrapper class which generically implements a specialization of UnplacedBase
// implemented in terms of another **existing** implementing volume ImplementingUnplaced.
template <typename UnplacedBase, typename ImplementingUnplaced>
class SUnplacedImplAs : public UnplacedBase {
  // static VECGEOM_ASSERT(make sure UnplacedBase is an UnplacedVolume

public:
  using UnplacedStruct_t = typename ImplementingUnplaced::UnplacedStruct_t;

  template <typename... Args>
  SUnplacedImplAs(Args... args) : UnplacedBase(args...)
  {
    fImplPtr = new ImplementingUnplaced(args...);
  }

  // implement the important unplaced volume interfaces
  VECCORE_ATT_HOST_DEVICE
  bool Contains(Vector3D<Precision> const &p) const override { return fImplPtr->ImplementingUnplaced::Contains(p); }

  VECCORE_ATT_HOST_DEVICE
  EnumInside Inside(Vector3D<Precision> const &p) const override { return fImplPtr->ImplementingUnplaced::Inside(p); }

  // DistanceToOut
  VECCORE_ATT_HOST_DEVICE
  Precision DistanceToOut(Vector3D<Precision> const &p, Vector3D<Precision> const &d,
                          Precision step_max = kInfLength) const override
  {
    return fImplPtr->ImplementingUnplaced::DistanceToOut(p, d, step_max);
  }

  VECCORE_ATT_HOST_DEVICE
  Precision DistanceToOut(Vector3D<Precision> const &p, Vector3D<Precision> const &d, Precision step_max,
                          SurfaceHitView<Precision> *hit_info) const override
  {
    return fImplPtr->ImplementingUnplaced::DistanceToOut(p, d, step_max, hit_info);
  }

  // SafetyToOut
  VECCORE_ATT_HOST_DEVICE
  Precision SafetyToOut(Vector3D<Precision> const &p) const override
  {
    return fImplPtr->ImplementingUnplaced::SafetyToOut(p);
  }

  VECCORE_ATT_HOST_DEVICE
  Precision SafetyToIn(Vector3D<Precision> const &position) const override
  {
    return fImplPtr->ImplementingUnplaced::SafetyToIn(position);
  }

  // ---------------- SamplePointOnSurface ----------------------------------------------------------
  Vector3D<Precision> SamplePointOnSurface() const override
  {
    return fImplPtr->ImplementingUnplaced::SamplePointOnSurface();
  }

  // ----------------- Extent --------------------------------------------------------------------
  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    fImplPtr->ImplementingUnplaced::Extent(aMin, aMax);
  }

  using StructType = decltype(std::declval<ImplementingUnplaced>().GetStruct());
  VECCORE_ATT_HOST_DEVICE
  StructType GetStruct() const { return fImplPtr->ImplementingUnplaced::GetStruct(); }

private:
  // select target specialization helpers with SFINAE
  template <typename IUnplaced, typename ImplementingKernel>
  VPlacedVolume *changeTypeKeepTransformation(VPlacedVolume const *p) const
  {
    auto c = VolumeFactory::ChangeTypeKeepTransformation<SpecializedVolImplHelper, ImplementingKernel>(p);
    return c;
  }

  VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume, Transformation3D const *const transformation,
                                   VPlacedVolume *const placement = NULL) const override
  {
    auto p = fImplPtr->ImplementingUnplaced::SpecializedVolume(volume, transformation, placement);

    using ImplementingKernel = IndirectImplementation<SUnplacedImplAs<UnplacedBase, ImplementingUnplaced>,
                                                      typename ImplementingUnplaced::Kernel>;

    // the right function to use will be selected with SFINAE and enable_if
    return changeTypeKeepTransformation<ImplementingUnplaced, ImplementingKernel>(p);
    // TODO: original p is no longer needed in principle: delete and deregister from GeoManager
    // delete p;
  }

  ImplementingUnplaced *fImplPtr;
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VOLUMES_SUNPLACEDVOLUME_IMPLAS_H_ */
