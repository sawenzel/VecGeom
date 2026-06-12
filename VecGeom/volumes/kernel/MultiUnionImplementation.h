//===-- kernel/MultiUnionImplementation.h ---------------------------*- C++ -*-===//
//===--------------------------------------------------------------------------===//
/// @file MultiUnionImplementation.h
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_MULTIUNIONIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_MULTIUNIONIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/MultiUnionStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct MultiUnionImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, MultiUnionImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedMultiUnion;
struct MultiUnionStruct;
class UnplacedMultiUnion;

struct MultiUnionImplementation {

  using PlacedShape_t    = PlacedMultiUnion;
  using UnplacedStruct_t = MultiUnionStruct;
  using UnplacedVolume_t = UnplacedMultiUnion;

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &munion,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {
    auto containshook = [&](size_t id) {
      inside = munion.fVolumes[id]->Contains(point);
      return inside;
    };

    HybridNavigator<> *boxNav = (HybridNavigator<> *)HybridNavigator<>::Instance();
    boxNav->BVHContainsLooper(*munion.fNavHelper, point, containshook);
  }

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &munion,
                                                                  Vector3D<Real_v> const &point, Inside_v &inside)
  {
    inside          = EInside::kOutside;
    auto insidehook = [&](size_t id) {
      auto inside_crt = munion.fVolumes[id]->Inside(point);
      if (inside_crt == EInside::kInside) {
        inside = EInside::kInside;
        return true;
      }
      if (inside_crt == EInside::kSurface) inside = EInside::kSurface;

      return false;
    };

    HybridNavigator<> *boxNav = (HybridNavigator<> *)HybridNavigator<>::Instance();
    boxNav->BVHContainsLooper(*munion.fNavHelper, point, insidehook);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void InsideComponent(UnplacedStruct_t const &munion,
                                                                           Vector3D<Real_v> const &point,
                                                                           int &component)
  {
    component       = -1;
    auto insidehook = [&](size_t id) {
      auto inside_crt = munion.fVolumes[id]->Inside(point);
      if (inside_crt != EInside::kOutside) {
        component = id;
        return true;
      }
      return false;
    };

    HybridNavigator<> *boxNav = (HybridNavigator<> *)HybridNavigator<>::Instance();
    boxNav->BVHContainsLooper(*munion.fNavHelper, point, insidehook);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void InsideCluster(UnplacedStruct_t const &munion,
                                                                         Vector3D<Real_v> const &point, int &component)
  {
    // loop cluster overlap candidates for current component, then update component
    size_t *cluster = munion.fNeighbours[component];
    size_t ncluster = munion.fNneighbours[component];
    for (size_t i = 0; i < ncluster; ++i) {
      if (munion.fVolumes[cluster[i]]->Inside(point) == EInside::kInside) {
        component = cluster[i];
        return;
      }
    }
    component = -1;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &munion,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    // Check if the bounding box is hit
    const Vector3D<Real_v> invdir(Real_v(1.0) / NonZero(direction.x()), Real_v(1.0) / NonZero(direction.y()),
                                  Real_v(1.0) / NonZero(direction.z()));
    Vector3D<int> sign;
    sign[0]  = invdir.x() < 0;
    sign[1]  = invdir.y() < 0;
    sign[2]  = invdir.z() < 0;
    distance = BoxImplementation::IntersectCachedKernel2<Real_v, Real_v>(
        &munion.fMinExtent, point, invdir, sign.x(), sign.y(), sign.z(), -kTolerance, InfinityLength<Real_v>());
    if (distance >= stepMax) return;
    distance = kInfLength;
    // Lambda function to be called for each candidate selected by the bounding box navigator
    auto userhook = [&](HybridManager2::BoxIdDistancePair_t hitbox) {
      // Stop searching if the distance to the current box is bigger than the
      // requested limit or than the current distance
      if (hitbox.second > vecCore::math::Min(stepMax, distance)) return true;
      // Compute distance to the cluster (in both ToIn or ToOut assumptions)
      auto distance_crt = munion.fVolumes[hitbox.first]->DistanceToIn(point, direction, stepMax);
      if (distance_crt < distance) distance = distance_crt;
      return false;
    };

    HybridNavigator<> *boxNav = (HybridNavigator<> *)HybridNavigator<>::Instance();
    // intersect ray with the BVH structure and use hook
    boxNav->BVHSortedIntersectionsLooper(*munion.fNavHelper, point, direction, stepMax, userhook);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &munion,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    constexpr Real_v eps = 10 * kTolerance;
    distance             = -1.;
    // Locate the component containing the point
    int comp;
    InsideComponent(munion, point, comp);
    if (comp < 0) return; // Point not inside
    // Compute distance to exit current component
    distance              = -eps;
    Real_v dstep          = 1.;
    Vector3D<Real_v> pnew = point;
    Vector3D<Real_v> local, ldir;
    while (dstep > kTolerance && comp >= 0) {
      size_t component = (size_t)comp;
      munion.fVolumes[component]->GetTransformation()->Transform(pnew, local);
      munion.fVolumes[component]->GetTransformation()->TransformDirection(direction, ldir);
      dstep = munion.fVolumes[component]->DistanceToOut(local, ldir, stepMax);
      VECGEOM_ASSERT(dstep < kInfLength);
      distance += dstep + eps;
      // If no neighbours, exit
      if (!munion.fNneighbours[component]) return;
      // Propagate to exit of current component
      pnew += (dstep + eps) * direction;
      // Try to relocate inside the cluster of neighbours
      MultiUnionImplementation::InsideCluster(munion, pnew, comp);
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToInComp(UnplacedStruct_t const &munion,
                                                                          Vector3D<Real_v> const &point, Real_v &safety,
                                                                          int &component)
  {
    safety        = vecgeom::InfinityLength<Real_v>();
    component     = -1;
    auto userhook = [&](HybridManager2::BoxIdDistancePair_t hitbox) {
      // Stop searching if the safety to the current cluster is bigger than the
      // current safety
      if (hitbox.second > safety * safety) return true;
      // Compute distance to the cluster
      Real_v safetycrt = munion.fVolumes[hitbox.first]->SafetyToIn(point);
      if (safetycrt > 0 && safetycrt < safety) {
        safety    = safetycrt;
        component = hitbox.first;
      }
      return false;
    };

    HybridSafetyEstimator *safEstimator = (HybridSafetyEstimator *)HybridSafetyEstimator::Instance();
    // Use the BVH structure and connect hook
    safEstimator->BVHSortedSafetyLooper(*munion.fNavHelper, point, userhook, safety);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &munion,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    int comp;
    InsideComponent(munion, point, comp);
    if (comp > -1) {
      safety = -1.;
      return;
    }

    SafetyToInComp<Real_v>(munion, point, safety, comp);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &munion,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    // Locate the component containing the point
    int comp;
    MultiUnionImplementation::InsideComponent(munion, point, comp);
    if (comp < 0) {
      safety = -1.; // Point not inside
      return;
    }
    // Compute safety to exit current component
    Vector3D<Real_v> const local = munion.fVolumes[comp]->GetTransformation()->Transform(point);
    safety                       = munion.fVolumes[comp]->SafetyToOut(local);
    VECGEOM_ASSERT(safety > -kTolerance);
    // Loop cluster of neighbours
    size_t *cluster = munion.fNeighbours[comp];
    size_t ncluster = munion.fNneighbours[comp];
    for (size_t i = 0; i < ncluster; ++i) {
      Vector3D<Real_v> const local = munion.fVolumes[cluster[i]]->GetTransformation()->Transform(point);
      Real_v safetycrt             = munion.fVolumes[cluster[i]]->SafetyToOut(local);
      if (safetycrt > 0 && safetycrt < safety) safety = safetycrt;
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(
      UnplacedStruct_t const &munion, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &valid)
  {
    // Locate the component containing the point
    int comp;
    valid = false;
    Vector3D<Real_v> direction;

    InsideComponent(munion, point, comp);
    // If component not found, locate closest one
    Real_v safety;
    if (comp < 0) SafetyToInComp(munion, point, safety, comp);
    if (comp < 0) return direction;

    Vector3D<Real_v> local = munion.fVolumes[comp]->GetTransformation()->Transform(point);
    Vector3D<Real_v> ldir;
    valid = munion.fVolumes[comp]->GetUnplacedVolume()->Normal(local, ldir);
    if (valid) direction = munion.fVolumes[comp]->GetTransformation()->InverseTransformDirection(ldir);
    return direction;
  }

}; // end MultiUnionImplementation
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_MULTIUNIONIMPLEMENTATION_H_
