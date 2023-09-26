/// \file BVHSafetyEstimator.h
/// \author Guilherme Amadio

#ifndef VECGEOM_NAVIGATION_BVHSAFETYESTIMATOR_H_
#define VECGEOM_NAVIGATION_BVHSAFETYESTIMATOR_H_

#include "VecGeom/management/BVHManager.h"
#include "VecGeom/navigation/VSafetyEstimator.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Safety estimator class using the bounding volume hierarchy of each
 * logical volume for acceleration.
 */

class BVHSafetyEstimator : public VSafetyEstimatorHelper<BVHSafetyEstimator> {
private:
  /** Constructor. Private since this is a singleton class accessed only via the @c Instance() static method. */
  VECCORE_ATT_DEVICE
  BVHSafetyEstimator() : VSafetyEstimatorHelper<BVHSafetyEstimator>() {}

public:
  static constexpr const char *gClassNameString = "BVHSafetyEstimator";

#ifndef VECCORE_CUDA
  /** Return instance of this singleton class. */
  static VSafetyEstimator *Instance()
  {
    static BVHSafetyEstimator instance;
    return &instance;
  }
#else
  // If used on device, this needs to be implemented in a .cu file rather than in this header
  // This hack is used also by NewSimpleNavigator, implemented in LogicalVolume.cpp
  // This is now implemented in BVHManager.cu
  VECCORE_ATT_DEVICE
  static VSafetyEstimator *Instance();
#endif

  VECCORE_ATT_HOST_DEVICE
  static Precision CandidateSafetyToIn(int aItemIndex, int index, Vector3D<Precision> localpoint)
  {
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
    return vecgeom::globaldevicegeomdata::gDeviceLogicalVolumes[aItemIndex].
                                          GetDaughters()[index]->SafetyToIn(localpoint);
#else
    return GeoManager::Instance().FindLogicalVolume(aItemIndex)->GetDaughters()[index]->SafetyToIn(localpoint);
#endif
  };

  /**
   * Compute safety of a point given in the local coordinates of the placed volume @p pvol.
   * @param[in] localpoint Point in the local coordinates of the placed volume.
   * @param[in] pvol Placed volume.
   */
  VECCORE_ATT_HOST_DEVICE
  Precision ComputeSafetyForLocalPoint(Vector3D<Precision> const &localpoint, 
                                       VPlacedVolume const *pvol) const final
  {
    Precision safety = pvol->SafetyToOut(localpoint);

    if (safety > 0.0 && pvol->GetDaughters().size() > 0)
      safety = BVHManager::GetBVH(pvol->GetLogicalVolume())->ComputeSafety<BVHSafetyEstimator>(localpoint, safety);

    return safety;
  }

  /**
   * Compute safety of a point given in the local coordinates of the logical volume @p lvol against
   * all its child volumes. Uses the bounding volume hierarchy associated with the logical volume
   * for acceleration.
   * @param[in] localpoint Point in the local coordinates of the placed volume.
   * @param[in] lvol Logical volume.
   */
  VECCORE_ATT_HOST_DEVICE
  Precision ComputeSafetyToDaughtersForLocalPoint(Vector3D<Precision> const &localpoint,
                                                  LogicalVolume const *lvol) const final
  {
    return BVHManager::GetBVH(lvol)->ComputeSafety<BVHSafetyEstimator>(localpoint, kInfLength);
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
