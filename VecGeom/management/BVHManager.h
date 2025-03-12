/// \file BVHManager.h
/// \author Guilherme Amadio

#ifndef VECGEOM_MANAGEMENT_BVHMANAGER_H_
#define VECGEOM_MANAGEMENT_BVHMANAGER_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/BVH.h"
#include "VecGeom/volumes/LogicalVolume.h"

#ifdef VECGEOM_ENABLE_CUDA
#include "VecGeom/backend/cuda/Interface.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
template <typename Real_t>
inline std::vector<BVH<Real_t> *> hBVH;
#ifdef VECGEOM_ENABLE_CUDA
template <typename Real_t>
inline __device__ BVH<Real_t> *dBVH;
#endif

// Macro allowing downstream codes to use GetDeviceBVH
#define VECGEOM_BVHMANAGER_DEVICE

/**
 * @brief The @c BVHManager class is a singleton class to manage the association between
 * logical volumes and their bounding volume hierarchies, using the logical volumes' ids.
 */
class BVHManager {

// to avoid changing the user code via templates, the precision is chosen at compile-time in the BVHManager and
// BVHNavigator
#ifdef VECGEOM_BVH_SINGLE
  using Real_t = float;
#else
  using Real_t = double;
#endif

public:
  BVHManager() = delete;

  /**
   * Initializes the bounding volume hierarchies for all logical volumes in the geometry.
   * Since it uses the ABBoxManager to fetch the pre-computed bounding boxes for each logical volume,
   * it must be called after the bounding boxes have already been computed. The depth is not specified,
   * to allow the BVH class to choose the depth dynamically based on the number of children of each
   * logical volume. This function is called automatically when the geometry is closed.
   *
   * The BVHManager assumes all volumes have an associated BVH, but only BVHs for volumes whose
   * navigator is set to the BVHNavigatorV are actually accessed at runtime.
   */
  static void Init();

  /** Initializes bounding volume hierarchies on the GPU. */
  static cuda::BVH<Real_t> const *DeviceInit();

  /** Access the device BVH pointer if CUDA is enabled. **/
  static cuda::BVH<Real_t> const *GetDeviceBVH();

  VECCORE_ATT_HOST_DEVICE
  static BVH<Real_t> const *GetBVH(int id)
  {
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
    return &cuda::dBVH<Real_t>[id];
#else
    return hBVH<Real_t>[id];
#endif
  }

  VECCORE_ATT_HOST_DEVICE
  static BVH<Real_t> const *GetBVH(LogicalVolume const *v) { return GetBVH(v->id()); }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
