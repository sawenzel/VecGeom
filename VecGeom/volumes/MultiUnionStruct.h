#ifndef VECGEOM_VOLUMES_MULTIUNIONSTRUCT_H_
#define VECGEOM_VOLUMES_MULTIUNIONSTRUCT_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/management/HybridManager2.h"
#include "VecGeom/navigation/HybridNavigator2.h"
#include "VecGeom/management/ABBoxManager.h"
#ifndef VECCORE_CUDA
#include <atomic>
#endif

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 @brief Struct containing placed volume components representing a multiple union solid
 @author mihaela.gheata@cern.ch
*/

struct MultiUnionStruct {
  template <typename U>
  using vector_t     = vecgeom::Vector<U>;
  using BVHStructure = HybridManager2::HybridBoxAccelerationStructure;

  vector_t<VPlacedVolume const *> fVolumes; ///< Component placed volumes
  BVHStructure *fNavHelper = nullptr;       ///< Navigation helper using bounding boxes

  Vector3D<Precision> fMinExtent;      ///< Minimum extent
  Vector3D<Precision> fMaxExtent;      ///< Maximum extent
  mutable Precision fCapacity    = -1; ///< Capacity of the multiple union
  mutable Precision fSurfaceArea = -1; ///< Surface area of the multiple union
#ifndef VECCORE_CUDA
  mutable std::atomic<size_t> fLast; ///< Last located component for opportunistic relocation
#endif
  size_t **fNeighbours = nullptr; ///< Array of lists of neigbours
  size_t *fNneighbours = nullptr; ///< Number of neighbours for each component

  size_t *fBuffer = nullptr; ///< Scratch space for storing neighbours

  VECCORE_ATT_HOST_DEVICE
  MultiUnionStruct()
  {
    fMinExtent.Set(kInfLength);
    fMaxExtent.Set(-kInfLength);
#ifndef VECCORE_CUDA
    fLast.store(0);
#endif
  }

  VECCORE_ATT_HOST_DEVICE
  ~MultiUnionStruct()
  {
    delete[] fNeighbours;
    delete[] fNneighbours;
    delete[] fBuffer;
  }

  VECCORE_ATT_HOST_DEVICE
  void AddNode(VPlacedVolume const *volume)
  {
    using vecCore::math::Max;
    using vecCore::math::Min;
    Vector3D<Precision> amin, amax;
    ABBoxManager<Precision>::ComputeABBox(volume, &amin, &amax);
    fMinExtent.Set(Min(fMinExtent.x(), amin.x()), Min(fMinExtent.y(), amin.y()), Min(fMinExtent.z(), amin.z()));
    fMaxExtent.Set(Max(fMaxExtent.x(), amax.x()), Max(fMaxExtent.y(), amax.y()), Max(fMaxExtent.z(), amax.z()));
    fVolumes.push_back(volume);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool ABBoxOverlap(Vector3D<Precision> const &amin1, Vector3D<Precision> const &amax1,
                    Vector3D<Precision> const &amin2, Vector3D<Precision> const &amax2)
  {
    // Check if two aligned boxes overlap
    if ((amax1 - amin2).Min() < -kTolerance || (amax2 - amin1).Min() < -kTolerance) return false;
    return true;
  }

  VECCORE_ATT_HOST_DEVICE
  void Close()
  {
    // This method prepares the navigation structure
    using Boxes_t           = ABBoxManager<Precision>::ABBoxContainer_t;
    using BoxCorner_t       = ABBoxManager<Precision>::ABBox_s;
    size_t nboxes           = fVolumes.size();
    BoxCorner_t *boxcorners = new BoxCorner_t[2 * nboxes];
    Vector3D<Precision> amin, amax;
    for (size_t i = 0; i < nboxes; ++i)
      ABBoxManager<Precision>::ComputeABBox(fVolumes[i], &boxcorners[2 * i], &boxcorners[2 * i + 1]);
    Boxes_t boxes = &boxcorners[0];
    fNavHelper    = HybridManager2::Instance().BuildStructure(boxes, nboxes);
    // Compute the lists of possibly overlapping neighbours
    fNeighbours  = new size_t *[nboxes];
    fNneighbours = new size_t[nboxes];
    memset(fNneighbours, 0, nboxes * sizeof(size_t));
    fBuffer = new size_t[nboxes * nboxes];
    for (size_t i = 0; i < nboxes; ++i) {
      fNeighbours[i] = fBuffer + i * nboxes;
    }
    size_t newsize = 0;
    for (size_t i = 0; i < nboxes - 1; ++i) {
      for (size_t j = i + 1; j < nboxes; ++j) {
        if (ABBoxOverlap(boxcorners[2 * i], boxcorners[2 * i + 1], boxcorners[2 * j], boxcorners[2 * j + 1])) {
          fNeighbours[i][fNneighbours[i]++] = j;
          fNeighbours[j][fNneighbours[j]++] = i;
          newsize += 2;
        }
      }
    }
    // Compacting buffer of neighbours
    size_t *buffer  = new size_t[newsize];
    size_t *nextloc = buffer;
    for (size_t i = 0; i < nboxes; ++i) {
      memcpy(nextloc, fNeighbours[i], fNneighbours[i] * sizeof(size_t));
      fNeighbours[i] = nextloc;
      nextloc += fNneighbours[i];
    }
    delete[] fBuffer;
    fBuffer = buffer;
  }

}; // End struct

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VECGEOM_VOLUMES_MULTIUNIONSTRUCT_H_ */
