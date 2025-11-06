/// \file BVH.h
/// \author Guilherme Amadio

#ifndef VECGEOM_BASE_BVH_H_
#define VECGEOM_BASE_BVH_H_

#include "VecGeom/base/AABB.h"
#include "VecGeom/base/Config.h"
#include "VecGeom/base/Cuda.h"
#include "VecGeom/navigation/NavStateIndex.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
// #include "VecGeom/surfaces/Model.h"

#include <vector>

// Forward-declare CPUsurfData
namespace vgbrep {
template <typename Real_t>
struct CPUsurfData;
}

namespace vecgeom {
namespace cuda {
template <typename Real_t>
class BVH;
}
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, BVH, typename);
inline namespace VECGEOM_IMPL_NAMESPACE {

class LogicalVolume;
class VPlacedVolume;

/**
 * @brief Bounding Volume Hierarchy class to represent an axis-aligned bounding volume hierarchy.
 * @details BVH instances can be associated with logical volumes to accelerate queries to their child volumes.
 */
template <typename Real_t>
class BVH {
private:
  uint fRootId    = 0;           ///< Id of the root element this BVH was constructed for
  int fRootNChild = 0;           ///< Number of children of the root element
  int fDepth      = 0;           ///< Depth of the BVH
  int *fPrimId{nullptr};         ///< Child volume ids for each BVH node
  int *fOffset{nullptr};         ///< Offset in @c fPrimId for first child of each BVH node
  int *fNChild{nullptr};         ///< Number of children for each BVH node
  AABB<Real_t> *fNodes{nullptr}; ///< AABBs of BVH nodes
  AABB<Real_t> *fAABBs{nullptr}; ///< AABBs of children of the BVH root element

public:
  // Default constructor
  BVH()
      : fRootId(0), fRootNChild(0), fPrimId(nullptr), fOffset(nullptr), fNChild(nullptr), fNodes(nullptr),
        fAABBs(nullptr), fDepth(0)
  {
  }

  uint GetRootId() const { return fRootId; }
  int GetRootNChild() const { return fRootNChild; };
  int GetDepth() const { return fDepth; };
  const int *GetPrimId() const { return fPrimId; };
  const int *GetOffset() const { return fOffset; };
  const int *GetNChild() const { return fNChild; };
  const AABB<Real_t> *GetAABBs() const { return fAABBs; };
  const AABB<Real_t> *GetNodes() const { return fNodes; };

  /** Maximum depth. */
  static constexpr int BVH_MAX_DEPTH = 32;
  /**
   * Constructor.
   * @param volume Pointer to logical volume for which the BVH will be created.
   * @param surfacesBVH Whether to build this BVH from AABBs created for solids or surfaces
   * @param depth Depth of the BVH binary tree. Defaults to zero, in which case
   * the actual depth will be chosen dynamically based on the number of child volumes.
   * When a fixed depth is chosen, it cannot be larger than @p BVH_MAX_DEPTH.
   */
  BVH(LogicalVolume const &volume, bool surfacesBVH = false, vgbrep::CPUsurfData<Precision> const *surfData = nullptr,
      int depth = 0);

  /** Destructor. */
  ~BVH() { Clear(); }
  void Clear();

  /**
   * Initializer used by BVHcreator. Takes as input pre-constructed BVH buffers.
   * @param id  Id of the logical volume
   * @param nchild Number of children of the volume
   * @param depth Depth of the BVH binary tree stored in the device buffers.
   * @param dPrimId Device buffer with child volume ids
   * @param dAABBs  Device buffer with AABBs of child volumes
   * @param dOffset Device buffer with offsets in @c dPrimId for first child of each BVH node
   * @param dNChild Device buffer with number of children for each BVH node
   * @param dNodes AABBs of BVH nodes
   */
  VECCORE_ATT_HOST_DEVICE
  void Set(int id, int nchild, int depth, int *dPrimId, vecgeom::AABB<Real_t> *dAABBs, int *dOffset, int *dNChild,
           vecgeom::AABB<Real_t> *dNodes)
  {
    fRootId     = id;
    fRootNChild = nchild;
    fDepth      = depth;
    SetPointers(dPrimId, dOffset, dNChild, dAABBs, dNodes);
  }

  /**
   * Setter for all member arrays. Used to update the pointers after a copy from host to device
   * @param dPrimId Device buffer with child volume ids
   * @param dAABBs  Device buffer with AABBs of child volumes
   * @param dOffset Device buffer with offsets in @c dPrimId for first child of each BVH node
   * @param dNChild Device buffer with number of children for each BVH node
   * @param dNodes AABBs of BVH nodes
   */
  VECCORE_ATT_HOST_DEVICE
  void SetPointers(int *dPrimId, int *dOffset, int *dNChild, AABB<Real_t> *dAABBs, AABB<Real_t> *dNodes)
  {
    fPrimId = dPrimId;
    fOffset = dOffset;
    fNChild = dNChild;
    fAABBs  = dAABBs;
    fNodes  = dNodes;
  }

#ifdef VECGEOM_ENABLE_CUDA
  /**
   * Constructor for GPU. Takes as input pre-constructed BVH buffers.
   * @param volume  Reference to logical volume on the device
   * @param depth Depth of the BVH binary tree stored in the device buffers.
   * @param dPrimId Device buffer with child volume ids
   * @param dAABBs  Device buffer with AABBs of child volumes
   * @param dOffset Device buffer with offsets in @c dPrimId for first child of each BVH node
   * @param dNChild Device buffer with number of children for each BVH node
   * @param dNodes AABBs of BVH nodes
   */
  VECCORE_ATT_DEVICE
  BVH(LogicalVolume const *volume, int depth, int *dPrimId, AABB<Real_t> *dAABBs, int *dOffset, int *NChild,
      AABB<Real_t> *dNodes);

  /**
   * Constructor for GPU. Takes as input pre-constructed BVH buffers.
   * @param id  Id of the logical volume
   * @param nchild Number of children of the volume
   * @param depth Depth of the BVH binary tree stored in the device buffers.
   * @param dPrimId Device buffer with child volume ids
   * @param dAABBs  Device buffer with AABBs of child volumes
   * @param dOffset Device buffer with offsets in @c dPrimId for first child of each BVH node
   * @param dNChild Device buffer with number of children for each BVH node
   * @param dNodes AABBs of BVH nodes
   */
  VECCORE_ATT_HOST_DEVICE
  BVH(int id, int nchild, int depth, int *dPrimId, vecgeom::AABB<Real_t> *dAABBs, int *dOffset, int *dNChild,
      vecgeom::AABB<Real_t> *dNodes)
      : fRootId(id), fRootNChild(nchild), fPrimId(dPrimId), fOffset(dOffset), fNChild(dNChild), fNodes(dNodes),
        fAABBs(dAABBs), fDepth(depth)
  {
  }

#endif

#ifdef VECGEOM_CUDA_INTERFACE
  /** Copy and construct an instance of this BVH on the device, at the device address @p addr. */
  DevicePtr<cuda::BVH<Real_t>> CopyToGpu(void *addr) const;
#endif

  // void CopyToGpu(BVH *dBVH) const;

  /** Print a summary of BVH contents */
  VECCORE_ATT_HOST_DEVICE
  void Print(bool verbose = false) const;

  uint GetAllocatedSize() const
  {
    uint nodes = (2 << fDepth) - 1;
    uint size{0};
    // fAABBs
    size += fRootNChild * sizeof(vecgeom::AABB<Real_t>);
    // fNodes
    size += nodes * sizeof(vecgeom::AABB<Real_t>);
    // fNChild
    size += nodes * sizeof(int);
    // fOffset
    size += nodes * sizeof(int);
    // fPrimId
    size += fRootNChild * sizeof(int);
    return size;
  }

  /**
   * Check ray defined by <tt>localpoint + t * localdir</tt> for intersections with children
   * of the root element of the BVH, and within a maximum distance of @p step
   * along the ray, while ignoring the @p last_exited_id volume.
   * @param[in] localpoint Point in the local coordinates of the BVH root.
   * @param[in] localdir Direction in the local coordinates of the BVH root.
   * @param[in,out] step Maximum step distance for which intersections should be considered.
   * @param[in] last_exited_id Last exited element. This element is ignored when reporting intersections.
   * @param[out] hitcandidate_index Index of element for which closest intersection was found. -1 if no intersection
   * is found within the current step distance.
   */
  /*
   * BVH::ComputeDaughterIntersections() computes the intersection of a ray against all children of
   * the logical volume. A stack is kept of the node ids that need to be checked. It needs to be at
   * most as deep as the binary tree itself because we always first pop the current node, and then
   * add at most the two children. For example, for depth two, we pop the root node, then at most we
   * add both of its leaves onto the stack to be checked. We initialize ptr with &stack[1] such that
   * when we pop the first time as we enter the loop, the position we read from is the first position
   * of the stack, which contains the id 0 for the root node. When we pop the stack such that ptr
   * points before &stack[0], it means we've checked all we needed and the loop can be terminated.
   * In order to determine if a node of the tree is internal or not, we check if the node id of its
   * left child is past the end of the array (in which case we know we are at the maximum depth), or
   * if the sum of children in both leaves is the same as in the current node, as for leaf nodes, the
   * sum of children in the left+right child nodes will be less than for the current node.
   */
  template <typename Navigator, typename Real_i>
  VECCORE_ATT_HOST_DEVICE void CheckDaughterIntersections(const Vector3D<Real_i> &localpoint,
                                                          const Vector3D<Real_i> &localdir, Real_i &step,
                                                          long const last_exited_id, long &hitcandidate_index) const
  {
    unsigned int stack[BVH_MAX_DEPTH], *ptr = &stack[1];
    stack[0] = 0;

    /* Calculate and reuse inverse direction to save on divisions */
    Vector3D<Real_t> binvdir(static_cast<Real_t>(1.0) / vecgeom::NonZero(localdir[0]),
                             static_cast<Real_t>(1.0) / vecgeom::NonZero(localdir[1]),
                             static_cast<Real_t>(1.0) / vecgeom::NonZero(localdir[2]));
    Vector3D<Real_t> blocalpoint(static_cast<Real_t>(localpoint[0]), static_cast<Real_t>(localpoint[1]),
                                 static_cast<Real_t>(localpoint[2]));
    Vector3D<Real_t> blocaldir(static_cast<Real_t>(localdir[0]), static_cast<Real_t>(localdir[1]),
                               static_cast<Real_t>(localdir[2]));
    Real_t bstep = static_cast<Real_t>(step);

    do {
      const unsigned int id = *--ptr; /* pop next node id to be checked from the stack */

      // If the current distance is shorter than the distance to the node we can safely ignore it
      Real_t min{vecgeom::InfinityLength<Real_t>()}, max{-vecgeom::InfinityLength<Real_t>()};
      fNodes[id].ComputeIntersectionInvDir(blocalpoint, binvdir, min, max);
      if (min > max || max < Real_t{0.} || min >= step) {
        continue;
      }

      if (fNChild[id] >= 0) {

        /* For leaf nodes, loop over children */
        for (int i = 0; i < fNChild[id]; ++i) {
          const int prim = fPrimId[fOffset[id] + i];
          if (last_exited_id >= 0 && Navigator::SkipItem(fRootId, prim, last_exited_id)) continue;
          /* Check AABB first, then the element itself if needed */
          Real_t approach;
          if (fAABBs[prim].IntersectInvDirApproach(blocalpoint, binvdir, bstep, approach)) {
            auto dist = Navigator::CandidateDistanceToIn(
                fRootId, prim, localpoint + static_cast<Real_i>(approach) * localdir, localdir, step);
            // Only compensate with the approach distance if the distance is positive (i.e. not a wrong-side error)
            dist += (dist > 0.) * static_cast<Real_i>(approach);
            /* If distance to current child is smaller than current step, update step and hitcandidate */
            if (dist < step) {
              dist               = vecCore::Max(dist, 0.);
              step               = dist;
              bstep              = static_cast<Real_t>(dist);
              hitcandidate_index = prim;
            }
          }
        }
      } else {
        const unsigned int childL = 2 * id + 1;
        const unsigned int childR = 2 * id + 2;

        /* For internal nodes, check AABBs to know if we need to traverse left and right children */
        Real_t tminL = vecgeom::InfinityLength<Real_t>(), tmaxL = -vecgeom::InfinityLength<Real_t>(),
               tminR = vecgeom::InfinityLength<Real_t>(), tmaxR = -vecgeom::InfinityLength<Real_t>();

        fNodes[childL].ComputeIntersectionInvDir(blocalpoint, binvdir, tminL, tmaxL);
        fNodes[childR].ComputeIntersectionInvDir(blocalpoint, binvdir, tminR, tmaxR);

        const bool traverseL = tminL <= tmaxL && tmaxL >= static_cast<Real_t>(0.0) && tminL < bstep;
        const bool traverseR = tminR <= tmaxR && tmaxR >= static_cast<Real_t>(0.0) && tminR < bstep;

        /*
         * If both left and right nodes need to be checked, check closest one first.
         * This ensures step gets short as fast as possible so we can skip more nodes without checking.
         */
        if (tminR < tminL) {
          if (traverseL) *ptr++ = childL;
          if (traverseR) *ptr++ = childR;
        } else {
          if (traverseR) *ptr++ = childR;
          if (traverseL) *ptr++ = childL;
        }
      }
    } while (ptr > stack);
  }

  /**
   * Compute safety against children of the root element associated with the BVH.
   * @param[in] localpoint Point in the local coordinates of the root element.
   * @param[in] safety Maximum safety. Elements further than this are not checked.
   * @param[in] limit Do not call the primitive safety if farther than this value
   * @returns Minimum between safety to the closest child of root element and input @p safety.
   */
  /*
   * BVH::ComputeSafety is very similar to CheckDaughterIntersections regarding traversal of the tree, but
   * it computes only the safety instead of the intersection using a ray, so the logic is a bit simpler.
   */
  template <typename Navigator>
  VECCORE_ATT_HOST_DEVICE Precision ComputeSafety(Vector3D<Precision> localpoint, Precision safety,
                                                  Precision limit = InfinityLength<Precision>()) const
  {
    unsigned int stack[BVH_MAX_DEPTH], *ptr = &stack[1];
    stack[0] = 0;

    do {
      const unsigned int id = *--ptr;

      // We can safely ignore nodes that are farther than the current safety
      if (fNodes[id].Safety(localpoint) > safety) continue;

      if (fNChild[id] >= 0) {
        for (int i = 0; i < fNChild[id]; ++i) {
          const int prim = fPrimId[fOffset[id] + i];
          if (fAABBs[prim].Safety(localpoint) < safety) {
            auto safety_leaf = fAABBs[prim].Safety(localpoint);
            // If the distance to the current node is larger than the safety we can ignore it
            if (safety_leaf >= safety) continue;
            // Don't check daughters if the safety is larger than the accuracy limit
            if (safety_leaf > limit) {
              safety = safety_leaf;
              continue;
            }
            const Precision dist = Navigator::CandidateSafetyToIn(fRootId, prim, localpoint);
            if (dist > -vecgeom::kToleranceDist<Precision>) {
              if (dist < safety) safety = dist;
            }
          }
        }
      } else {
        const unsigned int childL = 2 * id + 1;
        const unsigned int childR = 2 * id + 2;

        const Real_t safetyL = fNodes[childL].Safety(localpoint);
        const Real_t safetyR = fNodes[childR].Safety(localpoint);

        const bool traverseL = safetyL < safety;
        const bool traverseR = safetyR < safety;

        if (safetyR < safetyL) {
          if (traverseR) *ptr++ = childR;
          if (traverseL) *ptr++ = childL;
        } else {
          if (traverseL) *ptr++ = childL;
          if (traverseR) *ptr++ = childR;
        }
      }
    } while (ptr > stack);

    return safety;
  }

  /**
   * Find child element inside which the given point @p localpoint is located.
   * @param[in] exclude_item_id Element that should be ignored.
   * @param[in] localpoint Point in the local coordinates of the BVH root element.
   * @param[out] container_id Id of the element in which @p localpoint is contained
   * @param[out] path Navigation state of the container element
   * @returns Whether @p localpoint falls within a child element of this BVH.
   */
  template <typename Navigator>
  VECCORE_ATT_HOST_DEVICE bool LevelLocate(int const exclude_item_id, Vector3D<Real_t> const &localpoint,
                                           int &container_id, vecgeom::NavigationState &path) const
  {
    unsigned int stack[BVH_MAX_DEPTH], *ptr = &stack[1];
    stack[0] = 0;

    do {
      const unsigned int id = *--ptr;

      if (fNChild[id] >= 0) {
        for (int i = 0; i < fNChild[id]; ++i) {
          const int prim = fPrimId[fOffset[id] + i];
          if (fAABBs[prim].Contains(localpoint)) {
            if (!Navigator::SkipItem(fRootId, prim, exclude_item_id) &&
                Navigator::CandidateContains(fRootId, prim, localpoint, path)) {
              container_id = Navigator::ItemId(fRootId, prim);
              return true;
            }
          }
        }
      } else {
        const unsigned int childL = 2 * id + 1;
        if (fNodes[childL].Contains(localpoint)) *ptr++ = childL;

        const unsigned int childR = 2 * id + 2;
        if (fNodes[childR].Contains(localpoint)) *ptr++ = childR;
      }
    } while (ptr > stack);

    return false;
  }

  /**
   * Find child element inside which the given point @p localpoint is located.
   * @param[in] exclude_item_id Element that should be ignored.
   * @param[in] localpoint Point in the local coordinates of the BVH root element.
   * @param[out] container_id Id of the element in which @p localpoint is contained
   * @param[out] daughterlocalpoint Point in the local coordinates of the container element
   * @returns Whether @p localpoint falls within a child element of this BVH.
   */
  template <typename Navigator>
  VECCORE_ATT_HOST_DEVICE vecgeom::Inside_t LevelInside(long const exclude_item_id,
                                                        Vector3D<Precision> const &localpoint, long &container_id,
                                                        Vector3D<Precision> &daughterlocalpoint) const
  {
    unsigned int stack[BVH_MAX_DEPTH], *ptr = &stack[1];
    stack[0] = 0;

    do {
      const unsigned int id = *--ptr;

      if (fNChild[id] >= 0) {
        for (int i = 0; i < fNChild[id]; ++i) {
          const int prim = fPrimId[fOffset[id] + i];
          if (fAABBs[prim].Contains(localpoint)) {
            if (Navigator::SkipItem(fRootId, prim, exclude_item_id)) continue;
            auto inside = Navigator::CandidateInside(fRootId, prim, localpoint, daughterlocalpoint);
            if (inside != kOutside) {
              container_id = Navigator::ItemId(fRootId, prim);
              return inside;
            }
          }
        }
      } else {
        const unsigned int childL = 2 * id + 1;
        if (fNodes[childL].Contains(localpoint)) *ptr++ = childL;

        const unsigned int childR = 2 * id + 2;
        if (fNodes[childR].Contains(localpoint)) *ptr++ = childR;
      }
    } while (ptr > stack);

    return kOutside;
  }

  /**
   * Check ray defined by <tt>localpoint + t * localdir</tt> for intersections with bounding
   * boxes of children of the root element of the BVH, and within a maximum
   * distance of @p step along the ray. Returns the distance to the first crossed box.
   * @param[in] localpoint Point in the local coordinates of the root element.
   * @param[in] localdir Direction in the local coordinates of the root element.
   * @param[in,out] step Maximum step distance for which intersections should be considered.
   * @param[in] last_exited_id Last exited element. This element is ignored when reporting intersections.
   */
  /*
   * BVH::ApproachNextDaughter is very similar to CheckDaughterIntersections but computes the first
   * hit daughter bounding box instead of the next hit shape. This lighter computation is used to
   * first approach the next hit solid before computing the actual distance, in the attempt to
   * reduce the numerical rounding error due to propagation to boundary.
   */
  template <typename Navigator>
  VECCORE_ATT_HOST_DEVICE void ApproachNextDaughter(Vector3D<Precision> localpoint, Vector3D<Precision> localdir,
                                                    Precision &step, long const last_exited_id) const
  {
    unsigned int stack[BVH_MAX_DEPTH] = {0}, *ptr = &stack[1];

    /* Calculate and reuse inverse direction to save on divisions */
    Vector3D<Real_t> invlocaldir(static_cast<Real_t>(1.0 / NonZero(localdir[0])),
                                 static_cast<Real_t>(1.0 / NonZero(localdir[1])),
                                 static_cast<Real_t>(1.0 / NonZero(localdir[2])));

    do {
      unsigned int id = *--ptr; /* pop next node id to be checked from the stack */

      if (fNChild[id] >= 0) {
        /* For leaf nodes, loop over children */
        for (int i = 0; i < fNChild[id]; ++i) {
          int prim = fPrimId[fOffset[id] + i];
          /* Check vecgeom::AABB first, then the element itself if needed */
          if (fAABBs[prim].IntersectInvDir(localpoint, invlocaldir, step)) {
            const auto dist = Navigator::CandidateApproachSolid(fRootId, prim, localpoint, localdir);
            /* If distance to current child is smaller than current step, update step and hitcandidate */
            if (dist < step && !(dist <= 0.0 && Navigator::SkipItem(fRootId, prim, last_exited_id))) step = dist;
          }
        }
      } else {
        unsigned int childL = 2 * id + 1;
        unsigned int childR = 2 * id + 2;

        /* For internal nodes, check AABBs to know if we need to traverse left and right children */
        Real_t tminL = vecgeom::InfinityLength<Real_t>(), tmaxL = -vecgeom::InfinityLength<Real_t>(),
               tminR = vecgeom::InfinityLength<Real_t>(), tmaxR = -vecgeom::InfinityLength<Real_t>();

        fNodes[childL].ComputeIntersectionInvDir(localpoint, invlocaldir, tminL, tmaxL);
        fNodes[childR].ComputeIntersectionInvDir(localpoint, invlocaldir, tminR, tmaxR);

        bool traverseL = tminL <= tmaxL && tmaxL >= 0.0 && tminL < step;
        bool traverseR = tminR <= tmaxR && tmaxR >= 0.0 && tminR < step;

        /*
         * If both left and right nodes need to be checked, check closest one first.
         * This ensures step gets short as fast as possible so we can skip more nodes without checking.
         */
        if (tminR < tminL) {
          if (traverseR) *ptr++ = childR;
          if (traverseL) *ptr++ = childL;
        } else {
          if (traverseL) *ptr++ = childL;
          if (traverseR) *ptr++ = childR;
        }
      }
    } while (ptr > stack);
  }

  /**
   * Check ray defined by <tt>localpoint + t * localdir</tt> for intersections with children
   * of the root element of the BVH, and within a maximum distance of @p step
   * along the ray, while ignoring the @p last_exited_id volume.
   * @param[in] localpoint Point in the local coordinates of the BVH root.
   * @param[in] localdir Direction in the local coordinates of the BVH root.
   * @param[in,out] step Maximum step distance for which intersections should be considered.
   * @param[in] last_exited_id Last exited element. This element is ignored when reporting intersections.
   * @param[out] hitcandidate_index Index of element for which closest intersection was found. -1 if no intersection
   * is found within the current step distance.
   */
  /*
   * This function is meant to be used for benchmarking of the BVH, and not for actual navigation. It gathers stats
   * on the traversal of the BVH tree
   */
  template <typename Navigator, typename Real_i>
  VECCORE_ATT_HOST_DEVICE void CheckDaughterIntersectionsBenchmark(const Vector3D<Real_i> &localpoint,
                                                                   const Vector3D<Real_i> &localdir, Real_i &step,
                                                                   long const last_exited_id, long &hitcandidate_index,
                                                                   long &total_visited_children,
                                                                   long &total_visited_leaves, long &total_cut_nodes,
                                                                   long &total_stacked_nodes) const
  {
    total_visited_children = 0;
    total_visited_leaves   = 0;
    total_cut_nodes        = 0; // Number of nodes put on the stack but skipped when visited due to being too far
    total_stacked_nodes    = 0; // Number of nodes put on the stack to visit

    unsigned int stack[BVH_MAX_DEPTH], *ptr = &stack[1];
    stack[0] = 0;

    /* Calculate and reuse inverse direction to save on divisions */
    Vector3D<Real_t> binvdir(static_cast<Real_t>(1.0) / vecgeom::NonZero(localdir[0]),
                             static_cast<Real_t>(1.0) / vecgeom::NonZero(localdir[1]),
                             static_cast<Real_t>(1.0) / vecgeom::NonZero(localdir[2]));
    Vector3D<Real_t> blocalpoint(static_cast<Real_t>(localpoint[0]), static_cast<Real_t>(localpoint[1]),
                                 static_cast<Real_t>(localpoint[2]));
    Vector3D<Real_t> blocaldir(static_cast<Real_t>(localdir[0]), static_cast<Real_t>(localdir[1]),
                               static_cast<Real_t>(localdir[2]));
    Real_t bstep = static_cast<Real_t>(step);

    do {
      const unsigned int id = *--ptr; /* pop next node id to be checked from the stack */

      // If the current distance is shorter than the distance to the node we can safely ignore it
      Real_t min{vecgeom::InfinityLength<Real_t>()}, max{-vecgeom::InfinityLength<Real_t>()};
      fNodes[id].ComputeIntersectionInvDir(blocalpoint, binvdir, min, max);
      if (!(min <= max && max >= 0.0 && min < step)) {
        total_cut_nodes++;
        continue;
      }

      if (fNChild[id] >= 0) {
        total_visited_leaves++;
        /* For leaf nodes, loop over children */
        for (int i = 0; i < fNChild[id]; ++i) {
          const int prim = fPrimId[fOffset[id] + i];
          /* Check AABB first, then the element itself if needed */
          Real_t approach;
          if (fAABBs[prim].IntersectInvDirApproach(blocalpoint, binvdir, bstep, approach)) {
            auto dist = Navigator::CandidateDistanceToIn(
                fRootId, prim, localpoint + static_cast<Real_i>(approach) * localdir, localdir, step);
            dist += static_cast<Real_i>(approach);
            /* If distance to current child is smaller than current step, update step and hitcandidate */
            if (dist < step &&
                !(dist <= vecgeom::kToleranceDist<Real_i> && Navigator::SkipItem(fRootId, prim, last_exited_id))) {
              step               = dist;
              bstep              = static_cast<Real_t>(dist);
              hitcandidate_index = prim;
            }
          }
        }
      } else {
        const unsigned int childL = 2 * id + 1;
        const unsigned int childR = 2 * id + 2;

        /* For internal nodes, check AABBs to know if we need to traverse left and right children */
        Real_t tminL = vecgeom::InfinityLength<Real_t>(), tmaxL = -vecgeom::InfinityLength<Real_t>(),
               tminR = vecgeom::InfinityLength<Real_t>(), tmaxR = -vecgeom::InfinityLength<Real_t>();

        fNodes[childL].ComputeIntersectionInvDir(blocalpoint, binvdir, tminL, tmaxL);
        fNodes[childR].ComputeIntersectionInvDir(blocalpoint, binvdir, tminR, tmaxR);

        const bool traverseL = tminL <= tmaxL && tmaxL >= 0.0 && tminL < step;
        const bool traverseR = tminR <= tmaxR && tmaxR >= 0.0 && tminR < step;

        /*
         * If both left and right nodes need to be checked, check closest one first.
         * This ensures step gets short as fast as possible so we can skip more nodes without checking.
         */
        if (tminR < tminL) {
          if (traverseR) *ptr++ = childR;
          if (traverseL) *ptr++ = childL;
        } else {
          if (traverseL) *ptr++ = childL;
          if (traverseR) *ptr++ = childR;
        }

        if (traverseR) total_stacked_nodes++;
        if (traverseL) total_stacked_nodes++;
      }
    } while (ptr > stack);
  }

private:
  enum class ConstructionAlgorithm : unsigned int;
  /**
   * Compute internal nodes of the BVH recursively.
   * @param[in] id Node id of node to be computed.
   * @param[in] first Iterator pointing to the position of this node's first volume in @c fPrimId.
   * @param[in] last Iterator pointing to the position of this node's last volume in @c fPrimId.
   * @param[in] nodes Number of nodes for this BVH.
   * @param[in] constructionAlgorithm Index of the splitting function to use.
   *
   * @remark This function computes the bounding box of a node, then chooses a split plane and reorders
   * the elements of @c fPrimId within the first,last range such that volumes for its left child all come
   * before volumes for its right child, then launches itself to compute bounding boxes of each child.
   * Recursion stops when all children lie on one side of the splitting plane, or when the current node
   * contains only a single child volume.
   */
  void ComputeNodes(unsigned int id, int *first, int *last, unsigned int nodes, ConstructionAlgorithm);
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
