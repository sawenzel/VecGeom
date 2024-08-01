/// \file BVHsurf.h
/// Rewritten BVH.h as header-only class templated on the precision type

#ifndef VECGEOM_SURFACES_BVH_H_
#define VECGEOM_SURFACES_BVH_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Cuda.h"
#include "VecGeom/navigation/NavStateIndex.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/surfaces/Model.h"
#include "VecGeom/surfaces/bvh/AABBsurf.h"

#include <vector>

namespace vgbrep {
namespace bvh {

template <typename Real_t>
struct SurfData;

/**
 * \class BVH
 * The BVH class is based on a complete binary tree stored contiguously in an array.
 *
 * For a given node id, 2*id + 1 gives the left child id, and 2*id + 2 gives the right child id.
 * For example, node 0 has its children at positions 1 and 2 in the vector, and for node 2, the
 * child nodes are at positions 5 and 6. The tree has 1 + 2 + 4 + ... + d nodes in total, where
 * d is the depth of the tree, or 1, 3, 7, ..., 2^d - 1 nodes in total. Visually, the ids for each
 * node look like shown below for a tree of depth 3:
 *
 *                                              0
 *                                             / \
 *                                            1   2
 *                                           / \ / \
 *                                          3  4 5  6
 *
 * with 2^3 - 1 = 7 nodes in total. For each node id, fNChild[id] gives the number of children of
 * the logical volume that belong to that node, and fOffset[id] gives the offset in the id map
 * fPrimId where the ids of the child volumes are stored, such that accessing fPrimId[fOffset[id]]
 * gives the first child id, then fPrimId[fOffset[id] + 1] gives the second child, up to fNChild[id]
 * children. The bounding boxes stored in fAABBs are in the original order, so they are accessed by
 * the original child number (i.e. the id stored in fPrimId, not by a node id of the tree itself).
 */

/**
 * @brief Bounding Volume Hierarchy class to represent an axis-aligned bounding volume hierarchy.
 * @details BVH instances can be associated with logical volumes to accelerate queries to their child volumes.
 */
template <typename Real_t>
class BVHsurf {
private:
  uint fRootId    = 0;               ///< Id of the root element this BVH was constructed for
  int fRootNChild = 0;               ///< Number of children of the root element
  int fDepth      = 0;               ///< Depth of the BVH
  int *fPrimId{nullptr};             ///< Child volume ids for each BVH node
  int *fOffset{nullptr};             ///< Offset in @c fPrimId for first child of each BVH node
  int *fNChild{nullptr};             ///< Number of children for each BVH node
  AABBsurf<Real_t> *fNodes{nullptr}; ///< AABBs of BVH nodes
  AABBsurf<Real_t> *fAABBs{nullptr}; ///< AABBs of children of the BVH root element

public:
  using Real_b = Real_t;
  /** Maximum depth. */
  static constexpr int BVH_MAX_DEPTH = 32;
  /** Default constructor */
  BVHsurf() {}
  /** Destructor. */
  ~BVHsurf() { Clear(); }

  void Clear()
  {
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
    delete[] fPrimId;
    fPrimId = nullptr;
    delete[] fOffset;
    fOffset = nullptr;
    delete[] fNChild;
    fNChild = nullptr;
    delete[] fNodes;
    fNodes = nullptr;
    delete[] fAABBs;
    fAABBs = nullptr;
#endif
  }
  uint GetRootId() const { return fRootId; }
  int GetRootNChild() const { return fRootNChild; };
  int GetDepth() const { return fDepth; };
  const int *GetPrimId() const { return fPrimId; };
  const int *GetOffset() const { return fOffset; };
  const int *GetNChild() const { return fNChild; };
  const AABBsurf<Real_t> *GetAABBs() const { return fAABBs; };
  const AABBsurf<Real_t> *GetNodes() const { return fNodes; };

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
  BVHsurf(int id, int nchild, int depth, int *dPrimId, AABBsurf<Real_t> *dAABBs, int *dOffset, int *dNChild,
          AABBsurf<Real_t> *dNodes)
      : fRootId(id), fRootNChild(nchild), fPrimId(dPrimId), fOffset(dOffset), fNChild(dNChild), fNodes(dNodes),
        fAABBs(dAABBs), fDepth(depth)
  {
  }

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
  void Set(int id, int nchild, int depth, int *dPrimId, AABBsurf<Real_t> *dAABBs, int *dOffset, int *dNChild,
           AABBsurf<Real_t> *dNodes)
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
  void SetPointers(int *dPrimId, int *dOffset, int *dNChild, AABBsurf<Real_t> *dAABBs, AABBsurf<Real_t> *dNodes)
  {
    fPrimId = dPrimId;
    fOffset = dOffset;
    fNChild = dNChild;
    fAABBs  = dAABBs;
    fNodes  = dNodes;
  }

  /** Print a summary of BVH contents */
  VECCORE_ATT_HOST_DEVICE
  void Print(bool verbose = false) const
  {
    printf("\nBVH(%u): addr: %p, depth: %d, nodes: %d, children: %d, name: %s\n", fRootId, this, fDepth,
           (2 << fDepth) - 1, fRootNChild, " ");
    if (verbose) {
      constexpr auto width = 4;
      int nChildToPad      = 1;
      for (int depth = fDepth; depth >= 0; --depth) {
        const auto begin = (1 << depth) - 1;
        const auto end   = (2 << depth) - 1;
        for (int node = begin; node < end; ++node) {
          if (nChildToPad > 1) printf("%*c", (nChildToPad - 1) * width / 2, ' ');
          printf("%3d ", fNChild[node]);
          if (nChildToPad > 1) printf("%*c", (nChildToPad - 1) * width / 2, ' ');
        }
        printf("\n");
        nChildToPad *= 2;
      }
    }
  }

  uint GetAllocatedSize() const
  {
    uint nodes = (2 << fDepth) - 1;
    uint size{0};
    // fAABBs
    size += fRootNChild * sizeof(AABBsurf<Real_t>);
    // fNodes
    size += nodes * sizeof(AABBsurf<Real_t>);
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
   * @tparam Real_i Input precision type
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
    Vector3D<Real_t> binvdir(static_cast<Real_t>(1.0 / vecgeom::NonZero(localdir[0])),
                             static_cast<Real_t>(1.0 / vecgeom::NonZero(localdir[1])),
                             static_cast<Real_t>(1.0 / vecgeom::NonZero(localdir[2])));
    Vector3D<Real_t> blocalpoint(static_cast<Real_t>(localpoint[0]), static_cast<Real_t>(localpoint[1]),
                                 static_cast<Real_t>(localpoint[2]));
    Vector3D<Real_t> blocaldir(static_cast<Real_t>(localdir[0]), static_cast<Real_t>(localdir[1]),
                               static_cast<Real_t>(localdir[2]));

    Real_t bstep = Real_t(step);
    do {
      const unsigned int id = *--ptr; /* pop next node id to be checked from the stack */

      // If the current distance is shorter than the distance to the node we can safely ignore it
      Real_t min{vecgeom::InfinityLength<Real_t>()}, max{-vecgeom::InfinityLength<Real_t>()};
      fNodes[id].ComputeIntersectionInvDir(blocalpoint, binvdir, min, max);
      if (min > max || max < Real_t{0} || min >= static_cast<Real_t>(bstep)) {
        continue;
      }

      if (fNChild[id] >= 0) {
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
              step = bstep = dist;
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
      if (!(min <= max && max >= static_cast<Real_t>(0.0) && min < static_cast<Real_t>(step))) {
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
            total_visited_children++;
            auto dist = Navigator::CandidateDistanceToIn(
                fRootId, prim, localpoint + static_cast<Real_i>(approach) * localdir, localdir, step);
            dist += static_cast<Real_i>(approach);
            /* If distance to current child is smaller than current step, update step and hitcandidate */
            if (dist < step &&
                !(dist <= vecgeom::kToleranceDist<Real_i> && Navigator::SkipItem(fRootId, prim, last_exited_id))) {
              step = dist, hitcandidate_index = prim;
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

  /**
   * Compute safety against children of the root element associated with the BVH.
   * @param[in] localpoint Point in the local coordinates of the root element.
   * @param[in] safety Maximum safety. Elements further than this are not checked.
   * @returns Minimum between safety to the closest child of root element and input @p safety.
   */
  /*
   * BVH::ComputeSafety is very similar to CheckDaughterIntersections regarding traversal of the tree, but
   * it computes only the safety instead of the intersection using a ray, so the logic is a bit simpler.
   */
  template <typename Navigator>
  VECCORE_ATT_HOST_DEVICE Real_t ComputeSafety(Vector3D<Real_t> localpoint, Real_t safety) const
  {
    unsigned int stack[BVH_MAX_DEPTH], *ptr = &stack[1];
    stack[0] = 0;

    do {
      const unsigned int id = *--ptr;

      if (fNChild[id] >= 0) {
        for (int i = 0; i < fNChild[id]; ++i) {
          const int prim = fPrimId[fOffset[id] + i];
          if (fAABBs[prim].Safety(localpoint) < safety) {

            // printf("// BVH Call, AABB safety: %lf\n", fAABBs[prim].Safety(localpoint));

            const Real_t dist = Navigator::CandidateSafetyToIn(fRootId, prim, localpoint);
            if (dist < safety) safety = dist;
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
   * @param[out] daughterlocalpoint Point in the local coordinates of the container element
   * @returns Whether @p localpoint falls within a child element of this BVH.
   */
  template <typename Navigator>
  VECCORE_ATT_HOST_DEVICE bool LevelLocate(long const exclude_item_id, Vector3D<Real_t> const &localpoint,
                                           long &container_id, Vector3D<Real_t> &daughterlocalpoint) const
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
                Navigator::CandidateContains(fRootId, prim, localpoint, daughterlocalpoint)) {
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
  VECCORE_ATT_HOST_DEVICE void ApproachNextDaughter(Vector3D<Real_t> localpoint, Vector3D<Real_t> localdir,
                                                    Real_t &step, long const last_exited_id) const
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
          /* Check AABB first, then the element itself if needed */
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

} // namespace bvh
} // namespace vgbrep

#endif
