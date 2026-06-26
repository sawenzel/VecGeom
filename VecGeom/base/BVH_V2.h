/// \file BVH_V2.h
/// \brief Minimal, explicit-index BVH used as a drop-in replacement for the
///        facet BVH consumed by TessellatedImplementation.h.
///
/// This is a deliberately small rework of @c BVH (see base/BVH.h): instead of
/// a fixed-depth, complete binary tree stored as an implicit heap (children
/// of node `id` at `2*id+1`/`2*id+2`), nodes carry an explicit child index.
/// Children of an inner node are always allocated as a contiguous pair, so
/// traversal is still just `first`/`first+1`, but the tree itself can be
/// unbalanced and grow only as deep as the data requires. This is the layout
/// needed to eventually host high-quality builders (SAH sweep, reinsertion
/// optimization) that do not produce complete trees.
///
/// Scope is intentionally restricted to what TessellatedImplementation.h and generic
/// AABB-set queries (see test/core/TestBVH.cpp) use: construction from per-primitive
/// AABBs, ray intersection with a user-supplied leaf hook, and point-safety queries.
/// Everything else that @c BVH supports (GPU residency, navigation-specific traversals,
/// multiple splitting heuristics, ...) is left for later milestones.

#ifndef VECGEOM_BASE_BVH_V2_H_
#define VECGEOM_BASE_BVH_V2_H_

#include "VecGeom/base/AABB.h"
#include "VecGeom/base/Assert.h"
#include "VecGeom/base/BVH.h" // reuse BVHIntersectContext<Real_t> / BVHPointQueryContext<Real_t>
#include "VecGeom/base/Config.h"
#include "VecGeom/base/Cuda.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Minimal BVH over a set of primitive AABBs, using explicit child indices.
 * @details Built once on the host from per-primitive AABBs (e.g. facet boxes of a
 * tessellated solid). Per-primitive AABBs are retained (indexed by primitive id) so
 * that @c Intersect's default @c check_leaf_bb=true can report exact per-primitive
 * hits; callers that already do their own exact geometric test in the leaf hook (e.g.
 * a facet ray intersection) may pass @c check_leaf_bb=false to skip that box test.
 */
template <typename Real_t>
class BVH_V2 {
public:
  /** Maximum depth of the tree. Bounds the fixed traversal stack. */
  static constexpr int BVH_MAX_DEPTH = 32;

  /**
   * A single BVH node.
   * Leaf node (@c count > 0): @c first is the index into @c fPrimId of this
   * leaf's first primitive; it holds @c count primitives.
   * Inner node (@c count == 0): its two children are always allocated as a
   * contiguous pair, at node indices @c first and @c first + 1.
   */
  struct Node {
    /// Bounds laid out as [min_x, max_x, min_y, max_y, min_z, max_z] (matches madmann91 bvh::v2's
    /// Node layout) so IntersectOctant can pick the near/far corner per axis with a plain indexed
    /// load (bounds[2*axis + octant_bit]) instead of a runtime branch/select on separate min/max
    /// vectors -- see IntersectOctant for why that distinction matters.
    Real_t bounds[6];
    int first{0};
    int count{0};

    VECCORE_ATT_HOST_DEVICE
    bool IsLeaf() const { return count > 0; }

    VECCORE_ATT_HOST_DEVICE
    AABB<Real_t> Bounds() const
    {
      return AABB<Real_t>(Vector3D<Real_t>(bounds[0], bounds[2], bounds[4]),
                          Vector3D<Real_t>(bounds[1], bounds[3], bounds[5]));
    }

    VECCORE_ATT_HOST
    void SetBounds(const AABB<Real_t> &b)
    {
      const auto lo = b.Min();
      const auto hi = b.Max();
      bounds[0] = lo[0];
      bounds[1] = hi[0];
      bounds[2] = lo[1];
      bounds[3] = hi[1];
      bounds[4] = lo[2];
      bounds[5] = hi[2];
    }
  };

  /** No-op functor used as default for the optional traversal hooks. */
  struct IgnoreArgs {
    template <typename... Args>
    VECCORE_ATT_HOST_DEVICE void operator()(Args &&...) const
    {
    }
  };

  /**
   * Selects the build algorithm used by the constructor.
   * @c Original reproduces the split logic ported from BVH.cpp (source/BVH.cpp::surfaceAreaHeuristic):
   * always splits down to single-primitive leaves (or until @c BVH_MAX_DEPTH), using a sweep over the
   * surface-area heuristic with an extra balance term. Kept only as a fixed point for A/B comparison.
   * @c SweepSAH is the recommended policy: a proper cost-based sweep SAH (no balance-term hack), with
   * leaves capped at @c kMaxLeafSize primitives and a leaf-vs-split decision based on actual SAH cost.
   */
  enum class BuildPolicy { Original, SweepSAH };

  /**
   * SAH cost model used by @c BuildPolicy::SweepSAH (Wald-style constants: c_trav + c_isect * ...).
   * @c kCostIntersect is set higher than the textbook 1:1 ratio because the per-facet leaf test here
   * is not a bare ray-triangle check: it is TessellatedImplementation's Tile::Distance (a Moller-Trumbore
   * variant with extra epsilon/tolerance handling) plus parity/closure bookkeeping, which profiling
   * (facet-intersection counters vs. ROOT's bvh::v2-backed TGeoTessellated) showed dominates wall-clock
   * cost far more than a single AABB overlap test does. Biasing the cost model this way makes the
   * builder prefer more/smaller leaves -- trading additional (cheap) box tests for fewer (expensive)
   * facet tests. Treat this as a starting point for empirical tuning, not a derived constant.
   */
  static constexpr Real_t kCostTraversal = Real_t(1.0);
  static constexpr Real_t kCostIntersect = Real_t(3.0);
  /**
   * Leaves built by @c BuildPolicy::SweepSAH never exceed this many primitives (barring full degeneracy).
   * This is only a backstop for splits the cost model itself declines to take (e.g. coincident/duplicate
   * boxes); with @c kCostIntersect biased above, the SAH sweep should already settle on leaves well below
   * this cap in practice. Kept small for the same reason @c kCostIntersect was raised: bounding how many
   * facet tests a single leaf visit can ever cost.
   */
  static constexpr int kMaxLeafSize = 4;

  BVH_V2() = default;

  /**
   * Constructor.
   * @param rootId Id of the volume/solid this BVH was built for (opaque to this class).
   * @param ptrAABB Array of @p nChild (min,max) corner pairs, one per primitive.
   * @param nChild Number of primitives.
   * @param depth Fixed tree depth. Only used by @c BuildPolicy::Original. Defaults to 0, in which case a
   * depth is chosen dynamically based on @p nChild, capped at @c BVH_MAX_DEPTH.
   * @param policy Build algorithm; see @c BuildPolicy.
   */
  VECCORE_ATT_HOST
  BVH_V2(int rootId, Vector3D<Precision> *ptrAABB, int nChild, int depth = 0,
         BuildPolicy policy = BuildPolicy::SweepSAH)
      : fRootId(rootId), fNPrim(nChild)
  {
    VECGEOM_VALIDATE(nChild > 0, << "Cannot construct BVH_V2 for a volume with no primitives!");

    fPrimId = new int[nChild];
    std::iota(fPrimId, fPrimId + nChild, 0);

    // Per-primitive AABBs, indexed by primitive id (not by tree position) -- same layout and role as
    // BVH::fAABBs in base/BVH.h. Used both to build the node bounds and, by default, to let Intersect()
    // test each primitive's own box (check_leaf_bb=true) rather than just the merged leaf box. Corners
    // are rounded outward (min down, max up) when narrowing from Precision to Real_t, so a float BVH
    // built over double-precision geometry is always conservative and never culls a real intersection.
    fAABBs = new AABB<Real_t>[nChild];
    for (int i = 0; i < nChild; ++i) {
      Vector3D<Real_t> lo(RoundOutward<true>(ptrAABB[2 * i].x()), RoundOutward<true>(ptrAABB[2 * i].y()),
                          RoundOutward<true>(ptrAABB[2 * i].z()));
      Vector3D<Real_t> hi(RoundOutward<false>(ptrAABB[2 * i + 1].x()), RoundOutward<false>(ptrAABB[2 * i + 1].y()),
                          RoundOutward<false>(ptrAABB[2 * i + 1].z()));
      fAABBs[i] = AABB<Real_t>(lo, hi);
    }

    fMaxDepth = std::min(depth ? depth : BVH_MAX_DEPTH, BVH_MAX_DEPTH);

    // A binary tree with at most nChild leaves has at most 2 * nChild - 1 nodes in total.
    fNodes  = new Node[std::max(1, 2 * nChild - 1)];
    fNNodes = 1;

    if (policy == BuildPolicy::Original) {
      BuildNodeOriginal(0, 0, fAABBs, fPrimId, fPrimId + nChild);
    } else {
      BuildNodeSweepSAH(0, 0, fAABBs, fPrimId, fPrimId + nChild);
    }
  }

  ~BVH_V2() { Clear(); }

  VECCORE_ATT_HOST
  void Clear()
  {
    delete[] fNodes;
    fNodes = nullptr;
    delete[] fPrimId;
    fPrimId = nullptr;
    delete[] fAABBs;
    fAABBs = nullptr;
  }

  int GetRootId() const { return fRootId; }
  int GetNPrimitives() const { return fNPrim; }
  int GetNNodes() const { return fNNodes; }
  int GetMaxDepth() const { return fMaxDepth; }
  const Node *GetNodes() const { return fNodes; }
  const int *GetPrimId() const { return fPrimId; }

  /*
   * Intersect() walks the tree with a ray, calling the user-provided leaf hook for every
   * primitive in every leaf node whose bounding box is crossed within the current step.
   * Traversal mirrors BVH::Intersect() in base/BVH.h: nearest-child-first ordering, and
   * the leaf hook may shrink ctx.step_max to prune the remaining search.
   * When check_leaf_bb is true (the default, matching BVH::Intersect()), each primitive's own AABB
   * is tested before the hook is called, so the hook only fires for primitives whose own box is
   * actually crossed -- not just primitives that happen to share a leaf with one that is. Pass
   * check_leaf_bb=false only when the hook already performs its own exact geometric test (e.g. a
   * facet ray intersection), to skip the redundant per-primitive box check.
   */
  template <bool check_leaf_bb = true, typename Real_i, typename leaf_function, typename inner_function = IgnoreArgs,
            typename leaf_count_func = IgnoreArgs>
  VECCORE_ATT_HOST_DEVICE void Intersect(const Vector3D<Real_i> &localpoint, const Vector3D<Real_i> &localdir,
                                         Real_i step, leaf_function &&intersect_hook, inner_function &&inner = {},
                                         leaf_count_func &&count = {}) const
  {
    unsigned int stack[BVH_MAX_DEPTH];
    int sp = 0;

    Vector3D<Real_t> binvdir(static_cast<Real_t>(1.0) / vecgeom::NonZero(localdir[0]),
                             static_cast<Real_t>(1.0) / vecgeom::NonZero(localdir[1]),
                             static_cast<Real_t>(1.0) / vecgeom::NonZero(localdir[2]));
    Vector3D<Real_t> blocalpoint(static_cast<Real_t>(localpoint[0]), static_cast<Real_t>(localpoint[1]),
                                 static_cast<Real_t>(localpoint[2]));
    Real_t bstep = static_cast<Real_t>(step);

    // 3 sign bits of the ray direction, fixed for the whole traversal: lets every node's near/far
    // corner be picked directly (see IntersectOctant) instead of re-derived per node via min/max.
    const unsigned int octant = (binvdir[0] < Real_t(0.) ? 1u : 0u) | (binvdir[1] < Real_t(0.) ? 2u : 0u) |
                                (binvdir[2] < Real_t(0.) ? 4u : 0u);

    BVHIntersectContext<Real_t> hitcontext{-1, 0, Real_t(0.), bstep};

    Real_t tminRoot;
    if (!IntersectOctant(fNodes[0].bounds, blocalpoint, binvdir, octant, bstep, tminRoot)) {
      return;
    }

    unsigned int top = 0;
    for (;;) {
      while (!fNodes[top].IsLeaf()) {
        const unsigned int childL = fNodes[top].first;
        const unsigned int childR = childL + 1;
        inner();

        Real_t tminL, tminR;
        const bool hitL = IntersectOctant(fNodes[childL].bounds, blocalpoint, binvdir, octant, bstep, tminL);
        const bool hitR = IntersectOctant(fNodes[childR].bounds, blocalpoint, binvdir, octant, bstep, tminR);

        if (hitL) {
          unsigned int near = childL;
          if (hitR) {
            unsigned int far = childR;
            if (tminR < tminL) {
              near = childR;
              far  = childL;
            }
            stack[sp++] = far;
          }
          top = near;
        } else if (hitR) {
          top = childR;
        } else {
          goto pop;
        }
      }

      hitcontext.nLeafPrims = fNodes[top].count;
      count();
      for (int i = 0; i < fNodes[top].count; ++i) {
        const int prim = fPrimId[fNodes[top].first + i];
        Real_t approach{Real_t(0.)};
        if ((check_leaf_bb && fAABBs[prim].IntersectInvDirApproach(blocalpoint, binvdir, bstep, approach)) ||
            !check_leaf_bb) {
          hitcontext.primID    = prim;
          hitcontext.step_max  = bstep;
          hitcontext.tnear     = approach;
          const auto stop_here = intersect_hook(hitcontext);
          if (stop_here) {
            break;
          }
          bstep = hitcontext.step_max;
        }
      }

    pop:
      if (sp == 0) return;
      top = stack[--sp];
    }
  }

  /*
   * PointQuery() mirrors BVH::PointQuery() in base/BVH.h: it returns a squared safety
   * estimate against the primitives reachable from the root, calling the user-provided
   * leaf hook for every primitive in every visited leaf, which may shrink ctx.safetySqr.
   */
  template <bool return_top_estimate, typename leaf_function>
  VECCORE_ATT_HOST_DEVICE Precision PointQuery(Vector3D<Precision> localpoint, leaf_function &&primitive_hook,
                                               Precision limit_sq = InfinityLength<Precision>()) const
  {
    struct StackItem {
      unsigned int node;
      Real_t safetySq;
    };
    StackItem stack[BVH_MAX_DEPTH], *ptr = &stack[1];

    Vector3D<Real_t> blocalpoint(static_cast<Real_t>(localpoint[0]), static_cast<Real_t>(localpoint[1]),
                                 static_cast<Real_t>(localpoint[2]));

    Real_t safetySqr = static_cast<Real_t>(limit_sq);

    stack[0].node     = 0;
    stack[0].safetySq = fNodes[0].Bounds().SafetySqr(blocalpoint);

    if (return_top_estimate) {
      if (stack[0].safetySq > 0.) {
        return stack[0].safetySq;
      }
    }

    do {
      const auto top        = *--ptr;
      const unsigned int id = top.node;
      if (top.safetySq > safetySqr) continue;

      if (fNodes[id].IsLeaf()) {
        for (int i = 0; i < fNodes[id].count; ++i) {
          const int prim = fPrimId[fNodes[id].first + i];
          BVHPointQueryContext<Real_t> ctx{prim, fNodes[id].count, safetySqr};
          primitive_hook(ctx);
          safetySqr = vecCore::math::Min(safetySqr, ctx.safetySqr);
        }
      } else {
        const unsigned int childL = fNodes[id].first;
        const unsigned int childR = childL + 1;
        const Real_t safetySqrL   = fNodes[childL].Bounds().SafetySqr(blocalpoint);
        const Real_t safetySqrR   = fNodes[childR].Bounds().SafetySqr(blocalpoint);
        bool traverseL            = safetySqrL < safetySqr;
        bool traverseR            = safetySqrR < safetySqr;

        if (return_top_estimate) {
          constexpr float threshold = 0.1;
          if (traverseL && safetySqrL > threshold) {
            traverseL = false;
            safetySqr = vecCore::math::Min(safetySqr, safetySqrL);
          }
          if (traverseR && safetySqrR > threshold) {
            traverseR = false;
            safetySqr = vecCore::math::Min(safetySqr, safetySqrR);
          }
        }
        if (safetySqrR < safetySqrL) {
          if (traverseR) *ptr++ = StackItem{childR, safetySqrR};
          if (traverseL) *ptr++ = StackItem{childL, safetySqrL};
        } else {
          if (traverseL) *ptr++ = StackItem{childL, safetySqrL};
          if (traverseR) *ptr++ = StackItem{childR, safetySqrR};
        }
      }
    } while (ptr > stack);

    return safetySqr;
  }

private:
  int fRootId{0};
  int fNPrim{0};
  int fNNodes{0};
  int fMaxDepth{0};
  Node *fNodes{nullptr};
  int *fPrimId{nullptr};
  AABB<Real_t> *fAABBs{nullptr}; ///< Per-primitive AABBs, indexed by primitive id.

  /**
   * Ray-box test against @p bounds for a ray whose octant (3 bits, bit i set when the ray travels in
   * the negative direction along axis i) is already known. Unlike AABB::ComputeIntersectionInvDir,
   * this needs no per-axis min/max: the sign of @c invdir -- and hence which corner is entered first
   * along each axis -- depends only on the ray and is therefore constant for the whole traversal, so
   * the near/far corner per axis is selected directly from @p octant instead of being re-derived from
   * both corners at every node (mirrors madmann91 bvh::v2's @c Node::get_min_bounds/get_max_bounds).
   *
   * The ray's valid interval @c [0, step] is folded directly into the entry/exit reduction (the
   * `0` seeds the entry @c Max, @p step seeds the exit @c Min), so the caller's hit test collapses to a
   * single comparison @c tmin<=tmax instead of `tmin<=tmax && tmax>=0 && tmin<step` -- this mirrors
   * madmann91's `make_intersection_result` (node.h) and removes two compare-branches per child, which
   * profiling showed to be the dominant remaining IPC cost vs. ROOT once leaf granularity is matched.
   * @param[out] tmin Entry distance, clamped to >= 0 (used for near/far child ordering by the caller).
   * @return whether the ray crosses @p bounds within @c [0, step].
   * Keeps the same outward bias on the exit @c t as AABB::ComputeIntersectionInvDir for float-narrowing safety.
   */
  VECCORE_ATT_HOST_DEVICE
  static bool IntersectOctant(const Real_t (&bounds)[6], const Vector3D<Real_t> &point,
                              const Vector3D<Real_t> &invdir, unsigned int octant, Real_t step, Real_t &tmin)
  {
    // bit_i is 0/1 (not just truthy) so 2*axis+bit_i / 2*axis+(1-bit_i) are plain indexed loads into
    // the interleaved [min_x,max_x,min_y,max_y,min_z,max_z] array -- no compare/select per axis.
    const unsigned int bit0 = octant & 1u;
    const unsigned int bit1 = (octant >> 1) & 1u;
    const unsigned int bit2 = (octant >> 2) & 1u;

    const Real_t t0x = (bounds[bit0] - point[0]) * invdir[0];
    const Real_t t1x = (bounds[1 - bit0] - point[0]) * invdir[0];
    const Real_t t0y = (bounds[2 + bit1] - point[1]) * invdir[1];
    const Real_t t1y = (bounds[3 - bit1] - point[1]) * invdir[1];
    const Real_t t0z = (bounds[4 + bit2] - point[2]) * invdir[2];
    const Real_t t1z = (bounds[5 - bit2] - point[2]) * invdir[2];

    using vecCore::math::Max;
    using vecCore::math::Min;
    tmin              = Max(Max(Max(t0x, t0y), t0z), Real_t(0.));
    const Real_t tmax = Min(Min(Min(t1x, t1y), t1z) * (Real_t(1.) + vecgeom::kToleranceDist<Real_t>), step);
    return tmin <= tmax;
  }

  /**
   * Narrow @p v from @p From to @c Real_t, rounding outward (down if @c isMin, up otherwise) so the
   * result never lies inside the true interval [v, v] -- i.e. converting a double AABB corner to float
   * can only grow the box, never shrink it. A no-op whenever the narrowing is exact (e.g. Real_t==From).
   */
  template <bool isMin, typename From>
  VECCORE_ATT_HOST static Real_t RoundOutward(From v)
  {
    Real_t r = static_cast<Real_t>(v);
    if (isMin ? (static_cast<From>(r) > v) : (static_cast<From>(r) < v)) {
      r = isMin ? std::nextafter(r, -std::numeric_limits<Real_t>::infinity())
                : std::nextafter(r, std::numeric_limits<Real_t>::infinity());
    }
    return r;
  }

  /**
   * Compute a strict, stable 3D order along @p sortAxis, breaking ties using the
   * remaining two axes so STL sorting algorithms never see "equal" elements.
   */
  template <typename T>
  VECCORE_ATT_HOST static bool less3D(const T &left, const T &right, int sortAxis)
  {
    return left[sortAxis] < right[sortAxis] ||
           (left[sortAxis] == right[sortAxis] && (left[(sortAxis + 1) % 3] < right[(sortAxis + 1) % 3] ||
                                                  (left[(sortAxis + 1) % 3] == right[(sortAxis + 1) % 3] &&
                                                   left[(sortAxis + 2) % 3] < right[(sortAxis + 2) % 3])));
  }

  /**
   * Compute the surface areas of bounding boxes that surround the given primitives,
   * sweeping from left to right and vice-versa. See BVH.cpp::sweepSurfaceArea(), which
   * this is ported from verbatim.
   */
  VECCORE_ATT_HOST
  static std::vector<std::pair<double, double>> SweepSurfaceArea(AABB<Real_t> const *boxes, int const *begin,
                                                                 int const *end)
  {
    if (begin >= end) return {};

    std::vector<std::pair<double, double>> areas(std::distance(begin, end), {0., 0.});

    AABB<Real_t> box{boxes[*begin]};
    for (auto it = begin + 1; it < end; ++it) {
      areas[it - begin].first = box.SurfaceArea();
      box                     = AABB<Real_t>::Union(box, boxes[*it]);
    }

    AABB<Real_t> box2{boxes[*(end - 1)]};
    for (auto it = end - 1; it >= begin; --it) {
      box2                     = AABB<Real_t>::Union(box2, boxes[*it]);
      areas[it - begin].second = box2.SurfaceArea();
    }

    return areas;
  }

  /**
   * Surface-area-heuristic split, ported from BVH.cpp::surfaceAreaHeuristic() for build
   * parity. Sweeps all three axes looking for the split minimizing total child surface
   * area (with a balance penalty term to avoid degenerate splits), then partitions
   * [first,last) accordingly.
   * @return Iterator to the first element of the second group, or @p last if no split improves on a single leaf.
   */
  VECCORE_ATT_HOST
  static int *SurfaceAreaHeuristicSplit(AABB<Real_t> const *boxes, int *first, int *last)
  {
    int bestSplitAxis          = -1;
    double bestTraversalMetric = std::distance(first, last);
    int bestSplitObject        = -1;
    const auto nObj            = std::distance(first, last);

    int currentSortAxis = 0;
    auto sorter         = [boxes, &currentSortAxis](int a, int b) {
      const auto centroidA   = boxes[a].Center();
      const auto centroidB   = boxes[b].Center();
      constexpr double shift = 0.01;
      return less3D(centroidA + shift * (centroidA - boxes[a].Min()), centroidB + shift * (centroidB - boxes[b].Min()),
                    currentSortAxis);
    };

    for (int axis = 0; axis <= 2; ++axis) {
      currentSortAxis = axis;
      std::sort(first, last, sorter);

      const std::vector<std::pair<double, double>> sweep = SweepSurfaceArea(boxes, first, last);
      const auto totSurfArea                             = sweep.front().second;

      for (int *splitObject = first; splitObject < last; ++splitObject) {
        const auto left  = sweep[splitObject - first].first / totSurfArea;
        const auto right = sweep[splitObject - first].second / totSurfArea;

        const auto splitMetric = left * std::distance(first, splitObject) + right * std::distance(splitObject, last) +
                                 0.1 * std::abs(nObj / 2 - std::distance(first, splitObject) / nObj);

        if (splitMetric < bestTraversalMetric) {
          bestTraversalMetric = splitMetric;
          bestSplitAxis       = axis;
          bestSplitObject     = *splitObject;
        }
      }
    }

    if (bestSplitAxis == -1) return last;

    currentSortAxis = bestSplitAxis;
    return std::partition(first, last, [sorter, bestSplitObject](int i) { return sorter(i, bestSplitObject); });
  }

  /**
   * Recursively initialize node @p id for the primitive range [first,last), allocating
   * its two children (as a contiguous pair) when a useful split is found.
   * Implements @c BuildPolicy::Original.
   */
  VECCORE_ATT_HOST
  void BuildNodeOriginal(int id, int depth, AABB<Real_t> const *boxes, int *first, int *last)
  {
    Node &node  = fNodes[id];
    const int n = static_cast<int>(std::distance(first, last));

    AABB<Real_t> bbox = boxes[*first];
    for (int *it = first + 1; it != last; ++it)
      bbox = AABB<Real_t>::Union(bbox, boxes[*it]);
    node.SetBounds(bbox);

    if (n == 1 || depth >= fMaxDepth) {
      node.first = static_cast<int>(first - fPrimId);
      node.count = n;
      return;
    }

    int *pivot = SurfaceAreaHeuristicSplit(boxes, first, last);

    if (pivot == first || pivot == last) {
      node.first = static_cast<int>(first - fPrimId);
      node.count = n;
      return;
    }

    const int childL = fNNodes++;
    const int childR = fNNodes++;
    node.first       = childL;
    node.count       = 0;

    BuildNodeOriginal(childL, depth + 1, boxes, first, pivot);
    BuildNodeOriginal(childR, depth + 1, boxes, pivot, last);
  }

  /**
   * Partition [first,last) at the median centroid along the node's longest axis. Used as a fallback
   * by @c BuildPolicy::SweepSAH when no split improves on the SAH leaf cost (e.g. coincident centroids)
   * but the leaf-size cap still requires splitting. Always returns an iterator strictly between
   * @p first and @p last for n >= 2.
   */
  VECCORE_ATT_HOST
  static int *MedianSplitLongestAxis(AABB<Real_t> const *boxes, int *first, int *last, AABB<Real_t> const &bounds)
  {
    const auto size = bounds.Size();
    const int axis  = (size[0] > size[2]) ? (size[0] > size[1] ? 0 : 1) : (size[1] > size[2] ? 1 : 2);
    int *mid        = first + std::distance(first, last) / 2;
    std::nth_element(first, mid, last,
                     [boxes, axis](int a, int b) { return less3D(boxes[a].Center(), boxes[b].Center(), axis); });
    return mid;
  }

  /**
   * Cost-based sweep SAH split. Sweeps all three axes looking for the split with the lowest actual
   * SAH cost (c_trav + c_isect * sum of child-probability * child-count); the search baseline is the
   * cost of not splitting at all (a single leaf with @p n primitives), so a split is only reported if
   * it is genuinely cheaper.
   * @param[out] improves Set to whether a strictly cost-improving split was found.
   * @return Iterator to the first element of the second group when @p improves is true, otherwise @p last.
   */
  VECCORE_ATT_HOST
  static int *SweepSAHSplit(AABB<Real_t> const *boxes, int *first, int *last, Real_t nodeArea, bool &improves)
  {
    const auto n         = std::distance(first, last);
    double bestCost      = double(n) * double(kCostIntersect); // baseline: cost of a single leaf
    int bestSplitAxis    = -1;
    int bestSplitObject  = -1;
    const double invArea = 1.0 / double(nodeArea);

    int currentSortAxis = 0;
    auto sorter         = [boxes, &currentSortAxis](int a, int b) {
      const auto centroidA   = boxes[a].Center();
      const auto centroidB   = boxes[b].Center();
      constexpr double shift = 0.01;
      return less3D(centroidA + shift * (centroidA - boxes[a].Min()), centroidB + shift * (centroidB - boxes[b].Min()),
                    currentSortAxis);
    };

    for (int axis = 0; axis <= 2; ++axis) {
      currentSortAxis = axis;
      std::sort(first, last, sorter);

      const std::vector<std::pair<double, double>> sweep = SweepSurfaceArea(boxes, first, last);

      // k = number of elements in the left group; both groups must be non-empty.
      for (int *splitObject = first + 1; splitObject < last; ++splitObject) {
        const auto k      = std::distance(first, splitObject);
        const double cost = double(kCostTraversal) + double(kCostIntersect) * (sweep[k].first * invArea * k +
                                                                               sweep[k].second * invArea * (n - k));

        if (cost < bestCost) {
          bestCost        = cost;
          bestSplitAxis   = axis;
          bestSplitObject = *splitObject;
        }
      }
    }

    improves = (bestSplitAxis != -1);
    if (!improves) return last;

    currentSortAxis = bestSplitAxis;
    return std::partition(first, last, [sorter, bestSplitObject](int i) { return sorter(i, bestSplitObject); });
  }

  /**
   * Recursively initialize node @p id for the primitive range [first,last) using a cost-based sweep
   * SAH, capping leaves at @c kMaxLeafSize primitives. Implements @c BuildPolicy::SweepSAH.
   */
  VECCORE_ATT_HOST
  void BuildNodeSweepSAH(int id, int depth, AABB<Real_t> const *boxes, int *first, int *last)
  {
    Node &node  = fNodes[id];
    const int n = static_cast<int>(std::distance(first, last));

    AABB<Real_t> bbox = boxes[*first];
    for (int *it = first + 1; it != last; ++it)
      bbox = AABB<Real_t>::Union(bbox, boxes[*it]);
    node.SetBounds(bbox);

    if (n == 1 || depth >= fMaxDepth) {
      node.first = static_cast<int>(first - fPrimId);
      node.count = n;
      return;
    }

    const Real_t nodeArea = bbox.SurfaceArea();
    bool improves         = false;
    int *pivot            = (nodeArea > Real_t(0.)) ? SweepSAHSplit(boxes, first, last, nodeArea, improves) : last;

    if (!improves) {
      if (n <= kMaxLeafSize) {
        node.first = static_cast<int>(first - fPrimId);
        node.count = n;
        return;
      }
      // The leaf-size cap still requires a split, even though none improves the SAH cost
      // (e.g. coincident/duplicate boxes): fall back to a median split on the longest axis.
      pivot = MedianSplitLongestAxis(boxes, first, last, bbox);
      if (pivot == first || pivot == last) {
        // Truly degenerate (all primitives coincide): accept an oversized leaf rather than loop forever.
        node.first = static_cast<int>(first - fPrimId);
        node.count = n;
        return;
      }
    }

    const int childL = fNNodes++;
    const int childR = fNNodes++;
    node.first       = childL;
    node.count       = 0;

    BuildNodeSweepSAH(childL, depth + 1, boxes, first, pivot);
    BuildNodeSweepSAH(childR, depth + 1, boxes, pivot, last);
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_BASE_BVH_V2_H_
