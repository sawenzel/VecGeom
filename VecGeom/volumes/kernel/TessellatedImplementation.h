//===-- kernel/TessellatedImplementation.h ----------------------------------*- C++ -*-===//
//===--------------------------------------------------------------------------===//
/// @file TessellatedImplementation.h
/// @brief Navigation kernels for tessellated runtime solids.
/// @author mihaela.gheata@cern.ch, sandro.wenzel@cern.ch

#ifndef VECGEOM_VOLUMES_KERNEL_TESSELLATEDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TESSELLATEDIMPLEMENTATION_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/TessellatedStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>
#include <atomic>
#include <iostream>
#include <cstdio>

// NOTE: The facet-intersection statistics below rely on host-only std::atomic
// counters and std::cerr. They are not available in device code, so the whole
// instrumentation block is made invisible to the CUDA compiler.
#ifndef VECCORE_CUDA
// NOTE: these must be C++17 "inline" variables (true single-instance
// globals with external linkage), not anonymous-namespace statics. This
// header is included in many translation units, and Contains/DistanceToIn/
// DistanceToOut as well as reset/enable/disable_counters_ARGH() are all
// `inline` functions whose duplicate per-TU bodies get folded independently
// by the linker. With anonymous-namespace (internal-linkage) counters, the
// folded body kept for e.g. enable_counters_ARGH() could end up baked to a
// *different* TU's copy of the counters than the folded body kept for
// Contains(), so toggling the flag from one call site would silently have
// no effect on what another call site reads (ODR violation, ub) - this is
// exactly the failure mode where the recorded counters always stayed at 0.
inline std::atomic<unsigned long long> gfacetCounterDistToOut{0};
inline std::atomic<unsigned long long> gfacetCounterContains{0};
inline std::atomic<unsigned long long> gfacetCounterDistToIn{0};
inline std::atomic<unsigned long long> gLeafDistToOut{0};
inline std::atomic<unsigned long long> gLeafContains{0};
inline std::atomic<unsigned long long> gLeafDistToIn{0};
inline std::atomic<unsigned long long> gInnerDistToOut{0};
inline std::atomic<unsigned long long> gInnerContains{0};
inline std::atomic<unsigned long long> gInnerDistToIn{0};

// Counting is off by default so that setup/point-generation calls (which also
// go through Contains/DistanceToIn/DistanceToOut) never pollute the stats;
// callers must explicitly enable counting around the code they want measured.
inline std::atomic<bool> gCountingEnabledARGH{false};

struct facetIntersectionStatsDumper {
  ~facetIntersectionStatsDumper()
  {
    unsigned long long Contain = gfacetCounterContains.load() / 2;
    unsigned long long toIN    = gfacetCounterDistToIn.load() / 2;
    unsigned long long toOut   = gfacetCounterDistToOut.load() / 2;
    unsigned long long C       = gLeafContains.load() / 2;
    unsigned long long DI      = gLeafDistToIn.load() / 2;
    unsigned long long DO      = gLeafDistToOut.load() / 2;
    unsigned long long innerC  = gInnerContains.load() / 2;
    unsigned long long innerDI = gInnerDistToIn.load() / 2;
    unsigned long long innerDO = gInnerDistToOut.load() / 2;
    if (Contain != 0 && toIN != 0 && toOut != 0) {
      std::cerr << "VecGeom total facet intersections: "
                << "Contains: " << Contain << " distToinside: " << toIN << " distToOut: " << toOut << std::endl;
    }
    if (C != 0 && DI != 0 && DO != 0) {
      std::cerr << "VecGeom total leaf intersections: "
                << "Contains: " << C << " distToinside: " << DI << " distToOut: " << DO << std::endl;
    }
    if (innerC != 0 && innerDI != 0 && innerDO != 0) {
      std::cerr << "VecGeom total inner intersections: "
                << "Contains: " << innerC << " distToinside: " << innerDI << " distToOut: " << innerDO << std::endl;
    }
  }
};
inline facetIntersectionStatsDumper gFacetIntersectionStatsDumper;

inline void reset_counters_ARGH()
{
  gfacetCounterDistToOut.store(0, std::memory_order_relaxed);
  gfacetCounterContains.store(0, std::memory_order_relaxed);
  gfacetCounterDistToIn.store(0, std::memory_order_relaxed);
  gLeafDistToOut.store(0, std::memory_order_relaxed);
  gLeafContains.store(0, std::memory_order_relaxed);
  gLeafDistToIn.store(0, std::memory_order_relaxed);
  gInnerDistToOut.store(0, std::memory_order_relaxed);
  gInnerContains.store(0, std::memory_order_relaxed);
  gInnerDistToIn.store(0, std::memory_order_relaxed);
}

inline void enable_counters_ARGH() { gCountingEnabledARGH.store(true, std::memory_order_relaxed); }

inline void disable_counters_ARGH() { gCountingEnabledARGH.store(false, std::memory_order_relaxed); }
#endif // VECCORE_CUDA
namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct TessellatedImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, TessellatedImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTessellated;
template <size_t NVERT, typename T>
class TessellatedStruct;
class UnplacedTessellated;

/// Whether the per-facet AABB pre-filter is applied during BVH traversal (the check_leaf_bb template arg
/// of BVH::Intersect / BVH_V2::Intersect). When enabled, each facet's own conservatively rounded-outward
/// AABB is tested before the full facet.Distance() call, rejecting candidates whose box the ray never
/// crosses; this is a pure performance optimization (never a false negative). Tied to the same build switch
/// as the BVH type in TessellatedStruct.h so the two configurations reproduce the historical setup.
#ifdef VECGEOM_TESSELLATED_BVH_V2
static constexpr bool kCheckFacetBB = true;
#else
static constexpr bool kCheckFacetBB = false;
#endif

/// @brief Implements tessellated-solid navigation using BVH facet queries.
/// @details Point classification uses a fixed test ray and parity counting.
/// Distance and safety helpers query the runtime facet BVH and leave public
/// wrong-side sentinels to the wrapper methods.
struct TessellatedImplementation {

  using PlacedShape_t    = PlacedTessellated;
  using UnplacedStruct_t = TessellatedRuntimeStruct<Precision>;
  using UnplacedVolume_t = UnplacedTessellated;

  /// @brief Test whether a point is inside the closed tessellated shell.
  /// @details A bounding-box rejection is followed by a parity count of facet
  /// intersections along the cached test direction.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tessellated Runtime tessellated data.
  /// @param point Local point to test.
  /// @param[out] contains True when the parity count indicates containment.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &tessellated,
                                                                    Vector3D<Real_v> const &point, bool &contains)
  {
    // quick check against bounding box
    contains = false;
    ABBoxImplementation::ABBoxContainsKernel(tessellated.fMinExtent, tessellated.fMaxExtent, point, contains);
    if (!contains) {
      return;
    }

    // more expensive check involving intersection with the BVH
    int parity_counter       = 0;
    int intersection_counter = 0;
    int leafCounter_1        = 0;
    int innerCounter         = 0;
    auto userhook_bvh        = [&](BVHIntersectContext<float> &ctx) {
      intersection_counter += 1;
      const auto primID    = ctx.primID;
      const auto &facet    = tessellated.fFacets[primID];
      const auto this_dist = facet.Distance(point, tessellated.fTestDir /*, CAN GIVE EPSILON*/);
      if (this_dist < InfinityLength<Real_v>()) {
        parity_counter++;
      }
      return false; // do not stop here because we might see another triangle at
    };
    tessellated.fBVH->Intersect<kCheckFacetBB>(
        point, tessellated.fTestDir, InfinityLength<Real_v>(), userhook_bvh, [&innerCounter] { innerCounter += 1; },
        [&leafCounter_1]() { leafCounter_1 += 1; });

    contains = (parity_counter % 2 == 1);
#ifndef VECCORE_CUDA
    if (gCountingEnabledARGH.load(std::memory_order_relaxed)) {
      gfacetCounterContains.fetch_add(intersection_counter, std::memory_order_relaxed);
      gLeafContains.fetch_add(leafCounter_1, std::memory_order_relaxed);
      gInnerContains.fetch_add(innerCounter, std::memory_order_relaxed);
    }
#endif
  }

  /// @brief Classify a local point as inside, outside, or surface.
  /// @details Uses the same parity count as `Contains`, but with a
  /// tolerance-expanded bounding box and an early surface test based on the
  /// perpendicular distance to a hit facet along the cached test direction.
  /// @tparam Real_v Floating-point scalar type.
  /// @tparam Inside_t Integer-like type used for `EInside` values.
  /// @param tessellated Runtime tessellated data.
  /// @param point Local point to classify.
  /// @param[out] inside Set to `kInside`, `kOutside`, or `kSurface`.
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &tessellated,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    // quick check against (tolerance enlarged) bounding box
    bool contains = false;
    ABBoxImplementation::ABBoxContainsKernel(tessellated.fMinExtent - Vector3D<Real_v>(kHalfTolerance),
                                             tessellated.fMaxExtent + Vector3D<Real_v>(kHalfTolerance), point,
                                             contains);
    if (!contains) {
      inside = kOutside;
      return;
    }

    int parity_counter = 0;
    bool onSurface     = false;
    auto userhook_bvh  = [&](BVHIntersectContext<float> &ctx) {
      const auto primID    = ctx.primID;
      const auto &facet    = tessellated.fFacets[primID];
      const auto this_dist = facet.Distance(point, tessellated.fTestDir, -kHalfTolerance);
      if (this_dist < InfinityLength<Real_v>()) {
        parity_counter++;
        // Check uniform surface thickness via perpendicular projection
        const auto sp     = facet.fNormal.Dot(tessellated.fTestDir);
        const auto d_perp = this_dist * std::abs(sp);
        if (d_perp < kHalfTolerance) {
          onSurface = true;
          return true; // stop BVH search here: early exit
        }
      }
      return false; // do not stop here because we might see another triangle at
    };
    tessellated.fBVH->Intersect<kCheckFacetBB>(point, tessellated.fTestDir, InfinityLength<Real_v>(), userhook_bvh);
    if (onSurface) {
      inside = kSurface;
      return;
    }
    contains = (parity_counter % 2 == 1);
    inside   = contains ? Inside_t(kInside) : Inside_t(kOutside);
  }

  /// @brief Compute entry distance without wrong-side classification.
  /// @details The BVH traversal rejects facets whose normal faces away from an
  /// incoming ray. Surface starts can be recorded as a zero-entry fallback, but
  /// an ordinary positive intersection within tolerance wins when present.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tessellated Runtime tessellated data.
  /// @param point Local start point.
  /// @param direction Unit local direction.
  /// @param stepMax Maximum distance to consider.
  /// @param[out] distance Nearest entry candidate or `kInfLength`.
  /// @param allow_surface_zero Allow a tolerated surface start to return zero.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToInNoConv(UnplacedStruct_t const &tessellated,
                                                                              Vector3D<Real_v> const &point,
                                                                              Vector3D<Real_v> const &direction,
                                                                              Real_v const &stepMax, Real_v &distance,
                                                                              bool allow_surface_zero = false)
  {
    distance                        = InfinityLength<Real_v>();
    bool found_surface_entry        = false;
    Real_v surface_entry_projection = Real_v(0.);
    int intersection_counter        = 0;
    int leaf_counter_2              = 0;
    int innerCounter                = 0;
    // NOTE: a quick intersection check against the outer bounding box is already done as part of the BVH
    // intersection and does not need to be done in addition here
    auto userhook_bvh = [&](BVHIntersectContext<float> &ctx) {
      intersection_counter += 1;
      const auto primID = ctx.primID;
      const auto &facet = tessellated.fFacets[primID];
      // we are checking a triangle. Rule out early by a simple normal check
      const auto sp = facet.fNormal.Dot(direction);
      if (allow_surface_zero && sp < -kToleranceDist<Real_v>) {
        const auto plane_dist = (point - facet.fVertices[0]).Dot(facet.fNormal);
        if (vecCore::math::Abs(plane_dist) < kToleranceDist<Real_v> &&
            facet.template SafetySq<Real_v>(point) < kToleranceDistSquared<Real_v>) {
          // Keep this as a fallback only: a positive ordinary intersection,
          // when found within normal tolerance, must win over the surface-zero
          // convention.
          found_surface_entry      = true;
          surface_entry_projection = Max(surface_entry_projection, -sp);
        }
      }
      const bool wrong_orientation = sp > 0.; // coming from outside the dot product must be negative
      if (wrong_orientation) {
        return false;
      }
      const auto this_dist = facet.Distance(point, direction, -kHalfTolerance);
      if (this_dist < distance) {
        // update stuff
        distance     = vecCore::math::Max(this_dist, 0.);
        ctx.step_max = this_dist; // important for bvh culling (double to float conversion)
      }
      return false; // do not stop here
    };
    tessellated.fBVH->Intersect<kCheckFacetBB>(
        point, direction, stepMax, userhook_bvh, [&innerCounter]() { innerCounter += 1; },
        [&leaf_counter_2]() { leaf_counter_2 += 1; });

    if (found_surface_entry &&
        (distance == InfinityLength<Real_v>() || distance * surface_entry_projection > kToleranceDist<Real_v>)) {
      distance = Real_v(0.);
    }
#ifndef VECCORE_CUDA
    if (gCountingEnabledARGH.load(std::memory_order_relaxed)) {
      gfacetCounterDistToIn.fetch_add(intersection_counter, std::memory_order_relaxed);
      gLeafDistToIn.fetch_add(leaf_counter_2, std::memory_order_relaxed);
      gInnerDistToIn.fetch_add(innerCounter, std::memory_order_relaxed);
    }
#endif
  }

  /// @brief Compute distance from an outside or surface point to enter.
  /// @details Performs public wrong-side classification before delegating to
  /// the no-convention helper. Inside starts return `-1`.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tessellated Runtime tessellated data.
  /// @param point Local start point.
  /// @param direction Unit local direction.
  /// @param stepMax Maximum distance to consider.
  /// @param[out] distance Entry distance, `-1`, or `kInfLength`.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &tessellated,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
#ifndef VECGEOM_NO_WRONGSIDE_CONV
    Inside_t inside;
    Inside<Real_v, Inside_t>(tessellated, point, inside);
    if (inside == kInside) {
      distance = Real_v(-1.);
      return;
    }
    const bool allow_surface_zero = inside == kSurface;
#else
    const bool allow_surface_zero = false;
#endif
    DistanceToInNoConv<Real_v>(tessellated, point, direction, stepMax, distance, allow_surface_zero);
  }

  /// @brief Compute exit distance without wrong-side classification.
  /// @details The BVH traversal considers outward-facing facets and accepts
  /// tolerated tangent/outward starts as zero exits on the current facet.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tessellated Runtime tessellated data.
  /// @param point Local start point.
  /// @param direction Unit local direction.
  /// @param stepMax Maximum distance to consider.
  /// @param[out] distance Nearest exit candidate, `stepMax` when no exit is
  /// found before a finite limit, or `kInfLength` for an unbounded miss.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOutNoConv(UnplacedStruct_t const &tessellated,
                                                                               Vector3D<Real_v> const &point,
                                                                               Vector3D<Real_v> const &direction,
                                                                               Real_v const &stepMax, Real_v &distance)
  {
    distance                 = InfinityLength<Real_v>();
    int intersection_counter = 0;
    int leaf_counter         = 0;
    int innerCounter         = 0;
    auto userhook_bvh        = [&](BVHIntersectContext<float> &ctx) {
      intersection_counter += 1;
      const auto primID = ctx.primID;
      const auto &facet = tessellated.fFacets[primID];
      // we are checking a triangle. Rule out early by a simple normal check
      const auto sp                = facet.fNormal.Dot(direction);
      const bool wrong_orientation = sp < 0.; // coming from inside the dot product must be positive
      const auto plane_dist        = (point - facet.fVertices[0]).Dot(facet.fNormal);
      // Accept t=0 for tangent/outward surface starts on the current facet;
      // genuinely inward starts must continue to the next exit surface.
      if (sp >= -kToleranceDist<Real_v> && vecCore::math::Abs(plane_dist) < kToleranceDist<Real_v>) {
        const auto safety_sq = facet.template SafetySq<Real_v>(point);
        if (safety_sq < kToleranceDistSquared<Real_v>) {
          distance     = Real_v(0.);
          ctx.step_max = 0.f;
          return true;
        }
      }
      if (wrong_orientation) {
        return false;
      }
      // the -kHalfTolerance is to get 0 if we are on surface
      const auto this_dist = facet.Distance(point, direction, -kHalfTolerance); // /*, CAN GIVE EPSILON*/);
      if (this_dist < distance) {
        // update stuff
        distance     = vecCore::math::Max(this_dist, 0.);
        ctx.step_max = this_dist; // important for bvh culling (double to float conversion)
      }
      return false; // do not stop here because we might see triangles
    };
    tessellated.fBVH->Intersect<kCheckFacetBB>(
        point, direction, stepMax, userhook_bvh, [&innerCounter]() { innerCounter += 1; },
        [&leaf_counter]() { leaf_counter += 1; });

    if (distance == InfinityLength<Real_v>() && stepMax < InfinityLength<Real_v>()) {
      distance = stepMax;
    }
#ifndef VECCORE_CUDA
    if (gCountingEnabledARGH.load(std::memory_order_relaxed)) {
      gfacetCounterDistToOut.fetch_add(intersection_counter, std::memory_order_relaxed);
      gLeafDistToOut.fetch_add(leaf_counter, std::memory_order_relaxed);
      gInnerDistToOut.fetch_add(innerCounter, std::memory_order_relaxed);
    }
#endif
  }

  /// @brief Compute distance from an inside or surface point to leave.
  /// @details Performs public wrong-side classification before delegating to
  /// the no-convention helper. Outside starts return `-1`.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tessellated Runtime tessellated data.
  /// @param point Local start point.
  /// @param direction Unit local direction.
  /// @param stepMax Maximum distance to consider.
  /// @param[out] distance Exit distance, `-1`, `stepMax` for finite-limit
  /// misses, or `kInfLength`.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &tessellated,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
#ifndef VECGEOM_NO_WRONGSIDE_CONV
    Inside_t inside;
    Inside<Real_v, Inside_t>(tessellated, point, inside);
    if (inside == kOutside) {
      distance = Real_v(-1.);
      return;
    }
#endif
    DistanceToOutNoConv<Real_v>(tessellated, point, direction, stepMax, distance);
  }

  /// @brief Compute unsigned safety to enter without wrong-side classification.
  /// @details Uses cached surface anchor points to seed an upper BVH query
  /// limit, then asks the facet BVH for a squared safety estimate.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tessellated Runtime tessellated data.
  /// @param point Local point.
  /// @param[out] safety Non-negative entry safety.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToInNoConv(UnplacedStruct_t const &tessellated,
                                                                            Vector3D<Real_v> const &point,
                                                                            Real_v &safety)
  {
    int isurf;

    // get a quick upper limit from the min-distance to fixed set of anchor points on the surface
    // --> this limits BVH search from the start
    // TODO: this should also be vectorizable on the CPU
    float upper_limit_sq = InfinityLength<float>();
    const float px = point.x(), py = point.y(), pz = point.z();
    for (int i = 0; i < TessellatedRuntimeStruct<float>::N; ++i) {
      float dx       = tessellated.fTestPoints_x[i] - px;
      float dy       = tessellated.fTestPoints_y[i] - py;
      float dz       = tessellated.fTestPoints_z[i] - pz;
      float dist2    = dx * dx + dy * dy + dz * dz;
      upper_limit_sq = vecCore::math::Min(dist2, upper_limit_sq);
    }

    constexpr bool approxSafety = true; // can return early (without detailed safety, but never 0)
    const Real_v safetysq       = SafetySq<Real_v, approxSafety, double>(tessellated, point, isurf, upper_limit_sq);
    safety                      = vecCore::math::Sqrt(safetysq);

    // Keep the NoConv helper unsigned; public wrappers own wrong-side conventions.
    // safety on boundary should be zero --> see ShapeTester
    if (safety < kTolerance) {
      safety = 0.;
    }
  }

  /// @brief Compute safety from an outside or surface point to enter.
  /// @details Performs public wrong-side classification before delegating to
  /// the no-convention helper. Inside starts return `-1`, and surface starts
  /// return zero.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tessellated Runtime tessellated data.
  /// @param point Local point.
  /// @param[out] safety Entry safety, `-1`, or zero for surface starts.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &tessellated,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
#ifndef VECGEOM_NO_WRONGSIDE_CONV
    Inside_t inside;
    Inside<Real_v, Inside_t>(tessellated, point, inside);
    if (inside == kInside) {
      safety = Real_v(-1.);
      return;
    }
    // Surface-point safety is defined as zero; avoid the expensive BVH safety query in this case.
    if (inside == kSurface) {
      safety = Real_v(0.);
      return;
    }
#endif
    SafetyToInNoConv<Real_v>(tessellated, point, safety);
  }

  /// @brief Compute unsigned safety to leave without wrong-side classification.
  /// @details Uses cached surface anchor points to seed an upper BVH query
  /// limit, then asks the facet BVH for a squared safety estimate.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tessellated Runtime tessellated data.
  /// @param point Local point.
  /// @param[out] safety Non-negative exit safety.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOutNoConv(UnplacedStruct_t const &tessellated,
                                                                             Vector3D<Real_v> const &point,
                                                                             Real_v &safety)
  {
    int isurf;
    // get a quick upper limit from the min-distance to fixed set of anchor points on the surface
    // --> this limits BVH search from the start
    // TODO: this should also be vectorizable on the CPU

    float upper_limit_sq = InfinityLength<float>();
    const float px = point.x(), py = point.y(), pz = point.z();
    for (int i = 0; i < TessellatedRuntimeStruct<float>::N; ++i) {
      float dx       = tessellated.fTestPoints_x[i] - px;
      float dy       = tessellated.fTestPoints_y[i] - py;
      float dz       = tessellated.fTestPoints_z[i] - pz;
      float dist2    = dx * dx + dy * dy + dz * dz;
      upper_limit_sq = vecCore::math::Min(dist2, upper_limit_sq);
    }

    constexpr bool approxSafety = true;
    Real_v safetysq             = SafetySq<Real_v, approxSafety, double>(tessellated, point, isurf, upper_limit_sq);
    safety                      = vecCore::math::Sqrt(safetysq);

    // Keep the NoConv helper unsigned; public wrappers own wrong-side conventions.
    // safety on boundary should be zero --> see ShapeTester
    if (safety < kTolerance) {
      safety = 0.;
    }
  }

  /// @brief Compute safety from an inside or surface point to leave.
  /// @details Performs public wrong-side classification before delegating to
  /// the no-convention helper. Outside starts return `-1`, and surface starts
  /// return zero.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tessellated Runtime tessellated data.
  /// @param point Local point.
  /// @param[out] safety Exit safety, `-1`, or zero for surface starts.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &tessellated,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
#ifndef VECGEOM_NO_WRONGSIDE_CONV
    Inside_t inside;
    Inside<Real_v, Inside_t>(tessellated, point, inside);
    if (inside == kOutside) {
      safety = Real_v(-1.);
      return;
    }
    // Surface-point safety is defined as zero; avoid the expensive BVH safety query in this case.
    if (inside == kSurface) {
      safety = Real_v(0.);
      return;
    }
#endif
    SafetyToOutNoConv<Real_v>(tessellated, point, safety);
  }

  /// @brief Compute the normal of the closest tessellated facet.
  /// @tparam Real_v Floating-point scalar type.
  /// @param tessellated Runtime tessellated data.
  /// @param point Local point.
  /// @param[out] valid True when a closest facet was identified.
  /// @return Facet normal, or zero when @p valid is false.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &tessellated,
                                                                                    Vector3D<Real_v> const &point,
                                                                                    bool &valid)
  {
    // Computes the normal on a surface and returns it as a unit vector
    int isurf = -1;

    // TODO: should constrain search space !
    SafetySq<Real_v, false, double>(tessellated, point, isurf);
    if (isurf != -1) {
      valid = true;
      return tessellated.fFacets[isurf].fNormal;
    }
    valid = false;
    return Vector3D<Real_v>(0., 0., 0.);
  }

  /// @brief Query the BVH for squared distance to the closest relevant facet.
  /// @tparam Real_v Floating-point scalar type returned to callers.
  /// @tparam ToIn Selects the BVH side convention for entry or exit safety.
  /// @tparam T Internal precision used for facet safety comparisons.
  /// @param tessellated Runtime tessellated data.
  /// @param point Local point.
  /// @param[out] isurf Index of the closest visited facet, or `-1`.
  /// @param limit_sq Initial squared-distance limit for BVH pruning.
  /// @return Squared safety estimate returned by the BVH query.
  template <typename Real_v, bool ToIn, typename T = float>
  VECCORE_ATT_HOST_DEVICE static Real_v SafetySq(UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point,
                                                 int &isurf, Real_v limit_sq = InfinityLength<Real_v>())
  {
    T safetysq = limit_sq;
    isurf      = -1;
    Vector3D<T> pointv(point);

    auto userhook = [&](BVHPointQueryContext<float> &ctx) {
      const T this_safety_sq = tessellated.fFacets[ctx.primID].template SafetySq<T>(pointv);
      if (this_safety_sq < safetysq) {
        safetysq = this_safety_sq;
        isurf    = ctx.primID;
      }
      ctx.safetySqr =
          vecCore::math::Min((float)this_safety_sq,
                             ctx.safetySqr); // for culling bvh --> need to round up (this one is definitely in float)
    };

    const auto safety_estimated_sq = tessellated.fBVH->PointQuery<ToIn>(point, userhook, limit_sq);
    // if (isurf != -1) {
    //  this is the best value
    //   return safetysq;
    // }
    // Somehow we'll need to know the safety was estimated or correct

    return safety_estimated_sq; // an estimate
  }
}; // end TessellatedImplementation

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_TESSELLATEDIMPLEMENTATION_H_
