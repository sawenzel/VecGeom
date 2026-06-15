//===-- kernel/TessellatedImplementation.h ----------------------------------*- C++ -*-===//
//===--------------------------------------------------------------------------===//
/// @file TessellatedImplementation.h
/// @author mihaela.gheata@cern.ch, sandro.wenzel@cern.ch

#ifndef VECGEOM_VOLUMES_KERNEL_TESSELLATEDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TESSELLATEDIMPLEMENTATION_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/TessellatedStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct TessellatedImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, TessellatedImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTessellated;
template <size_t NVERT, typename T>
class TessellatedStruct;
class UnplacedTessellated;

struct TessellatedImplementation {

  using PlacedShape_t    = PlacedTessellated;
  using UnplacedStruct_t = TessellatedRuntimeStruct<Precision>;
  using UnplacedVolume_t = UnplacedTessellated;

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &tessellated,
                                                                    Vector3D<Real_v> const &point, Bool_v &contains)
  {
    // quick check against bounding box
    contains = false;
    ABBoxImplementation::ABBoxContainsKernel(tessellated.fMinExtent, tessellated.fMaxExtent, point, contains);
    if (!contains) {
      return;
    }

    // more expensive check involving intersection with the BVH
    int parity_counter = 0;
    auto userhook_bvh  = [&](BVHIntersectContext<float> &ctx) {
      const auto primID    = ctx.primID;
      const auto &facet    = tessellated.fFacets[primID];
      const auto this_dist = facet.Distance(point, tessellated.fTestDir /*, CAN GIVE EPSILON*/);
      if (this_dist < InfinityLength<Real_v>()) {
        parity_counter++;
      }
      return false; // do not stop here because we might see another triangle at
    };
    tessellated.fBVH->Intersect<false>(point, tessellated.fTestDir, InfinityLength<Real_v>(), userhook_bvh);
    contains = (parity_counter % 2 == 1);
  }

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &tessellated,
                                                                  Vector3D<Real_v> const &point, Inside_v &inside)
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
    tessellated.fBVH->Intersect<false>(point, tessellated.fTestDir, InfinityLength<Real_v>(), userhook_bvh);
    if (onSurface) {
      inside = kSurface;
      return;
    }
    contains = (parity_counter % 2 == 1);
    inside   = contains ? Inside_v(kInside) : Inside_v(kOutside);
  }

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

    // NOTE: a quick intersection check against the outer bounding box is already done as part of the BVH
    // intersection and does not need to be done in addition here
    auto userhook_bvh = [&](BVHIntersectContext<float> &ctx) {
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
    tessellated.fBVH->Intersect<false>(point, direction, stepMax, userhook_bvh);
    if (found_surface_entry &&
        (distance == InfinityLength<Real_v>() || distance * surface_entry_projection > kToleranceDist<Real_v>)) {
      distance = Real_v(0.);
    }
  }

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

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOutNoConv(UnplacedStruct_t const &tessellated,
                                                                               Vector3D<Real_v> const &point,
                                                                               Vector3D<Real_v> const &direction,
                                                                               Real_v const &stepMax, Real_v &distance)
  {
    distance = InfinityLength<Real_v>();

    auto userhook_bvh = [&](BVHIntersectContext<float> &ctx) {
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
    tessellated.fBVH->Intersect<false>(point, direction, stepMax, userhook_bvh);
    if (distance == InfinityLength<Real_v>() && stepMax < InfinityLength<Real_v>()) {
      distance = stepMax;
    }
  }

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

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(
      UnplacedStruct_t const &tessellated, Vector3D<Real_v> const &point, typename vecCore::Mask_v<Real_v> &valid)
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
