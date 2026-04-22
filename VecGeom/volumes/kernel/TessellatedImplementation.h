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
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &tessellated,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    distance = InfinityLength<Real_v>();

    // NOTE: a quick intersection check against the outer bounding box is already done as part of the BVH
    // intersection and does not need to be done in addition here
    auto userhook_bvh = [&](BVHIntersectContext<float> &ctx) {
      const auto primID = ctx.primID;
      const auto &facet = tessellated.fFacets[primID];
      // we are checking a triangle. Rule out early by a simple normal check
      const auto sp                = (facet.fNormal).Dot(direction);
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
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &tessellated,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    distance = InfinityLength<Real_v>();

    auto userhook_bvh = [&](BVHIntersectContext<float> &ctx) {
      const auto primID = ctx.primID;
      const auto &facet = tessellated.fFacets[primID];
      // we are checking a triangle. Rule out early by a simple normal check
      const auto sp                = (facet.fNormal).Dot(direction);
      const bool wrong_orientation = sp < 0.; // coming from inside the dot product must be positive
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
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &tessellated,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
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
    safety          = vecCore::math::Sqrt(safetysq);

    // if a best surface was identified, we can check if we are on the wrong side
    // to satisfy VecGeom shape conventions. Works for points close the the surface but not deeply wrong
    if (isurf != -1) {
      const auto &v0  = tessellated.fFacets[isurf].fVertices[0];
      const auto &n   = tessellated.fFacets[isurf].fNormal;
      bool wrong_side = n.Dot(point - v0) < 0;
      if (wrong_side) {
        safety = -1.;
      }
    }

    // safety on boundary should be zero --> see ShapeTester
    if (safety < kTolerance) {
      safety = 0.;
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &tessellated,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    int isurf;
    // Real_v upper_limit_sq = InfinityLength<Real_v>();
    // get a quick upper limit from the min-distance to fixed set of anchor points on the surface
    // --> this limits BVH search from the start
    // TODO: this should also be vectorizable on the CPU

    float upper_limit_sq = InfinityLength<float>(); // std::numeric_limits<float>::max();
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
    safety          = vecCore::math::Sqrt(safetysq);

    if (isurf != -1) {
      const auto &v0  = tessellated.fFacets[isurf].fVertices[0];
      const auto &n   = tessellated.fFacets[isurf].fNormal;
      bool wrong_side = n.Dot(point - v0) > 0.;
      if (wrong_side) {
        safety = -1.;
      }
    }
    // safety on boundary should be zero --> see ShapeTester
    if (safety < kTolerance) {
      safety = 0.;
    }
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
    T safetysq      = limit_sq;
    isurf           = -1;
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
