//===-- kernel/ExtrudedImplementation.h ----------------------------------*- C++ -*-===//
//===--------------------------------------------------------------------------===//
/// @file ExtrudedImplementation.h
/// @brief Navigation dispatcher for simple and tessellated extruded solids.
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_KERNEL_EXTRUDEDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_EXTRUDEDIMPLEMENTATION_H_

#include <cstdio>
#include <VecCore/VecCore>
#include "VecGeom/base/Config.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/base/Vector3D.h"

#include "TessellatedImplementation.h"
#include "SExtruImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct ExtrudedImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, ExtrudedImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedExtruded;
class ExtrudedStruct;
class UnplacedExtruded;

/// @brief Dispatches extruded-solid navigation to SExtru or tessellated helpers.
/// @details Single-section simple extrusions use `SExtruImplementation`.
/// General multi-section extrusions use tessellated helpers, with an optional
/// host-only per-section path when `fUseTslSections` is enabled.
struct ExtrudedImplementation {

  using PlacedShape_t    = PlacedExtruded;
  using UnplacedStruct_t = ExtrudedStruct;
  using UnplacedVolume_t = UnplacedExtruded;

  /// @brief Test whether a local point is inside or on the tolerated surface.
  /// @tparam Real_v Floating-point scalar type.
  /// @param extruded Runtime extruded data.
  /// @param point Local point to classify.
  /// @param[out] inside True when @p point is contained by the selected helper.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &extruded,
                                                                    Vector3D<Real_v> const &point, bool &inside)
  {
    inside = false;
    if (extruded.fIsSxtru) {
      SExtruImplementation::Contains<Real_v>(extruded.fSxtruHelper, point, inside);
      return;
    }

#ifndef VECGEOM_ENABLE_CUDA
    if (extruded.fUseTslSections) {
      // Find the Z section
      int zIndex = extruded.FindZSegment(point[2]);
      if ((zIndex < 0) || (zIndex >= (int)extruded.GetNSegments())) return;
      inside = extruded.fTslSections[zIndex]->Contains(point);
      return;
    }
#endif
    TessellatedImplementation::Contains<Real_v>(extruded.fTslRuntimeHelper, point, inside);
  }

  /// @brief Classify a local point as inside, outside, or surface.
  /// @details Dispatches to the simple-extruded helper, the host-only
  /// per-section tessellated helper, or the runtime tessellated helper. The
  /// per-section path promotes points on the first or last z section to
  /// `kSurface` when they are within tolerance of the section plane.
  /// @tparam Real_v Floating-point scalar type.
  /// @tparam Inside_t Integer-like type used for `EInside` values.
  /// @param extruded Runtime extruded data.
  /// @param point Local point to classify.
  /// @param[out] inside Set to `kInside`, `kOutside`, or `kSurface`.
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &extruded,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    inside = EInside::kOutside;

    if (extruded.fIsSxtru) {
      SExtruImplementation::Inside<Real_v, Inside_t>(extruded.fSxtruHelper, point, inside);
      return;
    }

#ifndef VECGEOM_ENABLE_CUDA
    if (extruded.fUseTslSections) {
      const int nseg = (int)extruded.GetNSegments();
      int zIndex     = extruded.FindZSegment(point[2]);
      if ((zIndex < 0) || (zIndex > nseg)) return;
      inside = extruded.fTslSections[Min(zIndex, nseg - 1)]->Inside(point);
      if (inside == EInside::kOutside) return;
      if (inside == EInside::kInside) {
        // Need to check if point on Z section
        if (((zIndex == 0) || (zIndex == nseg)) &&
            vecCore::math::Abs(point[2] - extruded.fZPlanes[zIndex]) < kTolerance) {
          inside = EInside::kSurface;
        }
      } else {
        inside = EInside::kSurface;
      }
      return;
    }
#endif
    TessellatedImplementation::Inside<Real_v, Inside_t>(extruded.fTslRuntimeHelper, point, inside);
  }

  /// @brief Compute distance from an outside or surface point to enter the solid.
  /// @details Simple extrusions are delegated to `SExtruImplementation`.
  /// General extrusions use the tessellated runtime helper unless the optional
  /// host-only section accelerator is compiled in.
  /// @tparam Real_v Floating-point scalar type.
  /// @param extruded Runtime extruded data.
  /// @param point Local start point.
  /// @param direction Unit local direction.
  /// @param stepMax Maximum distance to consider.
  /// @param[out] distance Entry distance, `-1` for wrong-side starts in public
  /// helper paths, or `kInfLength` when no entry is found.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &extruded,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
// Note that Real_v is always double here
#ifdef EFFICIENT_TSL_DISTANCETOIN
    if (extruded.fUseTslSections) {
      // Check if the bounding box is hit
      const Vector3D<Real_v> invdir(Real_v(1.0) / NonZero(direction.x()), Real_v(1.0) / NonZero(direction.y()),
                                    Real_v(1.0) / NonZero(direction.z()));
      Vector3D<int> sign;
      sign[0]  = invdir.x() < 0;
      sign[1]  = invdir.y() < 0;
      sign[2]  = invdir.z() < 0;
      distance = BoxImplementation::IntersectCachedKernel2<Real_v, Real_v>(&extruded.fTslRuntimeHelper.fMinExtent,
                                                                           point, invdir, sign.x(), sign.y(), sign.z(),
                                                                           -kTolerance, InfinityLength<Real_v>());
      if (distance >= stepMax) return;

      // Perform explicit Inside check to detect wrong side points. This impacts
      // DistanceToIn performance by about 5% for all topologies
      // auto inside = ScalarInsideKernel(unplaced, point);
      // if (inside == kInside) return -1.;

      int zIndex     = extruded.FindZSegment(point[2]);
      const int zMax = extruded.GetNSegments();
      // Don't go out of bounds here, as the first/last segment should be checked
      // even if the point is outside of Z-bounds
      bool fromOutZ =
          (point[2] < extruded.fZPlanes[0] + kTolerance) || (point[2] > extruded.fZPlanes[zMax] - kTolerance);
      zIndex = zIndex < 0 ? 0 : (zIndex >= zMax ? zMax - 1 : zIndex);

      // Traverse Z-segments left or right depending on sign of direction
      bool goingRight = direction[2] >= 0;

      distance = InfinityLength<Real_v>();
      if (goingRight) {
        for (int zSegCount = zMax; zIndex < zSegCount; ++zIndex) {
          bool skipZ = fromOutZ && (zSegCount == 0);
          if (skipZ)
            distance = extruded.fTslSections[zIndex]->DistanceToIn<true>(point, direction, invdir.z(), stepMax);
          else
            distance = extruded.fTslSections[zIndex]->DistanceToIn<false>(point, direction, invdir.z(), stepMax);
          // No segment further away can be at a shorter distance to the point, so
          // if a valid distance is found, only endcaps remain to be investigated
          if (distance >= -kTolerance && distance < InfinityLength<Precision>()) break;
        }
      } else {
        // Going left
        for (; zIndex >= 0; --zIndex) {
          bool skipZ = fromOutZ && (zIndex == zMax);
          if (skipZ)
            distance = extruded.fTslSections[zIndex]->DistanceToIn<true>(point, direction, invdir.z(), stepMax);
          else
            distance = extruded.fTslSections[zIndex]->DistanceToIn<false>(point, direction, invdir.z(), stepMax);
          // No segment further away can be at a shorter distance to the point, so
          // if a valid distance is found, only endcaps remain to be investigated
          if (distance >= -kTolerance && distance < InfinityLength<Precision>()) break;
        }
      }
    }
#endif
    if (extruded.fIsSxtru)
      SExtruImplementation::DistanceToIn<Real_v>(extruded.fSxtruHelper, point, direction, stepMax, distance);
    else
      TessellatedImplementation::DistanceToIn<Real_v>(extruded.fTslRuntimeHelper, point, direction, stepMax, distance);
  }

  /// @brief Compute distance from an inside or surface point to leave the solid.
  /// @tparam Real_v Floating-point scalar type.
  /// @param extruded Runtime extruded data.
  /// @param point Local start point.
  /// @param direction Unit local direction.
  /// @param stepMax Maximum distance to consider.
  /// @param[out] distance Exit distance, `-1` for wrong-side starts in public
  /// helper paths, or the tessellated helper miss convention for finite
  /// `stepMax` limits.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &extruded,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    if (extruded.fIsSxtru)
      SExtruImplementation::DistanceToOut<Real_v>(extruded.fSxtruHelper, point, direction, stepMax, distance);
    else
      TessellatedImplementation::DistanceToOut<Real_v>(extruded.fTslRuntimeHelper, point, direction, stepMax, distance);
  }

  /// @brief Compute safety from an outside point to enter the solid.
  /// @tparam Real_v Floating-point scalar type.
  /// @param extruded Runtime extruded data.
  /// @param point Local point.
  /// @param[out] safety Conservative distance to the nearest entry boundary,
  /// with public helper sentinels supplied by the selected implementation.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &extruded,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    if (extruded.fIsSxtru)
      SExtruImplementation::SafetyToIn<Real_v>(extruded.fSxtruHelper, point, safety);
    else
      TessellatedImplementation::SafetyToIn<Real_v>(extruded.fTslRuntimeHelper, point, safety);
  }

  /// @brief Compute safety from an inside point to leave the solid.
  /// @tparam Real_v Floating-point scalar type.
  /// @param extruded Runtime extruded data.
  /// @param point Local point.
  /// @param[out] safety Conservative distance to the nearest exit boundary,
  /// with public helper sentinels supplied by the selected implementation.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &extruded,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    if (extruded.fIsSxtru)
      SExtruImplementation::SafetyToOut<Real_v>(extruded.fSxtruHelper, point, safety);
    else
      TessellatedImplementation::SafetyToOut<Real_v>(extruded.fTslRuntimeHelper, point, safety);
  }

  /// @brief Compute a surface normal from the selected extruded helper.
  /// @tparam Real_v Floating-point scalar type.
  /// @param extruded Runtime extruded data.
  /// @param point Local surface point.
  /// @param[out] valid True when a surface normal could be assigned.
  /// @return Outward normal, or a fallback vector when @p valid is false.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &extruded,
                                                                                    Vector3D<Real_v> const &point,
                                                                                    bool &valid)
  {
    // Computes the normal on a surface and returns it as a unit vector
    if (extruded.fIsSxtru) return SExtruImplementation::NormalKernel<Real_v>(extruded.fSxtruHelper, point, valid);
    return TessellatedImplementation::NormalKernel<Real_v>(extruded.fTslRuntimeHelper, point, valid);
  }

}; // end ExtrudedImplementation

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_EXTRUDEDIMPLEMENTATION_H_
