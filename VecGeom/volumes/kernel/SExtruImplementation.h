/// @file SExtruImplementation.h
/// @brief Navigation kernels for simple extruded polygon solids.

#ifndef VECGEOM_VOLUMES_KERNEL_SEXTRUIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_SEXTRUIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/PolygonalShell.h"
#include "VecGeom/volumes/kernel/BoxImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct SExtruImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, SExtruImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedSExtru;
class PolygonalShell;
class UnplacedSExtruVolume;

/// @brief Implements scalar navigation for `UnplacedSExtruVolume`.
/// @details The solid is represented by a planar polygon extruded between two
/// z planes. Convex polygons use the specialized side-shell slab helpers.
/// Concave polygons use local winding and side-safety checks to distinguish
/// strict wrong-side starts in notches from tolerated side or cap surface starts.
struct SExtruImplementation {

  using PlacedShape_t    = PlacedSExtru;
  using UnplacedStruct_t = PolygonalShell;
  using UnplacedVolume_t = UnplacedSExtruVolume;

  /// @brief Test whether a point is outside the cached XY extent plus tolerance.
  /// @param unplaced Runtime SExtru shell data.
  /// @param point Local point.
  /// @param tolerance Tolerance added to the planar extent.
  /// @return True when the point cannot be on or inside the side shell.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOutsideXYExtent(UnplacedStruct_t const &unplaced,
                                                                             Vector3D<Real_v> const &point,
                                                                             Real_v tolerance)
  {
    return point.x() < unplaced.fPolygon.fMinX - tolerance || point.x() > unplaced.fPolygon.fMaxX + tolerance ||
           point.y() < unplaced.fPolygon.fMinY - tolerance || point.y() > unplaced.fPolygon.fMaxY + tolerance;
  }

  /// @brief Compute concave-polygon containment and closest side safety squared.
  /// @details This combines the winding test and nearest-edge distance in one
  /// polygon pass for concave wrong-side and safety checks.
  /// @param unplaced Runtime SExtru shell data.
  /// @param point Local point.
  /// @param[out] safetySqr Squared distance to the closest polygon edge.
  /// @return True when the XY projection is inside the polygon.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool ContainsAndSafetySqrConcave(UnplacedStruct_t const &unplaced,
                                                                                       Vector3D<Real_v> const &point,
                                                                                       Real_v &safetySqr)
  {
    bool inside = false;
    safetySqr   = Real_v(1E30);

    const auto &polygon = unplaced.fPolygon;
    const auto vertexX  = polygon.fVertices.x();
    const auto vertexY  = polygon.fVertices.y();
    const auto slopes   = polygon.fVertices.z();
    const auto size     = polygon.fVertices.size();
    const auto px       = point.x();
    const auto py       = point.y();

    for (size_t i = 0; i < size; ++i) {
      // Combine winding and closest-edge safety in one pass for concave wrong-side checks.
      const auto vertexYI = vertexY[i];
      const auto vertexYJ = polygon.fShiftedYJ[i];
      const bool crossesY = (vertexYI > py) != (vertexYJ > py);
      if (crossesY) {
        const auto vertexXI = vertexX[i];
        if (px < (slopes[i] * (py - vertexYI) + vertexXI)) inside = !inside;
      }

      const auto p1x = vertexX[i];
      const auto p1y = vertexY[i];
      const auto p2x = polygon.fShiftedXJ[i];
      const auto p2y = polygon.fShiftedYJ[i];
      const auto dx  = p2x - p1x;
      const auto dy  = p2y - p1y;
      auto dpx       = px - p1x;
      auto dpy       = py - p1y;

      const auto u = (dpx * dx + dpy * dy) * polygon.fInvLengthSqr[i];
      if (u > Real_v(1.)) {
        dpx = px - p2x;
        dpy = py - p2y;
      } else if (u >= Real_v(0.)) {
        dpx -= u * dx;
        dpy -= u * dy;
      }

      const auto edgeSafetySqr = dpx * dpx + dpy * dpy;
      if (edgeSafetySqr < safetySqr) safetySqr = edgeSafetySqr;
    }

    return inside;
  }

  /// @brief Test strict material ownership for concave `DistanceToIn` starts.
  /// @details The check excludes all z and side tolerance bands so surface
  /// starts remain navigation starts rather than wrong-side sentinels.
  /// @param unplaced Runtime SExtru shell data.
  /// @param point Local point.
  /// @return True when @p point is unambiguously inside the concave SExtru.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsStrictlyInsideConcave(UnplacedStruct_t const &unplaced,
                                                                                   Vector3D<Real_v> const &point)
  {
    const Real_v tolerance = kToleranceDist<Real_v>;
    if (point.z() <= unplaced.fLowerZ + tolerance || point.z() >= unplaced.fUpperZ - tolerance) return false;
    // Extent rejection avoids the polygon winding test for obvious outside points.
    if (IsOutsideXYExtent(unplaced, point, tolerance)) return false;
    if (!unplaced.fPolygon.Contains(point)) return false;

    int unused;
    // Keep wrong-side checks away from the side-surface tolerance band.
    return unplaced.fPolygon.SafetySqr(point, unused) > kToleranceDistSquared<Real_v>;
  }

  /// @brief Test strict wrong-side starts for concave `DistanceToOut`.
  /// @details Points in the side tolerance band are classified as surface
  /// starts, not wrong-side starts, even when the polygon winding test says
  /// outside.
  /// @param unplaced Runtime SExtru shell data.
  /// @param point Local point.
  /// @return True when @p point is unambiguously outside the concave SExtru.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOutsideConcave(UnplacedStruct_t const &unplaced,
                                                                            Vector3D<Real_v> const &point)
  {
    const Real_v tolerance = kToleranceDist<Real_v>;
    if (point.z() > unplaced.fUpperZ + tolerance || point.z() < unplaced.fLowerZ - tolerance) return true;
    // Beyond the 2D extent tolerance the point cannot be on the side surface.
    if (IsOutsideXYExtent(unplaced, point, tolerance)) return true;
    if (unplaced.fPolygon.Contains(point)) return false;

    int unused;
    // Points within the side-surface tolerance band are surface starts, not wrong-side starts.
    return unplaced.fPolygon.SafetySqr(point, unused) > kToleranceDistSquared<Real_v>;
  }

  /// @brief Test whether a local point is contained in or on the SExtru.
  /// @param unplaced Runtime SExtru shell data.
  /// @param p Local point.
  /// @param[out] inside Set to true for points inside the z slab and polygon.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &unplaced,
                                                                    Vector3D<Real_v> const &p, bool &inside)
  {
    inside = false;
    if (p.z() > Real_v(unplaced.fUpperZ) || p.z() < Real_v(unplaced.fLowerZ)) return;
    if (unplaced.fPolygon.IsConvex())
      inside = unplaced.fPolygon.ContainsConvex(p);
    else
      inside = unplaced.fPolygon.Contains(p);
  }

  /// @brief Classify a local point as inside, outside, or surface.
  /// @details Convex polygons delegate to the convex planar-polygon classifier.
  /// Concave polygons use the z caps first, then side safety to identify the
  /// diffuse surface band around edges and reentrant vertices.
  /// @param unplaced Runtime SExtru shell data.
  /// @param point Local point.
  /// @param[out] inside Set to `kInside`, `kOutside`, or `kSurface`.
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &unplaced,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    if (point.z() > unplaced.fUpperZ + kTolerance) {
      inside = vecgeom::kOutside;
      return;
    }
    if (point.z() < unplaced.fLowerZ - kTolerance) {
      inside = vecgeom::kOutside;
      return;
    }

    // check conditions for surface first
    bool onZ = Abs(point.z() - unplaced.fUpperZ) < kTolerance;
    onZ |= Abs(point.z() - unplaced.fLowerZ) < kTolerance;

    if (unplaced.fPolygon.IsConvex()) {
      inside = unplaced.fPolygon.InsideConvex(point);
      if (onZ && inside != vecgeom::kOutside) inside = vecgeom::kSurface;
      return;
    }

    if (onZ) {
      if (unplaced.fPolygon.Contains(point)) {
        inside = vecgeom::kSurface;
        return;
      }
    }

    // not on z-surface --> check other surface with safety for moment
    if (unplaced.fLowerZ <= point.z() && point.z() <= unplaced.fUpperZ) {
      int unused;
      auto s = unplaced.fPolygon.SafetySqr(point, unused);
      if (s < kTolerance * kTolerance) {
        inside = vecgeom::kSurface;
        return;
      }
    }

    bool c = false;
    Contains(unplaced, point, c);

    if (c)
      inside = vecgeom::kInside;
    else
      inside = vecgeom::kOutside;
    return;
  }

  /// @brief Compute distance from outside to enter the SExtru.
  /// @details Convex polygons use the side-shell interval helper. Concave
  /// starts that are strictly inside return `-1`; tolerated surface starts are
  /// allowed to enter through caps or through the side shell.
  /// @param polyshell Runtime SExtru shell data.
  /// @param p Local start point.
  /// @param dir Unit local direction.
  /// @param distance Set to entry distance, `kInfLength`, or `-1` for strict inside starts.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &polyshell,
                                                                        Vector3D<Real_v> const &p,
                                                                        Vector3D<Real_v> const &dir,
                                                                        Real_v const & /*stepMax*/, Real_v &distance)
  {
    if (polyshell.fPolygon.IsConvex()) {
      distance = polyshell.DistanceToInConvex(p, dir);
      return;
    }
    if (IsStrictlyInsideConcave(polyshell, p)) {
      distance = Real_v(-1.);
      return;
    }
    distance = Real_v(kInfLength);

    // Surface starts on z caps can enter through the cap before hitting a side.
    const auto s = (dir.z() > Real_v(0.)) ? p.z() - polyshell.fLowerZ : polyshell.fUpperZ - p.z();

    const auto canhit = (s < Real_v(kTolerance)) && (Abs(dir.z()) > kToleranceDist<Real_v>);
    if (canhit) {
      const auto dist    = -s / Abs(dir.z());
      const auto xInters = p.x() + dist * dir.x();
      const auto yInters = p.y() + dist * dir.y();

      const auto hits = polyshell.fPolygon.Contains(Vector3D<Real_v>(xInters, yInters, Real_v(0.)));

      if (hits) {
        distance = dist;
        return;
      }
    }

    // check collision with polyshell
    distance = polyshell.DistanceToIn(p, dir);
    return;
  }

  /// @brief Compute distance from inside or surface to leave the SExtru.
  /// @details Concave wrong-side starts return `-1`. Side-shell exits are tried
  /// before z caps so reentrant side boundaries keep their local ownership.
  /// Tolerated z-cap starts moving outward return zero.
  /// @param polyshell Runtime SExtru shell data.
  /// @param p Local start point.
  /// @param dir Unit local direction.
  /// @param distance Set to exit distance, `kInfLength`, or `-1` for wrong-side starts.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &polyshell,
                                                                         Vector3D<Real_v> const &p,
                                                                         Vector3D<Real_v> const &dir,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {
    if (polyshell.fPolygon.IsConvex()) {
      distance = polyshell.DistanceToOutConvex(p, dir);
      return;
    }
    distance = Real_v(-1.);
    if (IsOutsideConcave(polyshell, p)) return;

    const auto dshell = polyshell.DistanceToOut(p, dir);
    if (dshell < Real_v(kInfLength)) {
      distance = dshell;
      return;
    }
    // Tolerated Z-surface starts moving out are immediate exits; avoid
    // converting a tiny off-surface displacement into a negative distance.
    const auto exitsLowerZ = (Abs(p.z() - polyshell.fLowerZ) <= kToleranceDist<Real_v>) && (dir.z() < Real_v(0.));
    const auto exitsUpperZ = (Abs(p.z() - polyshell.fUpperZ) <= kToleranceDist<Real_v>) && (dir.z() > Real_v(0.));
    if (exitsLowerZ || exitsUpperZ) {
      distance = Real_v(0.);
      return;
    }
    if (Abs(dir.z()) <= kToleranceDist<Real_v>) {
      distance = Real_v(kInfLength);
      return;
    }
    const auto correctZ = (dir.z() > Real_v(0.)) ? Real_v(polyshell.fUpperZ) : Real_v(polyshell.fLowerZ);
    distance            = (correctZ - p.z()) / dir.z();
    return;
  }

  /// @brief Compute safety from outside to the SExtru boundary.
  /// @details Convex polygons use the maximum of z and side safety. Concave
  /// safety uses the bounding extent to short-circuit far outside points, then
  /// combines polygon containment and side safety to keep strict-inside starts
  /// at `-1`.
  /// @param polyshell Runtime SExtru shell data.
  /// @param point Local point.
  /// @param[out] safety Conservative distance to enter, zero on the surface, or `-1` inside.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &polyshell,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    if (polyshell.fPolygon.IsConvex()) {
      Real_v safeZ = vecCore::math::Max(polyshell.fLowerZ - point.z(), point.z() - polyshell.fUpperZ);
      safety       = vecCore::math::Max(safeZ, polyshell.fPolygon.SafetyConvex(point, false));
      return;
    }

    Vector3D<Precision> aMin, aMax;
    polyshell.Extent(aMin, aMax);

    bool isInExtent = false;
    ABBoxImplementation::ABBoxContainsKernelGeneric(aMin, aMax, point, isInExtent);

    // Outside the full extent, the box safety is a cheap conservative distance.
    if (!isInExtent) {
      const auto ssqr = ABBoxImplementation::ABBoxSafetySqr(aMin, aMax, point);
      if (ssqr <= 0.) {
        safety = 0.;
        return;
      }
      safety = std::sqrt(ssqr);
      return;
    }

    Real_v sideSafetySqr;
    const bool insideXY = ContainsAndSafetySqrConcave(polyshell, point, sideSafetySqr);
    if (insideXY && point.z() > polyshell.fLowerZ + kToleranceDist<Real_v> &&
        point.z() < polyshell.fUpperZ - kToleranceDist<Real_v> && sideSafetySqr > kToleranceDistSquared<Real_v>) {
      safety = Real_v(-1.);
      return;
    }

    const auto zSafety1 = polyshell.fLowerZ - point.z();
    const auto zSafety2 = polyshell.fUpperZ - point.z();
    if (Abs(zSafety1) < kTolerance || Abs(zSafety2) < kTolerance) {
      // Cap surface points over the polygon footprint have zero entry safety.
      if (insideXY) {
        safety = 0.;
        return;
      }
    }
    safety = std::sqrt(sideSafetySqr);
  }

  /// @brief Compute safety from inside to leave the SExtru.
  /// @details Concave starts outside the z slab, outside the XY extent, or
  /// outside the polygon by more than the side tolerance return `-1`; otherwise
  /// the minimum of side and z-cap safety is returned.
  /// @param polyshell Runtime SExtru shell data.
  /// @param point Local point.
  /// @param[out] safety Conservative distance to exit, zero on the surface, or `-1` outside.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &polyshell,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    if (polyshell.fPolygon.IsConvex()) {
      Real_v safeZ = vecCore::math::Min(point.z() - polyshell.fLowerZ, polyshell.fUpperZ - point.z());
      safety       = vecCore::math::Min(safeZ, polyshell.fPolygon.SafetyConvex(point, true));
      return;
    }
    if (point.z() > polyshell.fUpperZ + kToleranceDist<Real_v> ||
        point.z() < polyshell.fLowerZ - kToleranceDist<Real_v> ||
        IsOutsideXYExtent(polyshell, point, kToleranceDist<Real_v>)) {
      safety = Real_v(-1.);
      return;
    }

    Real_v sideSafetySqr;
    const bool insideXY = ContainsAndSafetySqrConcave(polyshell, point, sideSafetySqr);
    if (!insideXY && sideSafetySqr > kToleranceDistSquared<Real_v>) {
      safety = Real_v(-1.);
      return;
    }

    safety = std::sqrt(sideSafetySqr);
    safety = Min(safety, polyshell.fUpperZ - point.z());
    safety = Min(safety, point.z() - polyshell.fLowerZ);
  }

  /// @brief Compute an outward normal for a point on the SExtru surface.
  /// @details Z caps return axial normals. Side normals are taken from the
  /// nearest polygon edge; corner normals are not averaged.
  /// @param unplaced Runtime SExtru shell data.
  /// @param point Local surface point.
  /// @param[out] valid True when @p point is within the surface tolerance.
  /// @return Outward normal candidate.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &unplaced,
                                                                                    Vector3D<Real_v> const &point,
                                                                                    bool &valid)
  {
    valid = false;
    Vector3D<Real_v> normal(0., 0., 0.);

    bool onUpperZ = Abs(point.z() - unplaced.fUpperZ) < kTolerance;
    bool onLowerZ = Abs(point.z() - unplaced.fLowerZ) < kTolerance;

    if (onUpperZ || onLowerZ) {
      if (unplaced.fPolygon.Contains(point)) {
        valid = true;
        if (onUpperZ)
          normal = Vector3D<Real_v>(0., 0., 1);
        else {
          normal = Vector3D<Real_v>(0., 0., -1.);
        }
        return normal;
      }
    }

    // Side normals use the closest polygon edge; edge/corner averaging is intentionally not attempted.
    if (unplaced.fLowerZ <= point.z() && point.z() <= unplaced.fUpperZ) {
      int surfaceindex;
      auto s = unplaced.fPolygon.SafetySqr(point, surfaceindex);
      normal = Vector3D<Real_v>(-unplaced.fPolygon.fA[surfaceindex], -unplaced.fPolygon.fB[surfaceindex], 0.);
      if (s < kTolerance * kTolerance) {
        valid = true;
      }
    }
    return normal;
  }

}; // End struct SExtruImplementation
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
