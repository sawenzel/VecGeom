/// @file PolygonalShell.h
/// @brief Z-aligned side-shell helper for simple extruded polygons.

#ifndef VECGEOM_POLYGONAL_SHELL_H
#define VECGEOM_POLYGONAL_SHELL_H

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlanarPolygon.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PolygonalShell;);
VECGEOM_DEVICE_DECLARE_CONV(class, PolygonalShell);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// @brief Side-shell representation for a polygon extruded between two z planes.
/// @details Each polygon edge defines one z-aligned rectangular side surface.
/// The shell provides side-only distance helpers used by `SExtruImplementation`;
/// cap handling stays in the SExtru kernel.
class PolygonalShell : AlignedBase {

private:
  // the polygon (with friend access)
  PlanarPolygon fPolygon;
  Precision fLowerZ; // lower z plane
  Precision fUpperZ; // upper z plane

  friend class SimpleExtruPolygon;
  friend struct SExtruImplementation;
  friend class UnplacedSExtruVolume;

public:
  VECCORE_ATT_HOST_DEVICE
  PolygonalShell() : fPolygon() {}

  VECCORE_ATT_HOST_DEVICE
  PolygonalShell(int nvertices, Precision *x, Precision *y, Precision lowerz, Precision upperz)
      : fPolygon(nvertices, x, y), fLowerZ(lowerz), fUpperZ(upperz)
  {
  }

  VECCORE_ATT_HOST_DEVICE
  void Init(int nvertices, Precision *x, Precision *y, Precision lowerz, Precision upperz)
  {
    fPolygon.Init(nvertices, x, y);
    fLowerZ = lowerz;
    fUpperZ = upperz;
  }

  // the area of the shell ( does not include the area of the planar polygon )
  VECCORE_ATT_HOST_DEVICE
  Precision SurfaceArea() const
  {
    const auto kS = fPolygon.fVertices.size();
    Precision area(0.);
    for (size_t i = 0; i < kS; ++i) {
      // vertex length x (fUpperZ - fLowerZ)
      area += fPolygon.fLengthSqr[i];
    }
    return std::sqrt(area) * (fUpperZ - fLowerZ);
  }

  VECCORE_ATT_HOST_DEVICE
  PlanarPolygon const &GetPolygon() const { return fPolygon; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetLowerZ() const { return fLowerZ; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetUpperZ() const { return fUpperZ; }

  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE void Extent(Vector3D<Real_v> &aMin, Vector3D<Real_v> &aMax) const
  {
    aMin[0] = Real_v(fPolygon.GetMinX());
    aMin[1] = Real_v(fPolygon.GetMinY());
    aMin[2] = Real_v(fLowerZ);

    aMax[0] = Real_v(fPolygon.GetMaxX());
    aMax[1] = Real_v(fPolygon.GetMaxY());
    aMax[2] = Real_v(fUpperZ);
  }

  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v DistanceToIn(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    return fPolygon.IsConvex() ? DistanceToInConvex(point, dir) : DistanceToInConcave(point, dir);
  }

  /// @brief Compute the first side-shell entry for a convex polygon.
  /// @details Intersects candidate side planes, then filters by z range and
  /// segment ownership. Convex shells can return as soon as the first valid
  /// side hit is found.
  /// @param point Local start point.
  /// @param dir Unit local direction.
  /// @return Side entry distance or `kInfLength` when no side is hit.
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v DistanceToInConvex(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    bool done = false;
    Real_v result(kInfLength);
    const auto S = fPolygon.fVertices.size();

    for (size_t i = 0; i < S; ++i) { // side/rectangle index
      // approaching from right side?
      // under the assumption that surface normals points "inwards"
      const Real_v proj      = fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y();
      const bool sidecorrect = proj >= -kTolerance;
      if (!sidecorrect) continue;

      // the distance to the plane (specialized for fNormalsZ == 0)
      const Real_v pdist = fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i];

      const bool moving_away = pdist > kTolerance;
      if (moving_away) continue;

      const Real_v dist = -pdist / NonZero(proj);

      // propagate to plane (first just z)
      const Real_v zInters(point.z() + dist * dir.z());
      const Real_v tolerance = kToleranceDist<Real_v>;
      const bool zRangeOk    = (zInters <= fUpperZ + tolerance) && (zInters >= fLowerZ - tolerance);
      if (zRangeOk) {
        // check intersection with rest of rectangle
        const Real_v xInters(point.x() + dist * dir.x());
        const Real_v yInters(point.y() + dist * dir.y());

        // we could already check if intersection within the known extent
        const bool intersects = fPolygon.OnSegment<Real_v, Precision, bool>(i, xInters, yInters);

        if (!done && intersects) {
          result = Max(dist, Real_v(0.));
          done   = true;
        }
      }
      if (done) return result;
    }
    return result;
  }

  /// @brief Compute the nearest side-shell entry for a concave polygon.
  /// @details Concave shells may expose multiple side planes to the ray, so all
  /// accepted plane/segment candidates are reduced to the nearest non-negative
  /// entry distance.
  /// @param point Local start point.
  /// @param dir Unit local direction.
  /// @return Side entry distance or `kInfLength` when no side is hit.
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v DistanceToInConcave(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    Real_v result(kInfLength);
    const auto S = fPolygon.fVertices.size();

    for (size_t i = 0; i < S; ++i) { // side/rectangle index
      // approaching from right side?
      // under the assumption that surface normals points "inwards"
      const Real_v proj      = fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y();
      const bool sidecorrect = proj >= -kTolerance;
      if (!sidecorrect) continue;

      // the distance to the plane (specialized for fNormalsZ == 0)
      const Real_v pdist = fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i];

      const bool moving_away = pdist > kTolerance;
      if (moving_away) continue;

      const Real_v dist = -pdist / NonZero(proj);

      // propagate to plane (first just z)
      const Real_v zInters(point.z() + dist * dir.z());
      const Real_v tolerance = kToleranceDist<Real_v>;
      const bool zRangeOk    = (zInters <= fUpperZ + tolerance) && (zInters >= fLowerZ - tolerance);
      if (zRangeOk) {
        // check intersection with rest of rectangle
        const Real_v xInters(point.x() + dist * dir.x());
        const Real_v yInters(point.y() + dist * dir.y());

        // we could already check if intersection within the known extent
        const bool intersects = fPolygon.OnSegment<Real_v, Precision, bool>(i, xInters, yInters);

        if (intersects) result = Min(Max(dist, Real_v(0.)), result);
      }
    }
    return result;
  }

  // -- DistanceToOut --

  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    return fPolygon.IsConvex() ? DistanceToOutConvex(point, dir) : DistanceToOutConcave(point, dir);
  }

  /// @brief Compute the first side-shell exit for a convex polygon.
  /// @details This is the exit counterpart of `DistanceToInConvex`: candidate
  /// side-plane intersections are filtered by z range and segment ownership.
  /// Tolerated side-surface starts moving outward return zero.
  /// @param point Local start point.
  /// @param dir Unit local direction.
  /// @return Side exit distance or `kInfLength` when no side is hit.
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v DistanceToOutConvex(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    bool done = false;
    Real_v result(kInfLength);
    const auto S = fPolygon.fVertices.size();

    for (size_t i = 0; i < S; ++i) { // side/rectangle index
      // approaching from right side?
      // under the assumption that surface normals points "inwards"
      const Real_v proj = fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y();

      // the distance to the plane (specialized for fNormalsZ == 0)
      const Real_v pdist = fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i];

      // Tolerated side-surface starts moving out are immediate exits.
      const bool surfaceExit = (Abs(pdist) <= kToleranceDist<Real_v>) && (proj < Real_v(0.)) &&
                               (point.z() <= fUpperZ + kToleranceDist<Real_v>) &&
                               (point.z() >= fLowerZ - kToleranceDist<Real_v>) &&
                               fPolygon.OnSegment<Real_v, Precision, bool>(i, point.x(), point.y());
      if (!done && surfaceExit) {
        result = Real_v(0.);
        done   = true;
      }
      if (done) return result;

      const bool sidecorrect = proj <= -kTolerance;
      if (!sidecorrect) continue;

      const bool moving_away = pdist < -kTolerance;
      if (moving_away) continue;

      const Real_v dist = -pdist / NonZero(proj);

      // propagate to plane (first just z)
      const Real_v zInters(point.z() + dist * dir.z());
      const Real_v tolerance = kToleranceDist<Real_v>;
      const bool zRangeOk =
          (zInters <= fUpperZ + tolerance) && (zInters >= fLowerZ - tolerance) && sidecorrect && !moving_away;
      if (zRangeOk) {
        // check intersection with rest of rectangle
        const Real_v xInters(point.x() + dist * dir.x());
        const Real_v yInters(point.y() + dist * dir.y());

        // we could already check if intersection within the known extent
        const bool intersects =
            fPolygon.OnSegment<Real_v, Precision, bool>(i, xInters, yInters) && (dist >= -Real_v(kTolerance));

        if (!done && intersects) {
          result = dist;
          done   = true;
        }
      }
      if (done) return result;
    }
    return result;
  }

  /// @brief Compute the nearest side-shell exit for a concave polygon.
  /// @details Concave shells may have multiple valid side candidates, so the
  /// nearest accepted plane/segment hit is selected. Tolerated side-surface
  /// starts moving outward return zero.
  /// @param point Local start point.
  /// @param dir Unit local direction.
  /// @return Side exit distance or `kInfLength` when no side is hit.
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v DistanceToOutConcave(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    Real_v result(kInfLength);
    const auto S = fPolygon.fVertices.size();

    for (size_t i = 0; i < S; ++i) { // side/rectangle index
      // approaching from right side?
      // under the assumption that surface normals points "inwards"
      const Real_v proj = fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y();

      // the distance to the plane (specialized for fNormalsZ == 0)
      const Real_v pdist = fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i];

      // Tolerated side-surface starts moving out are immediate exits.
      const bool surfaceExit = (Abs(pdist) <= kToleranceDist<Real_v>) && (proj < Real_v(0.)) &&
                               (point.z() <= fUpperZ + kToleranceDist<Real_v>) &&
                               (point.z() >= fLowerZ - kToleranceDist<Real_v>) &&
                               fPolygon.OnSegment<Real_v, Precision, bool>(i, point.x(), point.y());
      if (surfaceExit) return Real_v(0.);

      const bool sidecorrect = proj < -kTolerance;
      if (!sidecorrect) continue;

      const bool moving_away = pdist < -kTolerance;
      if (moving_away) continue;

      const Real_v dist = -pdist / NonZero(proj);

      // propagate to plane (first just z)
      const Real_v zInters(point.z() + dist * dir.z());
      const Real_v tolerance = kToleranceDist<Real_v>;
      const bool zRangeOk =
          (zInters <= fUpperZ + tolerance) && (zInters >= fLowerZ - tolerance) && sidecorrect && !moving_away;
      if (zRangeOk) {
        // check intersection with rest of rectangle
        const Real_v xInters(point.x() + dist * dir.x());
        const Real_v yInters(point.y() + dist * dir.y());

        // we could already check if intersection within the known extent
        const bool intersects =
            fPolygon.OnSegment<Real_v, Precision, bool>(i, xInters, yInters) && (dist >= -Real_v(kTolerance));

        // Accepted negative roots are tolerated surface starts, not backward exits.
        if (intersects) result = Min(Max(dist, Real_v(0.)), result);
      }
    }
    return result;
  }

}; // end class

#define SPECIALIZATION
#ifdef SPECIALIZATION
// template specialization for Distance functions
template <>
VECCORE_ATT_HOST_DEVICE inline Precision PolygonalShell::DistanceToOutConvex(Vector3D<Precision> const &point,
                                                                             Vector3D<Precision> const &dir) const
{
  Precision dz         = 0.5 * (fUpperZ - fLowerZ);
  Precision pz         = point.z() - 0.5 * (fLowerZ + fUpperZ);
  const Precision safz = vecCore::math::Abs(pz) - dz;
  if (safz > kTolerance) return -kTolerance;

  Precision vz   = dir.z();
  Precision tmax = kInfLength;
  if (Abs(vz) > kTolerance) tmax = (vecCore::math::CopySign(dz, vz) - pz) / vz;
  const auto S = fPolygon.fVertices.size();
  for (size_t i = 0; i < S; ++i) { // side/rectangle index

    const Precision proj = -(fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y());
    // normals pointing inwards
    const Precision pdist = -(fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i]);
    if (pdist > kTolerance) return -kTolerance;
    if (proj > kTolerance) {
      const Precision dist = -pdist / proj;
      if (tmax > dist) tmax = dist;
    }
  }
  // Accepted surface exits may be slightly negative from plane arithmetic; the
  // navigation convention is an immediate zero step, not a backwards step.
  return Max(tmax, Precision(0.));
}

// template specialization for Distance functions
template <>
VECCORE_ATT_HOST_DEVICE inline Precision PolygonalShell::DistanceToInConvex(Vector3D<Precision> const &point,
                                                                            Vector3D<Precision> const &dir) const
{
  Precision dz = 0.5 * (fUpperZ - fLowerZ);
  Precision pz = point.z() - 0.5 * (fLowerZ + fUpperZ);
  if ((vecCore::math::Abs(pz) - dz) > -kTolerance && pz * dir.z() >= 0) return kInfLength;
  const Precision invz = -1. / NonZero(dir.z());
  const Precision ddz  = (invz < 0) ? dz : -dz;
  Precision tmin       = (pz + ddz) * invz;
  Precision tmax       = (pz - ddz) * invz;
  const auto S         = fPolygon.fVertices.size();
  for (size_t i = 0; i < S; ++i) { // side/rectangle index

    const Precision proj = -(fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y());
    // normals pointing inwards
    const bool moving_away = proj > -kTolerance;
    // the distance to the plane (specialized for fNormalsZ == 0)
    const Precision pdist   = -(fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i]);
    const bool side_correct = pdist > -kTolerance;
    if (side_correct) {
      if (moving_away) return kInfLength;
      const Precision dist = -pdist / NonZero(proj);
      if (dist > tmin) tmin = dist;
    } else if (moving_away) {
      const Precision dist = -pdist / NonZero(proj);
      if (dist < tmax) tmax = dist;
    }
  }
  if (tmax < tmin + kTolerance) return kInfLength;
  return tmin;
}

#endif

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
