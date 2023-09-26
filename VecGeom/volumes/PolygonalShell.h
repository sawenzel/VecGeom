#ifndef VECGEOM_POLYGONAL_SHELL_H
#define VECGEOM_POLYGONAL_SHELL_H

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlanarPolygon.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PolygonalShell;);
VECGEOM_DEVICE_DECLARE_CONV(class, PolygonalShell);

inline namespace VECGEOM_IMPL_NAMESPACE {

// a set of z-axis aligned rectangles
// looking from the z - direction the rectangles form a convex or concave polygon
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
      // vertex lengh x (fUpperZ - fLowerZ)
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

  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v DistanceToInConvex(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v done(false);
    Real_v result(kInfLength);
    const auto S = fPolygon.fVertices.size();

    for (size_t i = 0; i < S; ++i) { // side/rectangle index
      // approaching from right side?
      // under the assumption that surface normals points "inwards"
      const Real_v proj        = fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y();
      const Bool_v sidecorrect = proj >= -kTolerance;
      if (vecCore::MaskEmpty(sidecorrect)) {
        continue;
      }

      // the distance to the plane (specialized for fNormalsZ == 0)
      const Real_v pdist = fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i];

      const Bool_v moving_away = pdist > kTolerance;
      if (vecCore::MaskFull(moving_away)) {
        continue;
      }

      const Real_v dist = -pdist / NonZero(proj);

      // propagate to plane (first just z)
      const Real_v zInters(point.z() + dist * dir.z());
      const Bool_v zRangeOk = (zInters <= fUpperZ) && (zInters >= fLowerZ);
      if (!vecCore::MaskEmpty(zRangeOk)) {
        // check intersection with rest of rectangle
        const Real_v xInters(point.x() + dist * dir.x());
        const Real_v yInters(point.y() + dist * dir.y());

        // we could already check if intersection within the known extent
        const Bool_v intersects = fPolygon.OnSegment<Real_v, Precision, Bool_v>(i, xInters, yInters);

        vecCore::MaskedAssign(result, !done && intersects, dist);
        done |= intersects;
      }
      if (vecCore::MaskFull(done)) {
        return result;
      }
    }
    return result;
  }

  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v DistanceToInConcave(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Real_v result(kInfLength);
    const auto S = fPolygon.fVertices.size();

    for (size_t i = 0; i < S; ++i) { // side/rectangle index
      // approaching from right side?
      // under the assumption that surface normals points "inwards"
      const Real_v proj        = fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y();
      const Bool_v sidecorrect = proj >= -kTolerance;
      if (vecCore::MaskEmpty(sidecorrect)) {
        continue;
      }

      // the distance to the plane (specialized for fNormalsZ == 0)
      const Real_v pdist = fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i];

      const Bool_v moving_away = pdist > kTolerance;
      if (vecCore::MaskFull(moving_away)) {
        continue;
      }

      const Real_v dist = -pdist / NonZero(proj);

      // propagate to plane (first just z)
      const Real_v zInters(point.z() + dist * dir.z());
      const Bool_v zRangeOk = (zInters <= fUpperZ) && (zInters >= fLowerZ);
      if (!vecCore::MaskEmpty(zRangeOk)) {
        // check intersection with rest of rectangle
        const Real_v xInters(point.x() + dist * dir.x());
        const Real_v yInters(point.y() + dist * dir.y());

        // we could already check if intersection within the known extent
        const Bool_v intersects = fPolygon.OnSegment<Real_v, Precision, Bool_v>(i, xInters, yInters);

        vecCore__MaskedAssignFunc(result, intersects, Min(dist, result));
      }
      // if (vecCore::MaskFull(done)) {
      //        return result;
      //      }
    }
    return result;
  }

  // -- DistanceToOut --

  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    return fPolygon.IsConvex() ? DistanceToOutConvex(point, dir) : DistanceToOutConcave(point, dir);
  }

  // convex distance to out; checks for hits and aborts loop if hit found
  // NOTE: this kernel is the same as DistanceToIn apart from the comparisons for early return
  // these could become a template parameter
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v DistanceToOutConvex(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v done(false);
    Real_v result(kInfLength);
    const auto S = fPolygon.fVertices.size();

    for (size_t i = 0; i < S; ++i) { // side/rectangle index
      // approaching from right side?
      // under the assumption that surface normals points "inwards"
      const Real_v proj        = fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y();
      const Bool_v sidecorrect = proj <= kTolerance;
      if (vecCore::MaskEmpty(sidecorrect)) {
        continue;
      }

      // the distance to the plane (specialized for fNormalsZ == 0)
      const Real_v pdist = fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i];

      const Bool_v moving_away = pdist < -kTolerance;
      if (vecCore::MaskFull(moving_away)) {
        continue;
      }

      const Real_v dist = -pdist / NonZero(proj);

      // propagate to plane (first just z)
      const Real_v zInters(point.z() + dist * dir.z());
      const Bool_v zRangeOk = (zInters <= fUpperZ) && (zInters >= fLowerZ) && sidecorrect && !moving_away;
      if (!vecCore::MaskEmpty(zRangeOk)) {
        // check intersection with rest of rectangle
        const Real_v xInters(point.x() + dist * dir.x());
        const Real_v yInters(point.y() + dist * dir.y());

        // we could already check if intersection within the known extent
        const Bool_v intersects = fPolygon.OnSegment<Real_v, Precision, Bool_v>(i, xInters, yInters) && zRangeOk &&
                                  (dist >= -Real_v(kTolerance));

        vecCore::MaskedAssign(result, !done && intersects, dist);
        done |= intersects;
      }
      if (vecCore::MaskFull(done)) {
        return result;
      }
    }
    return result;
  }

  // DistanceToOut for the concave case
  // we should ideally combine this with the other kernel
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v DistanceToOutConcave(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Real_v result(kInfLength);
    const auto S = fPolygon.fVertices.size();

    for (size_t i = 0; i < S; ++i) { // side/rectangle index
      // approaching from right side?
      // under the assumption that surface normals points "inwards"
      const Real_v proj        = fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y();
      const Bool_v sidecorrect = proj <= kTolerance;
      if (vecCore::MaskEmpty(sidecorrect)) {
        continue;
      }

      // the distance to the plane (specialized for fNormalsZ == 0)
      const Real_v pdist = fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i];

      const Bool_v moving_away = pdist < -kTolerance;
      if (vecCore::MaskFull(moving_away)) {
        continue;
      }

      const Real_v dist = -pdist / NonZero(proj);

      // propagate to plane (first just z)
      const Real_v zInters(point.z() + dist * dir.z());
      const Bool_v zRangeOk = (zInters <= fUpperZ) && (zInters >= fLowerZ) && sidecorrect && !moving_away;
      if (!vecCore::MaskEmpty(zRangeOk)) {
        // check intersection with rest of rectangle
        const Real_v xInters(point.x() + dist * dir.x());
        const Real_v yInters(point.y() + dist * dir.y());

        // we could already check if intersection within the known extent
        const Bool_v intersects = fPolygon.OnSegment<Real_v, Precision, Bool_v>(i, xInters, yInters) && zRangeOk &&
                                  (dist >= -Real_v(kTolerance));

        vecCore__MaskedAssignFunc(result, intersects, Min(dist, result));
        // done |= intersects;
      }
      // if (vecCore::MaskFull(done)) {
      //  return result;
      //}
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
  Precision tmax = (vecCore::math::CopySign(dz, vz) - pz) / NonZero(vz);
  const auto S   = fPolygon.fVertices.size();
  for (size_t i = 0; i < S; ++i) { // side/rectangle index

    const Precision proj = -(fPolygon.fA[i] * dir.x() + fPolygon.fB[i] * dir.y());
    // normals pointing inwards
    const Precision pdist = -(fPolygon.fA[i] * point.x() + fPolygon.fB[i] * point.y() + fPolygon.fD[i]);
    if (pdist > kTolerance) return -kTolerance;
    if (proj > 0) {
      const Precision dist = -pdist / NonZero(proj);
      if (tmax > dist) tmax = dist;
    }
  }
  return tmax;
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
