#ifndef VECGEOM_PLANAR_IMPL_H
#define VECGEOM_PLANAR_IMPL_H

#include <VecGeom/surfaces/surf/SurfaceHelper.h>

namespace vgbrep {

/// @brief Partial specialization of surface helper for planar surfaces.
/// @tparam Real_t Floating-point precision type
template <typename Real_t>
struct SurfaceHelper<SurfaceType::kPlanar, Real_t> {

  SurfaceHelper() = default;

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /// @brief Inside half-space function
  /// @param point Point in local surface coordinates
  /// @param tol tolerance for determining if a point is inside or not
  /// @return True if the point is behind the normal within kTolerance (surface is included)
  bool Inside(Vector3D<Real_t> const &point, bool flip, Real_t tol = vecgeom::kToleranceStrict<Real_t>)
  {
    int flipsign = !flip ? 1 : -1;
    return point.z() < flipsign * tol;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /// @brief Find signed distance to next intersection from local point.
  /// @param point Point in local surface coordinates
  /// @param dir Direction in the local surface coordinates
  /// @param left_side Flag specifying if the surface is intersected from the left-side that defines the normal
  /// @param distance Computed distance to surface
  /// @return Validity of the intersection
  bool Intersect(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, bool left_side, Real_t &distance,
                 bool &two_solutions, Real_t &safety)
  {
    // Just need to propagate to (xOy) plane
    bool surfhit = left_side ^ (dir[2] < 0);
    distance     = -point[2] / vecgeom::NonZero(dir[2]);
    safety       = point[2];
    return surfhit;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /// @brief Computes the isotropic safe distance to unplaced surfaces
  /// @param point Point in local surface coordinates
  /// @param left_side Flag specifying if the surface is intersected from the left-side that defines the normal
  /// @param distance Computed isotropic safety
  /// @param onsurf Projection of the point on surface
  /// @return Validity of the calculation
  bool Safety(Vector3D<Real_t> const &point, bool left_side, Real_t &distance, Vector3D<Real_t> &onsurf) const
  {
    distance = left_side ? -point[2] : point[2];
    // Computing onsurf is cheap
    onsurf.Set(point[0], point[1], 0);
    return distance > -vecgeom::kToleranceDist<Real_t>;
  }
};

} // namespace vgbrep

#endif
