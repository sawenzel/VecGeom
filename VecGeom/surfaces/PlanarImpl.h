#ifndef VECGEOM_PLANAR_IMPL_H
#define VECGEOM_PLANAR_IMPL_H

#include <VecGeom/surfaces/SurfaceHelper.h>

namespace vgbrep {

/// @brief Partial specialization of surface helper for planar surfaces.
/// @tparam Real_t Floating-point precision type
template <typename Real_t>
struct SurfaceHelper<kPlanar, Real_t> {

  VECGEOM_FORCE_INLINE
  SurfaceHelper() = default;

  VECGEOM_FORCE_INLINE
  /// @brief Find signed distance to next intersection from local point.
  /// @param point Point in local surface coordinates
  /// @param dir Direction in the local surface coordinates
  /// @param flip_exiting Flag representing the logical XOR of the surface being exited and normal being flipped
  /// @param distance Computed distance to surface
  /// @return Validity of the intersection
  bool Intersect(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, bool flip_exiting, Real_t &distance)
  {
    // Just need to propagate to (xOy) plane
    bool surfhit = flip_exiting ^ (dir[2] < 0);
    distance     = surfhit ? -point[2] / vecgeom::NonZero(dir[2]) : -1;
    return surfhit;
  }

  VECGEOM_FORCE_INLINE
  /// @brief Computes the isotropic safe distance to unplaced surfaces
  /// @param point Point in local surface coordinates
  /// @param flip_exiting Flag representing the logical XOR of the surface being exited and normal being flipped
  /// @param distance Computed isotropic safety
  /// @param compute_onsurf Instructs to compute the projection of the point on surface
  /// @param onsurf Projection of the point on surface
  /// @return Validity of the calculation
  bool Safety(Vector3D<Real_t> const &point, bool flip_exiting, Real_t &distance, bool compute_onsurf,
              Vector3D<Real_t> &onsurf) const
  {
    distance = flip_exiting ? -point[2] : point[2];
    // Computing onsurf is cheap
    onsurf.Set(point[0], point[1], 0);
    return true;
  }
};

} // namespace vgbrep

#endif