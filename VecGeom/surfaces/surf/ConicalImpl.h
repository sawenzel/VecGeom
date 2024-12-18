#ifndef VECGEOM_CONICAL_IMPL_H
#define VECGEOM_CONICAL_IMPL_H

#include <VecGeom/surfaces/surf/SurfaceHelper.h>
#include <VecGeom/surfaces/base/Equations.h>

namespace vgbrep {

template <typename Real_t>
struct SurfaceHelper<SurfaceType::kConical, Real_t> {

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  SurfaceHelper() {}

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /// @brief Inside half-space function
  /// @param point Point in local surface coordinates
  /// @param tol tolerance for determining if a point is inside or not
  /// @return True if the point is behind the normal within kTolerance (surface is included)
  bool Inside(Vector3D<Real_t> const &point, bool flip, Real_t radius, Real_t slope,
              Real_t tol = vecgeom::kToleranceStrict<Real_t>)
  {
    int flipsign      = (radius < 0) ? -1 : 1;
    int bool_flipsign = !flip ? 1 : -1;
    Real_t coneR      = Abs(radius) + point.z() * slope;
    Real_t rho        = point.Perp();
    return flipsign * (rho - coneR) < bool_flipsign * tol;
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
                 bool &two_solutions, Real_t &safety, Real_t radius, Real_t slope)
  {
    QuadraticCoef<Real_t> coef;
    Real_t roots[2];
    int numroots      = 0;
    bool flip_exiting = left_side ^ (radius < 0);
    ConeEq<Real_t>(point, dir, Abs(radius), slope, coef);
    QuadraticSolver(coef, roots, numroots);
    two_solutions = (numroots == 2 && roots[0] > -vecgeom::kToleranceStrict<Real_t> &&
                     roots[1] > -vecgeom::kToleranceStrict<Real_t>);
    for (auto i = 0; i < numroots; ++i) {
      distance                = roots[i];
      Vector3D<Real_t> onsurf = point + distance * dir;
      // Exclude solutions beyond the tip of the cone. What if the tip is included? TODO
      if (Abs(radius) + onsurf[2] * slope < 0) continue;
      Vector3D<Real_t> normal(onsurf[0], onsurf[1], -std::sqrt(onsurf[0] * onsurf[0] + onsurf[1] * onsurf[1]) * slope);
      bool hit = flip_exiting ^ (dir.Dot(normal) < 0);
      // First solution giving a valid hit wins
      if (hit) {
        if (distance < -vecgeom::kToleranceStrict<Real_t>) {
          Real_t rho     = point.Perp();
          auto distanceR = Abs(radius) + point[2] * slope - rho;
          Real_t calf    = static_cast<Real_t>(1) / std::sqrt(static_cast<Real_t>(1) + slope * slope);
          safety         = distanceR * calf;
        }
        return true;
      }
    }
    return false;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /// @brief Computes the isotropic safe distance to unplaced surfaces
  /// @param point Point in local surface coordinates
  /// @param left_side Flag specifying if the surface is intersected from the left-side that defines the normal
  /// @param distance Computed isotropic safety
  /// @param onsurf Projection of the point on surface
  /// @return Validity of the calculation
  bool Safety(Vector3D<Real_t> const &point, bool left_side, Real_t &distance, Vector3D<Real_t> &onsurf, Real_t radius,
              Real_t slope) const
  {
    Real_t coneR      = Abs(radius) + point[2] * slope;
    Real_t rho        = point.Perp();
    bool flip_exiting = left_side ^ (radius < 0);
    auto distanceR    = flip_exiting ? coneR - rho : rho - coneR;
    Real_t calf       = static_cast<Real_t>(1) / std::sqrt(static_cast<Real_t>(1) + slope * slope);
    distance          = distanceR * calf;
    // We only use for the ZPhi frame safety the z of the point propagated on the cone surface
    onsurf = point;
    onsurf[2] += distance * slope * calf;
    return distance > -vecgeom::kToleranceDist<Real_t>;
  }
};

} // namespace vgbrep

#endif
