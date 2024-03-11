#ifndef VECGEOM_CONICAL_IMPL_H
#define VECGEOM_CONICAL_IMPL_H

#include <VecGeom/surfaces/surf/SurfaceHelper.h>
#include <VecGeom/surfaces/base/Equations.h>

namespace vgbrep {

template <typename Real_t>
struct SurfaceHelper<SurfaceType::kConical, Real_t> {
  ConeData<Real_t> const *fConeData{nullptr};

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  SurfaceHelper(ConeData<Real_t> const &conedata) { fConeData = &conedata; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /// @brief Inside half-space function
  /// @param point Point in local surface coordinates
  /// @return True if the point is behind the normal within kTolerance (surface is included)
  bool Inside(Vector3D<Real_t> const &point)
  {
    int flipsign = fConeData->IsFlipped() ? -1 : 1;
    Real_t coneR = fConeData->RadiusZ(point.z());
    Real_t rho   = point.Perp();
    return flipsign * (rho - coneR) < vecgeom::kTolerance;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /// @brief Find signed distance to next intersection from local point.
  /// @param point Point in local surface coordinates
  /// @param dir Direction in the local surface coordinates
  /// @param left_side Flag specifying if the surface is intersected from the left-side that defines the normal
  /// @param distance Computed distance to surface
  /// @return Validity of the intersection
  bool Intersect(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, bool left_side, Real_t &distance)
  {
    QuadraticCoef<Real_t> coef;
    Real_t roots[2];
    int numroots      = 0;
    bool flip_exiting = left_side ^ fConeData->IsFlipped();
    ConeEq<Real_t>(point, dir, fConeData->Radius(), fConeData->slope, coef);
    QuadraticSolver(coef, roots, numroots);
    for (auto i = 0; i < numroots; ++i) {
      distance                = roots[i];
      Vector3D<Real_t> onsurf = point + distance * dir;
      // Exclude solutions beyond the tip of the cone. What if the tip is included? TODO
      if (fConeData->Radius() + onsurf[2] * fConeData->slope < 0) continue;
      Vector3D<Real_t> normal(onsurf[0], onsurf[1],
                              -std::sqrt(onsurf[0] * onsurf[0] + onsurf[1] * onsurf[1]) * fConeData->slope);
      bool hit = flip_exiting ^ (dir.Dot(normal) < 0);
      // First solution giving a valid hit wins
      if (hit) return true;
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
  bool Safety(Vector3D<Real_t> const &point, bool left_side, Real_t &distance, Vector3D<Real_t> &onsurf) const
  {
    Real_t t          = fConeData->slope;
    Real_t coneR      = fConeData->RadiusZ(point[2]);
    Real_t rho        = point.Perp();
    bool flip_exiting = left_side ^ fConeData->IsFlipped();
    auto distanceR    = flip_exiting ? coneR - rho : rho - coneR;
    Real_t calf       = Real_t(1) / std::sqrt(Real_t(1) + t * t);
    distance          = distanceR * calf;
    // We only use for the ZPhi frame safety the z of the point propagated on the cone surface
    onsurf = point;
    onsurf[2] += distanceR * t / (Real_t(1) + t * t);
    return true;
  }
};

} // namespace vgbrep

#endif
