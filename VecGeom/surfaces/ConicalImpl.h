#ifndef VECGEOM_CONICAL_IMPL_H
#define VECGEOM_CONICAL_IMPL_H

#include <VecGeom/surfaces/SurfaceHelper.h>
#include <VecGeom/surfaces/Equations.h>

namespace vgbrep {

template <typename Real_t>
struct SurfaceHelper<kConical, Real_t> {
  ConeData<Real_t> const *fConeData{nullptr};

  VECGEOM_FORCE_INLINE
  SurfaceHelper(ConeData<Real_t> const &conedata) { fConeData = &conedata; }

  VECGEOM_FORCE_INLINE
  /// @brief Find signed distance to next intersection from local point.
  /// @param point Point in local surface coordinates
  /// @param dir Direction in the local surface coordinates
  /// @param flip_exiting Flag representing the logical XOR of the surface being exited and normal being flipped
  /// @param distance Computed distance to surface
  /// @return Validity of the intersection
  bool Intersect(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, bool flip_exiting, Real_t &distance)
  {
    QuadraticCoef<Real_t> coef;
    Real_t roots[2];
    int numroots = 0;
    flip_exiting ^= fConeData->IsFlipped();
    ConeEq<Real_t>(point, dir, fConeData->Radius(), fConeData->Slope(), coef);
    QuadraticSolver(coef, roots, numroots);
    for (auto i = 0; i < numroots; ++i) {
      distance                = roots[i];
      Vector3D<Real_t> onsurf = point + distance * dir;
      Vector3D<Real_t> normal(onsurf[0], onsurf[1],
                              -std::sqrt(onsurf[0] * onsurf[0] + onsurf[1] * onsurf[1]) * fConeData->Slope());
      bool hit = flip_exiting ^ (dir.Dot(normal) < 0);
      // First solution giving a valid hit wins
      if (hit) return true;
    }
    return false;
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
    Real_t t       = fConeData->Slope();
    Real_t coneR   = std::abs(fConeData->Radius() + point[2] * t);
    Real_t rho     = point.Perp();
    auto distanceR = flip_exiting ? coneR - rho : rho - coneR;
    Real_t calf    = Real_t(1) / std::sqrt(Real_t(1) + t * t);
    distance       = distanceR * calf;
    // the onsurf computation code is missing below

    return true;
  }
};

} // namespace vgbrep

#endif