#ifndef VECGEOM_SPHERICAL_IMPL_H
#define VECGEOM_SPHERICAL_IMPL_H

#include <VecGeom/surfaces/surf/SurfaceHelper.h>
#include <VecGeom/surfaces/base/Equations.h>

namespace vgbrep {

template <typename Real_t>
struct SurfaceHelper<SurfaceType::kSpherical, Real_t> {
  SphData<Real_t> const *fSphData{nullptr};

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  SurfaceHelper(SphData<Real_t> const &sphdata) { fSphData = &sphdata; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /// @brief Inside half-space function
  /// @param point Point in local surface coordinates
  /// @return True if the point is behind the normal within kTolerance (surface is included)
  bool Inside(Vector3D<Real_t> const &point, bool flip)
  {
    int flipsign      = fSphData->IsFlipped() ? -1 : 1;
    int bool_flipsign = !flip ? 1 : -1;
    Real_t sphR       = fSphData->Radius();
    Real_t rho        = point.Mag();
    return flipsign * (rho - sphR) < bool_flipsign * vecgeom::kToleranceStrict<Real_t>;
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
    QuadraticCoef<Real_t> coef;
    Real_t roots[2];
    int numroots      = 0;
    bool flip_exiting = left_side ^ fSphData->IsFlipped();
    SphereEq<Real_t>(point, dir, fSphData->Radius(), coef);
    QuadraticSolver(coef, roots, numroots);
    two_solutions = (numroots == 2 && roots[0] > -vecgeom::kToleranceStrict<Real_t> &&
                     roots[1] > -vecgeom::kToleranceStrict<Real_t>);
    for (auto i = 0; i < numroots; ++i) {
      distance                = roots[i];
      Vector3D<Real_t> onsurf = point + distance * dir;
      Vector3D<Real_t> normal(onsurf[0], onsurf[1], 0);
      bool hit = flip_exiting ^ (dir.Dot(normal) < 0);
      // First solution giving a valid hit wins
      if (hit) {
        if (distance < -vecgeom::kToleranceStrict<Real_t> && distance < -fSphData->Radius()) {
          Real_t rho = point.Mag();
          safety     = fSphData->Radius() - rho;
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
  bool Safety(Vector3D<Real_t> const &point, bool left_side, Real_t &distance, Vector3D<Real_t> &onsurf) const
  {
    Real_t sphR = fSphData->Radius();
    Real_t rho  = point.Mag();
    distance    = left_side ? sphR - rho : rho - sphR;
    // the onsurf computation code is missing below

    return true;
  }
};

} // namespace vgbrep

#endif
