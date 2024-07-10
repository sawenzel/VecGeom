#ifndef VECGEOM_CYLINDRICAL_IMPL_H
#define VECGEOM_CYLINDRICAL_IMPL_H

#include <VecGeom/surfaces/surf/SurfaceHelper.h>
#include <VecGeom/surfaces/base/Equations.h>

namespace vgbrep {

template <typename Real_t>
struct SurfaceHelper<SurfaceType::kCylindrical, Real_t> {
  CylData<Real_t> const *fCylData{nullptr};

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  SurfaceHelper(CylData<Real_t> const &cyldata) { fCylData = &cyldata; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /// @brief Inside half-space function
  /// @param point Point in local surface coordinates
  /// @return True if the point is behind the normal within kTolerance (surface is included)
  bool Inside(Vector3D<Real_t> const &point, bool flip)
  {
    int flipsign      = fCylData->IsFlipped() ? -1 : 1;
    int bool_flipsign = !flip ? 1 : -1;
    Real_t cylR       = fCylData->Radius();
    Real_t rho        = point.Perp();
    return flipsign * (rho - cylR) < bool_flipsign * vecgeom::kTolerance;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /// @brief Find signed distance to next intersection from local point.
  /// @param point Point in local surface coordinates
  /// @param dir Direction in the local surface coordinates
  /// @param left_side Flag specifying if the surface is intersected from the left-side that defines the normal
  /// @param distance Computed distance to surface
  /// @param two_solutions whether there are two possible solutions
  /// @return Validity of the intersection
  bool Intersect(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, bool left_side, Real_t &distance,
                 bool &two_solutions, Real_t &safety)
  {
    QuadraticCoef<Real_t> coef;
    Real_t roots[2];
    int numroots      = 0;
    bool flip_exiting = left_side ^ fCylData->IsFlipped();
    CylinderEq<Real_t>(point, dir, fCylData->Radius(), coef);
    QuadraticSolver(coef, roots, numroots);
    two_solutions = (numroots == 2 && roots[0] > -vecgeom::kTolerance && roots[1] > -vecgeom::kTolerance);
    for (auto i = 0; i < numroots; ++i) {
      distance                = roots[i];
      Vector3D<Real_t> onsurf = point + distance * dir;
      Vector3D<Real_t> normal(onsurf[0], onsurf[1], 0);
      bool hit = flip_exiting ^ (dir.Dot(normal) < 0);
      // First solution giving a valid hit wins
      if (hit) {
        if (distance < -vecgeom::kTolerance && distance < -fCylData->Radius()) {
          Real_t rho = point.Perp();
          safety     = fCylData->Radius() - rho;
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
  /// @param compute_onsurf Instructs to compute the projection of the point on surface
  /// @param onsurf Projection of the point on surface
  /// @return Validity of the calculation
  bool Safety(Vector3D<Real_t> const &point, bool left_side, Real_t &distance, Vector3D<Real_t> &onsurf) const
  {
    Real_t cylR       = fCylData->Radius();
    Real_t rho        = point.Perp();
    bool flip_exiting = left_side ^ fCylData->IsFlipped();
    distance          = flip_exiting ? cylR - rho : rho - cylR;
    onsurf            = point; // we only need the z of the projected point
    return true;
  }
};

} // namespace vgbrep

#endif
