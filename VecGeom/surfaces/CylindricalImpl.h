#ifndef VECGEOM_CYLINDRICAL_IMPL_H
#define VECGEOM_CYLINDRICAL_IMPL_H

#include <VecGeom/surfaces/SurfaceHelper.h>
#include <VecGeom/surfaces/Equations.h>

namespace vgbrep {

template <typename Real_t>
struct SurfaceHelper<kCylindrical, Real_t> {
  CylData<Real_t> const *fCylData{nullptr};

  VECGEOM_FORCE_INLINE
  SurfaceHelper(CylData<Real_t> const &cyldata) { fCylData = &cyldata; }

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
    flip_exiting ^= fCylData->IsFlipped();
    CylinderEq<Real_t>(point, dir, fCylData->Radius(), coef);
    QuadraticSolver(coef, roots, numroots);
    for (auto i = 0; i < numroots; ++i) {
      distance                = roots[i];
      Vector3D<Real_t> onsurf = point + distance * dir;
      Vector3D<Real_t> normal(onsurf[0], onsurf[1], 0);
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
    Real_t cylR = fCylData->Radius();
    Real_t rho  = point.Perp();
    distance    = flip_exiting ? cylR - rho : rho - cylR;
    // Cannot project if the point is on the center of the cylinder
    if (distance > -vecgeom::kTolerance && compute_onsurf) {
      onsurf.Set(0, 0, 0);
      if (rho > vecgeom::kTolerance) {
        auto invrho = 1. / rho;
        onsurf.Set(point[0] * invrho, point[1] * invrho, point[2]);
      }
    }
    return true;
  }
};

} // namespace vgbrep

#endif