#ifndef VECGEOM_ELLIPTICAL_IMPL_H
#define VECGEOM_ELLIPTICAL_IMPL_H

#include <VecGeom/surfaces/surf/SurfaceHelper.h>
#include <VecGeom/surfaces/base/Equations.h>

namespace vgbrep {

template <typename Real_t>
struct SurfaceHelper<SurfaceType::kElliptical, Real_t> {
  EllipData<Real_t> const *fEllipData{nullptr};

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  SurfaceHelper(EllipData<Real_t> const &ellipdata) { fEllipData = &ellipdata; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /// @brief Inside half-space function
  /// @param point Point in local surface coordinates
  /// @return True if the point is behind the normal within kTolerance (surface is included)
  bool Inside(Vector3D<Real_t> const &point, bool flip)
  {
    int flipsign = !flip ? 1 : -1;
    return (point.x() * point.x() / fEllipData->fDDx + point.y() * point.y() / fEllipData->fDDy <
            static_cast<Real_t>(1) + flipsign * vecgeom::kToleranceStrict<Real_t>);
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

    distance = vecgeom::InfinityLength<Real_t>();

    Vector3D<Real_t> pcur(point);

    // MISSING: optimsation to move point closer
    // Real_t offset(0.);
    // // Move point closer, if required
    // Real_v Rfar2(1024. * ellipticaltube.fRsph * ellipticaltube.fRsph); // 1024 = 32 * 32
    // vecCore__MaskedAssignFunc(pcur, ((pcur.Mag2() > Rfar2) && (direction.Dot(point) < Real_v(0.))),
    //                           pcur + (offset = pcur.Mag() - Real_v(2.) * ellipticaltube.fRsph) * direction);

    // Scale elliptical tube to cylinder
    Real_t px = pcur.x() * fEllipData->fSx;
    Real_t py = pcur.y() * fEllipData->fSy;
    Real_t vx = dir.x() * fEllipData->fSx;
    Real_t vy = dir.y() * fEllipData->fSy;

    // Find intersection with lateral surface, solve equation: A t^2 + 2B t + C = 0
    Real_t rr = px * px + py * py;
    Real_t A  = vx * vx + vy * vy;
    Real_t B  = px * vx + py * vy;
    Real_t C  = rr - fEllipData->R * fEllipData->R;

    QuadraticCoef<Real_t> coef;
    coef.phalf = B / vecgeom::NonZero(A);
    coef.q     = C / vecgeom::NonZero(A);

    Real_t roots[2];
    int numroots;
    QuadraticSolver(coef, roots, numroots);
    two_solutions = (numroots == 2 && roots[0] > -vecgeom::kToleranceStrict<Real_t> &&
                     roots[1] > -vecgeom::kToleranceStrict<Real_t>);
    for (auto i = 0; i < numroots; ++i) {
      distance                = roots[i];
      Vector3D<Real_t> onsurf = point + distance * dir;
      Vector3D<Real_t> normal(onsurf[0] * fEllipData->fDDy, onsurf[1] * fEllipData->fDDx, 0);
      bool hit = left_side ^ (dir.Dot(normal) < 0);
      // First solution giving a valid hit wins
      if (hit) {
        if (distance < -vecgeom::kToleranceStrict<Real_t>) {
          Real_t x   = point.x() * fEllipData->fSx;
          Real_t y   = point.y() * fEllipData->fSy;
          Real_t rho = vecCore::math::Sqrt(x * x + y * y);
          safety     = fEllipData->R - rho;
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
    Real_t x   = point.x() * fEllipData->fSx;
    Real_t y   = point.y() * fEllipData->fSy;
    Real_t rho = vecCore::math::Sqrt(x * x + y * y);
    distance   = left_side ? fEllipData->R - rho : rho - fEllipData->R;
    onsurf     = point; // we only need the z of the projected point
    return true;
  }
};

} // namespace vgbrep

#endif
