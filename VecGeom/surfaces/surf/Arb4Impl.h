#ifndef VECGEOM_ARB4_IMPL_H
#define VECGEOM_ARB4_IMPL_H

#include <VecGeom/surfaces/surf/SurfaceHelper.h>
#include <VecGeom/surfaces/base/Equations.h>

namespace vgbrep {

template <typename Real_t>
struct SurfaceHelper<SurfaceType::kArb4, Real_t> {
  Arb4Data<Real_t> const *fArb4Data{nullptr};

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  SurfaceHelper(Arb4Data<Real_t> const &arbdata) { fArb4Data = &arbdata; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /// @brief Inside half-space function
  /// @param point Point in local surface coordinates
  /// @param flip flipping the tolerance for inside (needed for negated booleans)
  /// @return True if the point is behind the normal within kTolerance (surface is included)
  bool Inside(Vector3D<Real_t> const &point, bool flip)
  {
    Real_t dz     = fArb4Data->halfH;
    Real_t dz_inv = fArb4Data->halfH_inv;

    // distance along the top-bottom connecting line to reach the z value of the point.
    Real_t cf = 0.5 * dz_inv * (dz - point.z());

    //  loop over edges connecting points i with i+4
    Real_t vertexX[2];
    Real_t vertexY[2];

    for (int i = 0; i < 2; i++) {
      // calculate x-y positions of vertex i at this z-height
      vertexX[i] = fArb4Data->verticesX[i + 2] + cf * fArb4Data->connecting_compX[i];
      vertexY[i] = fArb4Data->verticesY[i + 2] + cf * fArb4Data->connecting_compY[i];
    }

    Real_t dx = vertexX[1] - vertexX[0];
    Real_t dy = vertexY[1] - vertexY[0];
    Real_t px = point.x() - vertexX[0];
    Real_t py = point.y() - vertexY[0];

    int sign = !flip ? 1 : -1;
    return (dx * py - dy * px) < sign * vecgeom::kTolerance;
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
                 bool &two_solutions)
  {

    using Vector3D = vecgeom::Vector3D<Real_t>;

    Real_t dzp    = point[2] + fArb4Data->halfH;
    Real_t dz_inv = fArb4Data->halfH_inv;

    // positions in x and y at the connection between top and bottom points as a function of z
    Real_t xs1 = fArb4Data->verticesX[1] + fArb4Data->ftx1 * dzp;
    Real_t ys1 = fArb4Data->verticesY[1] + fArb4Data->fty1 * dzp;
    Real_t xs2 = fArb4Data->verticesX[0] + fArb4Data->ftx2 * dzp;
    Real_t ys2 = fArb4Data->verticesY[0] + fArb4Data->fty2 * dzp;
    Real_t dxs = xs2 - xs1;
    Real_t dys = ys2 - ys1;

    // calculate quadratic coefficients
    Real_t a = (fArb4Data->fDeltatx * dir[1] - fArb4Data->fDeltaty * dir[0] + fArb4Data->ft1crosst2 * dir[2]) * dir[2];
    Real_t b = dxs * dir[1] - dys * dir[0] +
               (fArb4Data->fDeltatx * point[1] - fArb4Data->fDeltaty * point[0] + fArb4Data->fty2 * xs1 -
                fArb4Data->fty1 * xs2 + fArb4Data->ftx1 * ys2 - fArb4Data->ftx2 * ys1) *
                   dir[2];
    Real_t c = dxs * point[1] - dys * point[0] + xs1 * ys2 - xs2 * ys1;

    QuadraticCoef<Real_t> coef;
    coef.phalf = Real_t(1.) * b / (Real_t(2.) * vecgeom::NonZero(a));
    coef.q     = c / a;

    Real_t roots[2];
    int numroots;
    QuadraticSolver(coef, roots, numroots);

    // validate hits with normal and frame check
    for (auto i = 0; i < numroots; ++i) {
      if (roots[i] < -vecgeom::kTolerance) continue;
      distance        = roots[i];
      Vector3D onsurf = point + distance * dir;
      if (Abs(onsurf.z()) > fArb4Data->halfH + vecgeom::kTolerance) continue;

      // calculate the normal
      // when calculating the normal, we also calculate the vertical ratio rz and the horizontal ratios r in x and y.
      // They can be immediately used as a framecheck, so that the frame does not need to be checked later again.
      Real_t r   = -1;
      Real_t rz  = 0.5 * dz_inv * (onsurf[2] + fArb4Data->halfH);
      Real_t num = (onsurf.x() - fArb4Data->verticesX[1]) - rz * (fArb4Data->verticesX[3] - fArb4Data->verticesX[1]);
      Real_t denom =
          (fArb4Data->verticesX[0] - fArb4Data->verticesX[1]) +
          rz * (fArb4Data->verticesX[2] - fArb4Data->verticesX[0] - fArb4Data->verticesX[3] + fArb4Data->verticesX[1]);
      if (Abs(denom) > 1e-6) {
        r = num / vecgeom::NonZero(denom);
        if (!((r >= -vecgeom::kTolerance) && (r <= 1. + vecgeom::kTolerance))) continue;
      }
      num = (onsurf.y() - fArb4Data->verticesY[1]) - rz * (fArb4Data->verticesY[3] - fArb4Data->verticesY[1]);
      denom =
          (fArb4Data->verticesY[0] - fArb4Data->verticesY[1]) +
          rz * (fArb4Data->verticesY[2] - fArb4Data->verticesY[0] - fArb4Data->verticesY[3] + fArb4Data->verticesY[1]);
      if (Abs(denom) > 1e-6) r = num / vecgeom::NonZero(denom);

      // frame check?
      if (!((r >= -vecgeom::kTolerance) && (r <= 1. + vecgeom::kTolerance))) continue;
      if (!((rz >= -vecgeom::kTolerance) && (rz <= 1. + vecgeom::kTolerance))) continue;

      // calculating normal
      Vector3D unorm = fArb4Data->fViCrossHi0 + rz * fArb4Data->fViCrossVj + r * fArb4Data->fHi1CrossHi0;

      bool hit = left_side ^ (dir.Dot(unorm) < 0);
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

    Real_t dzp = fArb4Data->halfH + point[2];

    // safety in z
    distance = (fabs(point[2]) > fArb4Data->halfH + vecgeom::kTolerance) ? fabs(point[2]) - fArb4Data->halfH : 0.;

#ifdef SURF_ACCURATE_SAFETY
    Real_t v_safety         = vecgeom::InfinityLength<Real_t>();
    Real_t tmp              = 0.;
    Vector3D<Real_t> vplane = {fArb4Data->verticesX[0], fArb4Data->verticesY[0], -fArb4Data->halfH};
    tmp                     = (point - vplane).Dot(fArb4Data->normal0);
    if (tmp > 0) v_safety = tmp;

    vplane = {fArb4Data->verticesX[1], fArb4Data->verticesY[1], -fArb4Data->halfH};
    tmp    = (point - vplane).Dot(fArb4Data->normal1);
    if (tmp > 0) v_safety = Min(v_safety, tmp);

    vplane = {fArb4Data->verticesX[2], fArb4Data->verticesY[2], fArb4Data->halfH};
    tmp    = (point - vplane).Dot(fArb4Data->normal2);
    if (tmp > 0) v_safety = Min(v_safety, tmp);

    vplane = {fArb4Data->verticesX[3], fArb4Data->verticesY[3], -fArb4Data->halfH};
    tmp    = (point - vplane).Dot(fArb4Data->normal3);
    if (tmp > 0) v_safety = Min(v_safety, tmp);

    if (v_safety < vecgeom::InfinityLength<Real_t>()) {
      distance = Max(distance, v_safety);
    }

#endif

    Real_t xs1 = fArb4Data->verticesX[1] + fArb4Data->ftx1 * dzp;
    Real_t ys1 = fArb4Data->verticesY[1] + fArb4Data->fty1 * dzp;
    Real_t xs2 = fArb4Data->verticesX[0] + fArb4Data->ftx2 * dzp;
    Real_t ys2 = fArb4Data->verticesY[0] + fArb4Data->fty2 * dzp;
    Real_t dxs = xs2 - xs1;
    Real_t dys = ys2 - ys1;

    Real_t umin   = 0.;
    Real_t safety = vecgeom::InfinityLength<Real_t>();

    Real_t dx1 = 0.;
    Real_t dx2 = 0.;
    Real_t dy1 = 0.;
    Real_t dy2 = 0.;

    Real_t dpx = point[0] - xs1;
    Real_t dpy = point[1] - ys1;
    Real_t lsq = dxs * dxs + dys * dys;
    Real_t u   = (dpx * dxs + dpy * dys) / vecgeom::NonZero(lsq);
    if (u > 1 + vecgeom::kTolerance) {
      dpx = point[0] - xs2;
      dpy = point[1] - ys2;
    }
    if (u >= -vecgeom::kTolerance && u <= 1 + vecgeom::kTolerance) {
      dpx = dpx - u * dxs;
      dpy = dpy - u * dys;
    }
    Real_t ssq = dpx * dpx + dpy * dpy; // safety squared
    if (ssq < safety) {
      dx1    = fArb4Data->verticesX[0] - fArb4Data->verticesX[1];
      dx2    = fArb4Data->verticesX[2] - fArb4Data->verticesX[3];
      dy1    = fArb4Data->verticesY[0] - fArb4Data->verticesY[1];
      dy2    = fArb4Data->verticesY[2] - fArb4Data->verticesY[3];
      umin   = u;
      safety = ssq;
    }
    if (umin < -vecgeom::kTolerance || umin > 1 + vecgeom::kTolerance) umin = 0.;
    dxs = dx1 + umin * (dx2 - dx1);
    dys = dy1 + umin * (dy2 - dy1);
    // Denominator below always positive as fDz>0
    safety *= Real_t(1.) - Real_t(4.) * fArb4Data->halfH * fArb4Data->halfH /
                               (dxs * dxs + dys * dys + Real_t(4.) * fArb4Data->halfH * fArb4Data->halfH);
    distance = Max(distance, Sqrt(safety));

    return true;
  }
};

} // namespace vgbrep

#endif
