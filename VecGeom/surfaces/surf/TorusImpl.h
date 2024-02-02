#ifndef VECGEOM_TORUS_IMPL_H
#define VECGEOM_TORUS_IMPL_H

#include <VecGeom/surfaces/surf/SurfaceHelper.h>
#include <VecGeom/surfaces/base/Equations.h>

namespace vgbrep {

template <typename Real_t>
struct SurfaceHelper<kTorus, Real_t> {
  TorusData<Real_t> const *fTorusData{nullptr};

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  SurfaceHelper(TorusData<Real_t> const &torusdata) { fTorusData = &torusdata; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /// @brief Inside half-space function
  /// @param point Point in local surface coordinates
  /// @return True if the point is behind the normal within kTolerance (surface is included)
  bool Inside(Vector3D<Real_t> const &point)
  {
    if (!fTorusData->InsidePhi(point)) return false;

    bool flipped = fTorusData->IsFlipped() ? 1 : 0;
    Real_t rho   = Sqrt(point.x() * point.x() + point.y() * point.y());
    Real_t rTor  = fTorusData->Radius();
    Real_t rTube = fTorusData->RadiusTube();

    bool check1 = (rho - (rTor - rTube)) > -vecgeom::kTolerance;
    bool check2 = (rho - (rTor + rTube)) < vecgeom::kTolerance;
    bool check3 = Sqrt(point.z() * point.z() + (rho - rTor) * (rho - rTor)) - rTube < vecgeom::kTolerance;

    if (!flipped) {
      return check1 && check2 && check3;
    } else {
      return !check1 || !check2 || !check3;
    }
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

    Vector3D<Real_t> localpoint = point;

    Real_t tubeDistance = 0;

    // if point is outside bounding tube of the torus, propagate to the bounding tube.
    if (!SurfaceHelper<kCylindrical, Real_t>(fTorusData->GetCylData()).Inside(point) &&
        (point[2] > fTorusData->RadiusTube() || point[2] < -fTorusData->RadiusTube())) {

      bool tubehit = false;
      localpoint[2] -= fTorusData->RadiusTube(); // -tubeR in z to emulate translation

      // check upper plane
      tubehit |= SurfaceHelper<kPlanar, Real_t>().Intersect(localpoint, dir, left_side, tubeDistance);

      // emulate transformation of flipped lower surface
      localpoint[2]             = -point[2] - fTorusData->RadiusTube();
      Vector3D<Real_t> localdir = {dir[0], dir[1], -dir[2]};

      // check lower plane
      Real_t tmp = vecgeom::kInfLength;
      tubehit |= SurfaceHelper<kPlanar, Real_t>().Intersect(localpoint, localdir, left_side, tmp);
      if (tmp > -vecgeom::kTolerance) {
        if (tubeDistance > -vecgeom::kTolerance) {
          tubeDistance = Min(tubeDistance, tmp);
        } else {
          tubeDistance = tmp;
        }
      }

      // check cylinder
      tubehit |= SurfaceHelper<kCylindrical, Real_t>(fTorusData->GetCylData()).Intersect(point, dir, left_side, tmp);
      if (tmp > -vecgeom::kTolerance) {
        if (tubeDistance > -vecgeom::kTolerance) {
          tubeDistance = Min(tubeDistance, tmp);
        } else {
          tubeDistance = tmp;
        }
      }

      // bounding tube not hit, cannot hit torus either, return
      if (tubeDistance < -vecgeom::kTolerance) return false;

      // transform to hit point on bounding tube
      localpoint = point + tubeDistance * dir;
    }

    QuarticCoef<Real_t> coef;
    Real_t roots[4]          = {vecgeom::kInfLength, vecgeom::kInfLength, vecgeom::kInfLength, vecgeom::kInfLength};
    int numroots             = 0;
    Real_t s                 = vecgeom::kInfLength;
    VECGEOM_CONST Real_t tol = 100. * vecgeom::kTolerance;

    bool flip_exiting = left_side ^ fTorusData->IsFlipped();
    TorusEq<Real_t>(localpoint, dir, fTorusData->Radius(), fTorusData->RadiusTube(), coef);

    // special condition
    if (Abs(dir[2]) < 1E-3 && Abs(localpoint[2]) < 0.1 * fTorusData->RadiusTube()) {
      Real_t r0 = fTorusData->Radius() -
                  Sqrt((fTorusData->RadiusTube() - localpoint[2]) * (fTorusData->RadiusTube() + localpoint[2]));
      Real_t invdirxy2 = 1. / (1 - dir.z() * dir.z());
      Real_t b0        = (localpoint[0] * dir[0] + localpoint[1] * dir[1]) * invdirxy2;
      Real_t c0        = (localpoint[0] * localpoint[0] + (localpoint[1] - r0) * (localpoint[1] + r0)) * invdirxy2;
      Real_t delta     = b0 * b0 - c0;
      if (delta > 0) {
        roots[numroots] = -b0 - Sqrt(delta);
        if (roots[numroots] > -tol) numroots++;
        roots[numroots] = -b0 + Sqrt(delta);
        if (roots[numroots] > -tol) numroots++;
      }
      r0 = fTorusData->Radius() +
           Sqrt((fTorusData->RadiusTube() - localpoint[2]) * (fTorusData->RadiusTube() + localpoint[2]));
      c0    = (localpoint[0] * localpoint[0] + (localpoint[1] - r0) * (localpoint[1] + r0)) * invdirxy2;
      delta = b0 * b0 - c0;
      if (delta > 0) {
        roots[numroots] = -b0 - Sqrt(delta);
        if (roots[numroots] > -tol) numroots++;
        roots[numroots] = -b0 + Sqrt(delta);
        if (roots[numroots] > -tol) numroots++;
      }
      if (numroots) {
        Sort4(roots);
      }
    } else {
      QuarticSolver(coef, roots, numroots);
    }

    // iterate over possible solutions to find the first valid one
    for (auto i = 0; i < numroots; ++i) {
      distance = roots[i];

      // discard obviously incorrect solutions.
      // Before Newton refinement is applied a macroscopic number of -10 is used
      if (distance < -10) continue;

      Vector3D<Real_t> onsurf = localpoint + distance * dir;

      // apply phi cut
      if (!fTorusData->InsidePhi(onsurf)) continue;

      // calculate normal
      // note: using the normal before the Newton refinement could be dangerous but has proven safe so far.
      Vector3D<Real_t> normal = onsurf;
      onsurf.z()              = 0.;
      onsurf.Normalize();
      onsurf *= fTorusData->Radius();
      normal -= onsurf;

      bool hit = flip_exiting ^ (dir.Dot(normal) < 0);
      // First solution giving a valid hit wins
      if (hit) {

        // refine solution with Newton iterations
        s            = roots[i];
        Real_t eps   = vecgeom::kInfLength;
        Real_t delta = s * s * s * s + coef.a * s * s * s + coef.b * s * s + coef.c * s + coef.d;
        Real_t eps0  = -delta / (4. * s * s * s + 3. * coef.a * s * s + 2. * coef.b * s + coef.c);
        int ntry     = 0;
        while (Abs(eps) > vecgeom::kTolerance) {
          if (Abs(eps0) > 100) break;
          s += eps0;
          if (Abs(s + eps0) < vecgeom::kTolerance) break;
          delta = s * s * s * s + coef.a * s * s * s + coef.b * s * s + coef.c * s + coef.d;
          eps   = -delta / (4. * s * s * s + 3. * coef.a * s * s + 2. * coef.b * s + coef.c);
          if (Abs(eps) >= Abs(eps0)) break;
          ntry++;
          // Avoid infinite recursion
          if (ntry > 100) break;
          eps0 = eps;
        }
        // discard this solution
        if (s < -tol) continue;
        // use more accurate solution
        distance = Max(Real_t(0.), s);

        distance += tubeDistance; // add distance to bounding tube (0 if inside)

        return true;
      } else {
        continue;
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
  bool Safety(Vector3D<Real_t> const &point, bool left_side, Real_t &distance, bool compute_onsurf,
              Vector3D<Real_t> &onsurf) const
  {
    Real_t rho   = Sqrt(point.x() * point.x() + point.y() * point.y());
    Real_t rTor  = fTorusData->Radius();
    Real_t rTube = fTorusData->RadiusTube();

    distance = left_side ? rTube - Sqrt(point.z() * point.z() + (rho - rTor) * (rho - rTor))
                         : Sqrt(point.z() * point.z() + (rho - rTor) * (rho - rTor)) - rTube;

    // Todo: take care of phi cut, this can be an underestimation

    return true;
  }
};

} // namespace vgbrep

#endif
