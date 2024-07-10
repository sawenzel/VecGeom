#ifndef VECGEOM_TORUS_IMPL_H
#define VECGEOM_TORUS_IMPL_H

#include <VecGeom/surfaces/surf/SurfaceHelper.h>
#include <VecGeom/surfaces/base/Equations.h>

namespace vgbrep {

template <typename Real_t>
struct SurfaceHelper<SurfaceType::kTorus, Real_t> {
  TorusData<Real_t> const *fTorusData{nullptr};

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  SurfaceHelper(TorusData<Real_t> const &torusdata) { fTorusData = &torusdata; }

  /// @brief Fills the 3D extent of the Arb4
  /// @param aMin Bottom extent corner
  /// @param aMax Top extent corner
  void Extent3D(Vector3D<Real_t> &aMin, Vector3D<Real_t> &aMax) const
  {
    auto rtor  = fTorusData->Radius();
    auto rtube = fTorusData->RadiusTube();
    aMin.Set(-rtor - rtube, -rtor - rtube, -rtube);
    aMax.Set(rtor + rtube, rtor + rtube, rtube);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /// @brief Inside half-space function
  /// @param point Point in local surface coordinates
  /// @return True if the point is behind the normal within kTolerance (surface is included)
  bool Inside(Vector3D<Real_t> const &point, bool flip = false)
  {
    if (!fTorusData->InsidePhi(point, flip)) return false;

    int flipsign = (flip ^ fTorusData->IsFlipped()) ? -1 : 1;
    bool flipped = fTorusData->IsFlipped() ? 1 : 0;
    Real_t rho   = Sqrt(point.x() * point.x() + point.y() * point.y());
    Real_t rTor  = fTorusData->Radius();
    Real_t rTube = fTorusData->RadiusTube();

    bool check1 = (rho - (rTor - rTube)) > -flipsign * vecgeom::kTolerance;
    bool check2 = (rho - (rTor + rTube)) < flipsign * vecgeom::kTolerance;
    bool check3 = Sqrt(point.z() * point.z() + (rho - rTor) * (rho - rTor)) - rTube < flipsign * vecgeom::kTolerance;

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
  bool Intersect(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, bool left_side, Real_t &distance,
                 bool &two_solutions, Real_t &safety)
  {

    // RESCALING, from now on RadTube is normalized to TorusRadius and TorusRadius = 1!
    Vector3D<Real_t> localpoint = point / fTorusData->Radius();
    Real_t RadTube_R0           = fTorusData->RadiusTube() / fTorusData->Radius();

    Real_t tubeDistance = 0;

    // if point is outside bounding tube of the torus, propagate to the bounding tube.
    if (!(SurfaceHelper<SurfaceType::kCylindrical, Real_t>(fTorusData->GetInnerCylData()).Inside(localpoint, false) &&
          SurfaceHelper<SurfaceType::kCylindrical, Real_t>(fTorusData->GetOuterCylData()).Inside(localpoint, false)) ||
        (Abs(localpoint[2]) > Abs(RadTube_R0 + vecgeom::kTolerance))) {

      Real_t tmp   = vecgeom::kInfLength;
      tubeDistance = -1; // ensure it returns if no hit
      // check upper plane
      // emulate transformation of upper surface
      localpoint[2] = point[2] / fTorusData->Radius() - RadTube_R0;
      Vector3D<Real_t> tmppoint;
      Real_t tmp_rho;
      if (SurfaceHelper<SurfaceType::kPlanar, Real_t>().Intersect(localpoint, dir, false, tmp, two_solutions, safety)) {
        tmppoint = point / fTorusData->Radius() + tmp * dir;
        tmp_rho  = Sqrt(tmppoint[0] * tmppoint[0] + tmppoint[1] * tmppoint[1]);
        if ((tmp > -vecgeom::kTolerance) && (tmp_rho - (1. - RadTube_R0) > -vecgeom::kTolerance) &&
            (tmp_rho - (1. + RadTube_R0) < vecgeom::kTolerance)) {
          tubeDistance = tmp;
        }
      }

      // check lower plane
      // emulate transformation of lower surface
      Vector3D<Real_t> localdir = {dir[0], dir[1], -dir[2]};
      localpoint[2]             = -point[2] / fTorusData->Radius() - RadTube_R0;
      if (SurfaceHelper<SurfaceType::kPlanar, Real_t>().Intersect(localpoint, localdir, false, tmp, two_solutions,
                                                                  safety)) {
        tmppoint = point / fTorusData->Radius() + tmp * dir;
        tmp_rho  = Sqrt(tmppoint[0] * tmppoint[0] + tmppoint[1] * tmppoint[1]);
        if ((tmp > -vecgeom::kTolerance) && (tmp_rho - (1. - RadTube_R0) > -vecgeom::kTolerance) &&
            (tmp_rho - (1. + RadTube_R0) < vecgeom::kTolerance) && (tmp > -vecgeom::kTolerance)) {
          if (tubeDistance > -vecgeom::kTolerance) {
            tubeDistance = Min(tubeDistance, tmp);
          } else {
            tubeDistance = tmp;
          }
        }
      }

      // check outer cylinder
      localpoint = point / fTorusData->Radius();
      if (SurfaceHelper<SurfaceType::kCylindrical, Real_t>(fTorusData->GetOuterCylData())
              .Intersect(localpoint, dir, false, tmp, two_solutions, safety)) {
        tmppoint = point / fTorusData->Radius() + tmp * dir;
        if (tmp > -vecgeom::kTolerance && (Abs(tmppoint[2]) < Abs(RadTube_R0 + vecgeom::kTolerance))) {
          if (tubeDistance > -vecgeom::kTolerance) {
            tubeDistance = Min(tubeDistance, tmp);
          } else {
            tubeDistance = tmp;
          }
        }
      }

      // check inner cylinder
      localpoint = point / fTorusData->Radius();
      if (SurfaceHelper<SurfaceType::kCylindrical, Real_t>(fTorusData->GetInnerCylData())
              .Intersect(localpoint, dir, false, tmp, two_solutions, safety)) {
        tmppoint = point / fTorusData->Radius() + tmp * dir;
        if (tmp > -vecgeom::kTolerance && (Abs(tmppoint[2]) < Abs(RadTube_R0 + vecgeom::kTolerance))) {
          if (tubeDistance > -vecgeom::kTolerance) {
            tubeDistance = Min(tubeDistance, tmp);
          } else {
            tubeDistance = tmp;
          }
        }
      }

      // bounding tube not hit, cannot hit torus either, return
      if (tubeDistance < -vecgeom::kTolerance) return false;

      // transform to hit point on bounding tube
      localpoint = point / fTorusData->Radius() + tubeDistance * dir;
    }

    QuarticCoef<Real_t> coef;
    Real_t roots[4]          = {vecgeom::InfinityLength<Real_t>(), vecgeom::InfinityLength<Real_t>(),
                                vecgeom::InfinityLength<Real_t>(), vecgeom::InfinityLength<Real_t>()};
    int numroots             = 0;
    Real_t s                 = vecgeom::InfinityLength<Real_t>();
    VECGEOM_CONST Real_t tol = 100. * vecgeom::kTolerance;

    bool flip_exiting = left_side ^ fTorusData->IsFlipped();
    TorusEq<Real_t>(localpoint, dir, 1, RadTube_R0, coef);

    // special condition
    if (Abs(dir[2]) < 1E-3 && Abs(localpoint[2]) < 0.1 * RadTube_R0) {
      Real_t r0        = 1 - Sqrt((RadTube_R0 - localpoint[2]) * (RadTube_R0 + localpoint[2]));
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
      r0    = 1 + Sqrt((RadTube_R0 - localpoint[2]) * (RadTube_R0 + localpoint[2]));
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
      if (distance < -100) continue;

      Vector3D<Real_t> onsurf = localpoint + distance * dir;

      // apply phi cut
      if (!fTorusData->InsidePhi(onsurf, false)) continue;

      // calculate normal
      // note: using the normal before the Newton refinement could be dangerous but has proven safe so far.
      Vector3D<Real_t> normal = onsurf;
      onsurf.z()              = 0.;
      onsurf.Normalize();
      // onsurf *= fTorusData->Radius();
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
          if (Abs(eps0) > 200) break;
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

        distance += tubeDistance;         // add distance to bounding tube (0 if inside)
        distance *= fTorusData->Radius(); // denormalize

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
  /// @param onsurf Projection of the point on surface
  /// @return Validity of the calculation
  bool Safety(Vector3D<Real_t> const &point, bool left_side, Real_t &distance, Vector3D<Real_t> &onsurf) const
  {
    if (!fTorusData->InsidePhi(point, false)) {
      // If the point is not in the phi range, there are other surfaces closer than this one
      distance = vecgeom::InfinityLength<Real_t>();
      return true;
    }
    Real_t rho   = Sqrt(point.x() * point.x() + point.y() * point.y());
    Real_t rTor  = fTorusData->Radius();
    Real_t rTube = fTorusData->RadiusTube();

    distance = left_side ? rTube - Sqrt(point.z() * point.z() + (rho - rTor) * (rho - rTor))
                         : Sqrt(point.z() * point.z() + (rho - rTor) * (rho - rTor)) - rTube;

    return true;
  }
};

} // namespace vgbrep

#endif
