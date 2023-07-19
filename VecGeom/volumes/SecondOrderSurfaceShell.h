/// \file SecondOrderSurfaceShell.h
/// \author: swenzel
///  Created on: Aug 1, 2014
///  Modified and completed: mihaela.gheata@cern.ch

#ifndef VECGEOM_SECONDORDERSURFACESHELL_H_
#define VECGEOM_SECONDORDERSURFACESHELL_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"

namespace vecgeom {
/**
 * Templated class providing a (SOA) encapsulation of
 * second order curved surfaces used in generic trap
 */
VECGEOM_DEVICE_FORWARD_DECLARE(class SecondOrderSurfaceShell;);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <int N>
class SecondOrderSurfaceShell {

  using Vertex_t = Vector3D<Precision>;

private:
  // caching some important values for each of the curved planes
  Precision fxa[N], fya[N], fxb[N], fyb[N], fxc[N], fyc[N], fxd[N], fyd[N]; /** Coordinates of vertices */
  Precision ftx1[N], fty1[N], ftx2[N], fty2[N];                             /** Connecting components */
  Precision ft1crosst2[N]; /** Cross term ftx1[i]*fty2[i] - ftx2[i]*fty1[i] */
  Precision fDeltatx[N];   /** Term ftx2[i] - ftx1[i] */
  Precision fDeltaty[N];   /** Term fty2[i] - fty1[i] */

  Precision fDz;          /** height of surface (coming from the height of the GenTrap) */
  Precision fDz2;         /** 0.5/fDz */
  Precision fiscurved[N]; /** Indicate which surfaces are planar */
  bool fisplanar;         /** Flag that all surfaces are planar */
  bool fdegenerated[N];   /** Flags for each top-bottom edge marking that this is degenerated */

  Vertex_t fNormals[N]; /** Pre-computed normals for the planar case */

  // pre-computed cross products for normal computation
  Vertex_t fViCrossHi0[N];  /** Pre-computed vi X hi0 */
  Vertex_t fViCrossVj[N];   /** Pre-computed vi X vj */
  Vertex_t fHi1CrossHi0[N]; /** Pre-computed hi1 X hi0 */

public:
  /** @brief SecondOrderSurfaceShell dummy constructor */
  VECCORE_ATT_HOST_DEVICE
  SecondOrderSurfaceShell() : fDz(0), fDz2(0) {}

  /** @brief SecondOrderSurfaceShell constructor
   * @param verticesx X positions of vertices in array form
   * @param verticesy Y positions of vertices in array form
   * @param dz The half-height of the GenTrap
   */
  VECCORE_ATT_HOST_DEVICE
  SecondOrderSurfaceShell(const Precision *verticesx, const Precision *verticesy, Precision dz) : fDz(0.), fDz2(0.)
  {
    // Constructor
    Initialize(verticesx, verticesy, dz);
  }

  VECCORE_ATT_HOST_DEVICE
  void Initialize(const Precision *verticesx, const Precision *verticesy, Precision dz)
  {
#ifndef VECCORE_CUDA
    if (dz <= 0.) throw std::runtime_error("Half-length of generic trapezoid must be positive");
#endif
    fDz  = dz;
    fDz2 = 0.5 / dz;
    // Store vertex coordinates
    Vertex_t va, vb, vc, vd;
    for (int i = 0; i < N; ++i) {
      int j = (i + 1) % N;
      va.Set(verticesx[i], verticesy[i], -dz);
      fxa[i] = verticesx[i];
      fya[i] = verticesy[i];
      vb.Set(verticesx[i + N], verticesy[i + N], dz);
      fxb[i] = verticesx[i + N];
      fyb[i] = verticesy[i + N];
      vc.Set(verticesx[j], verticesy[j], -dz);
      fxc[i] = verticesx[j];
      fyc[i] = verticesy[j];
      vd.Set(verticesx[j + N], verticesy[j + N], dz);
      fxd[i]          = verticesx[N + j];
      fyd[i]          = verticesy[N + j];
      fdegenerated[i] = (Abs(fxa[i] - fxc[i]) < kTolerance) && (Abs(fya[i] - fyc[i]) < kTolerance) &&
                        (Abs(fxb[i] - fxd[i]) < kTolerance) && (Abs(fyb[i] - fyd[i]) < kTolerance);
      ftx1[i] = fDz2 * (fxb[i] - fxa[i]);
      fty1[i] = fDz2 * (fyb[i] - fya[i]);
      ftx2[i] = fDz2 * (fxd[i] - fxc[i]);
      fty2[i] = fDz2 * (fyd[i] - fyc[i]);

      ft1crosst2[i] = ftx1[i] * fty2[i] - ftx2[i] * fty1[i];
      fDeltatx[i]   = ftx2[i] - ftx1[i];
      fDeltaty[i]   = fty2[i] - fty1[i];
      fNormals[i]   = Vertex_t::Cross(vb - va, vc - va);
      // The computation of normals is done also for the curved surfaces, even if they will not be used
      if (fNormals[i].Mag2() < kTolerance) {
        // points i and i+1/N are overlapping - use i+N and j+N instead
        fNormals[i] = Vertex_t::Cross(vb - va, vd - vb);
        if (fNormals[i].Mag2() < kTolerance) fNormals[i].Set(0., 0., 1.); // No surface, just a line
      }
      fNormals[i].Normalize();
      // Cross products used for normal computation
      fViCrossHi0[i]  = Vertex_t::Cross(vb - va, vc - va);
      fViCrossVj[i]   = Vertex_t::Cross(vb - va, vd - vc);
      fHi1CrossHi0[i] = Vertex_t::Cross(vd - vb, vc - va);
    }

    // Analyze planarity and precompute normals
    fisplanar = true;
    for (int i = 0; i < N; ++i) {
      fiscurved[i] = (((Abs(fxc[i] - fxa[i]) < kTolerance) && (Abs(fyc[i] - fya[i]) < kTolerance)) ||
                      ((Abs(fxd[i] - fxb[i]) < kTolerance) && (Abs(fyd[i] - fyb[i]) < kTolerance)) ||
                      (Abs((fxc[i] - fxa[i]) * (fyd[i] - fyb[i]) - (fxd[i] - fxb[i]) * (fyc[i] - fya[i])) < kTolerance))
                         ? 0
                         : 1;
      if (fiscurved[i]) fisplanar = false;
    }
  }

  //______________________________________________________________________________
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /** @brief Stripped down version of Inside method used for boundary and wrong side detection
   * @param point Starting point in the local frame
   * @param completelyinside Inside flag
   * @param completelyoutside Onside flag
   * @param onsurf On boundary flag
   */
  void CheckInside(Vector3D<Real_v> const &point, vecCore::Mask_v<Real_v> &completelyinside,
                   vecCore::Mask_v<Real_v> &completelyoutside, vecCore::Mask_v<Real_v> &onsurf) const
  {

    using Bool_v                        = vecCore::Mask_v<Real_v>;
    VECGEOM_CONST Precision tolerancesq = 10000. * kTolerance * kTolerance;

    onsurf            = Bool_v(false);
    completelyinside  = (Abs(point.z()) < Real_v(MakeMinusTolerant<true>(fDz)));
    completelyoutside = (Abs(point.z()) > Real_v(MakePlusTolerant<true>(fDz)));
    //  if (vecCore::EarlyReturnAllowed()) {
    if (vecCore::MaskFull(completelyoutside)) return;
    //  }

    Real_v cross;
    Real_v vertexX[N];
    Real_v vertexY[N];
    Real_v dzp = fDz + point[2];
    // vectorizes for scalar backend
    for (int i = 0; i < N; i++) {
      // calculate x-y positions of vertex i at this z-height
      vertexX[i] = fxa[i] + ftx1[i] * dzp;
      vertexY[i] = fya[i] + fty1[i] * dzp;
    }
    Bool_v degenerated = Bool_v(true); // Can only happen if |zpoint| = fDz
    for (int i = 0; i < N; i++) {
      //      if (fdegenerated[i])
      //        continue;
      int j          = (i + 1) % 4;
      Real_v DeltaX  = vertexX[j] - vertexX[i];
      Real_v DeltaY  = vertexY[j] - vertexY[i];
      Real_v deltasq = DeltaX * DeltaX + DeltaY * DeltaY;
      // If the current vertex is degenerated, ignore the check
      Bool_v samevertex = deltasq < Real_v(MakePlusTolerant<true>(0.));
      degenerated       = degenerated && samevertex;
      // Cross product to check if point is right side or left side
      // If vertices are same, this will be 0
      cross = (point.x() - vertexX[i]) * DeltaY - (point.y() - vertexY[i]) * DeltaX;

      onsurf            = (cross * cross < tolerancesq * deltasq) && (!samevertex);
      completelyoutside = completelyoutside || (((cross < Real_v(MakeMinusTolerant<true>(0.))) && (!onsurf)));
      completelyinside =
          completelyinside && (samevertex || ((cross > Real_v(MakePlusTolerant<true>(0.))) && (!onsurf)));
    }
    onsurf = (!completelyoutside) && (!completelyinside);
    // In fully degenerated case consider the point outside always
    completelyoutside = completelyoutside || degenerated;
  }

  //______________________________________________________________________________
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /** @brief Compute distance to a set of curved/planar surfaces
   * @param point Starting point in the local frame
   * @param dir Direction in the local frame
   */
  Real_v DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {

    using Bool_v = vecCore::Mask_v<Real_v>;

    // Planar case
    if (fisplanar) return DistanceToOutPlanar<Real_v>(point, dir);

    // The algorithmic tolerance in distance
    const Real_v tolerance(100. * kTolerance);

    Real_v dist(InfinityLength<Real_v>());
    Real_v smin[N], smax[N];
    Vector3D<Real_v> unorm;
    Real_v r(-1.0);
    Real_v rz;
    Bool_v completelyinside, completelyoutside, onsurf;
    CheckInside<Real_v>(point, completelyinside, completelyoutside, onsurf);

    // If on the wrong side, return -1.
    Real_v wrongsidedist(-1.0);
    vecCore::MaskedAssign(dist, completelyoutside, wrongsidedist);
    if (vecCore::MaskFull(completelyoutside)) return dist;

    // Solve the second order equation and return distance solutions for each surface
    ComputeSminSmax<Real_v>(point, dir, smin, smax);
    for (int i = 0; i < N; ++i) {
      // Check if point(s) is(are) on boundary, and in this case compute normal
      Bool_v crtbound = (Abs(smin[i]) < tolerance || Abs(smax[i]) < tolerance);
      if (!vecCore::MaskEmpty(crtbound)) {
        if (fiscurved[i])
          UNormal<Real_v>(point, i, unorm, rz, r);
        else
          unorm = fNormals[i];
      }
      // Starting point may be propagated close to boundary
      // === MaskedMultipleAssign needed
      vecCore__MaskedAssignFunc(smin[i], !completelyoutside && Abs(smin[i]) < tolerance && dir.Dot(unorm) < 0,
                                InfinityLength<Real_v>());
      vecCore__MaskedAssignFunc(smax[i], !completelyoutside && Abs(smax[i]) < tolerance && dir.Dot(unorm) < 0,
                                InfinityLength<Real_v>());

      vecCore__MaskedAssignFunc(dist, !completelyoutside && (smin[i] > -tolerance) && (smin[i] < dist),
                                Max(smin[i], Real_v(0.)));
      vecCore__MaskedAssignFunc(dist, !completelyoutside && (smax[i] > -tolerance) && (smax[i] < dist),
                                Max(smax[i], Real_v(0.)));
    }
    vecCore__MaskedAssignFunc(dist, dist < tolerance && onsurf, Real_v(0.));
    return (dist);
  } // end of function

  //______________________________________________________________________________
  /** @brief Compute distance to exiting the set of surfaces in the planar case.
   * @param point Starting point in the local frame
   * @param dir Direction in the local frame
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Real_v DistanceToOutPlanar(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {

    using Bool_v = vecCore::Mask_v<Real_v>;
    //    const Real_v tolerance = 100. * kTolerance;

    Vertex_t va;         // vertex i of lower base
    Vector3D<Real_v> pa; // same vertex converted to backend type
    Real_v distance = InfinityLength<Real_v>();

    // Check every surface
    Bool_v outside = (Abs(point.z()) > MakePlusTolerant<true>(fDz)); // If point is outside, we need to know
    for (int i = 0; i < N && (!vecCore::MaskFull(outside)); ++i) {
      if (fdegenerated[i]) continue;
      // Point A is the current vertex on lower Z. P is the point we come from.
      pa.Set(Real_v(fxa[i]), Real_v(fya[i]), Real_v(-fDz));
      Vector3D<Real_v> vecAP = point - pa;
      // Dot product between AP vector and normal to surface has to be negative
      Real_v dotAPNorm = vecAP.Dot(fNormals[i]);
      Bool_v otherside = (dotAPNorm > Real_v(10.) * kTolerance);
      // Dot product between direction and normal to surface has to be positive
      Real_v dotDirNorm = dir.Dot(fNormals[i]);
      Bool_v outgoing   = (dotDirNorm > Real_v(0.));
      // Update globally outside flag
      outside      = outside || otherside;
      Bool_v valid = outgoing && (!otherside);
      if (vecCore::MaskEmpty(valid)) continue;
      Real_v snext = -dotAPNorm / NonZero(dotDirNorm);
      vecCore__MaskedAssignFunc(distance, valid && snext < distance, Max(snext, Real_v(0.)));
    }
    // Return -1 for points actually outside
    vecCore__MaskedAssignFunc(distance, distance < kTolerance, Real_v(0.));
    vecCore__MaskedAssignFunc(distance, outside, Real_v(-1.));
    return distance;
  }

  //______________________________________________________________________________
  /** @brief Compute distance to entering the set of surfaces in the planar case.
   * @param point Starting point in the local frame
   * @param dir Direction in the local frame
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Real_v DistanceToInPlanar(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir,
                            vecCore::Mask_v<Real_v> &done) const
  {
    //    const Real_v tolerance = 100. * kTolerance;

    using Bool_v = vecCore::Mask_v<Real_v>;
    Vertex_t va;         // vertex i of lower base
    Vector3D<Real_v> pa; // same vertex converted to backend type
    Real_v distance = InfinityLength<Real_v>();

    // Check every surface
    Bool_v inside = (Abs(point.z()) < MakeMinusTolerant<true>(fDz)); // If point is inside, we need to know
    for (int i = 0; i < N; ++i) {
      if (fdegenerated[i]) continue;
      // Point A is the current vertex on lower Z. P is the point we come from.
      pa.Set(Real_v(fxa[i]), Real_v(fya[i]), Real_v(-fDz));
      Vector3D<Real_v> vecAP = point - pa;
      // Dot product between AP vector and normal to surface has to be positive
      Real_v dotAPNorm = vecAP.Dot(fNormals[i]);
      Bool_v otherside = (dotAPNorm < Real_v(-10.) * kTolerance);
      // Dot product between direction and normal to surface has to be negative
      Real_v dotDirNorm = dir.Dot(fNormals[i]);
      Bool_v ingoing    = (dotDirNorm < Real_v(0.));
      // Update globally outside flag
      inside       = inside && otherside;
      Bool_v valid = ingoing && (!otherside);
      if (vecCore::MaskEmpty(valid)) continue;
      Real_v snext = -dotAPNorm / NonZero(dotDirNorm);
      // Now propagate the point to surface and check if in range
      Vector3D<Real_v> psurf = point + snext * dir;
      valid                  = valid && InSurfLimits<Real_v>(psurf, i);
      vecCore__MaskedAssignFunc(distance, (!done) && valid && snext < distance, Max(snext, Real_v(0.)));
    }
    // Return -1 for points actually inside
    vecCore__MaskedAssignFunc(distance, (!done) && (distance < kTolerance), Real_v(0.));
    vecCore__MaskedAssignFunc(distance, (!done) && inside, Real_v(-1.));
    return distance;
  }

  //______________________________________________________________________________
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /**
   * A generic function calculation the distance to a set of curved/planar surfaces
   *
   * things to improve: error handling for boundary cases
   *
   * another possibility relies on the following idea:
   * we always have an even number of planar/curved surfaces. We could organize them in separate substructures...
   */
  Real_v DistanceToIn(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir, vecCore::Mask_v<Real_v> &done) const
  {
    // Planar case
    using Bool_v = vecCore::Mask_v<Real_v>;
    if (fisplanar) return DistanceToInPlanar<Real_v>(point, dir, done);
    Real_v crtdist;
    Vector3D<Real_v> hit;
    Real_v distance = InfinityLength<Real_v>();
    Real_v tolerance(100. * kTolerance);
    Vector3D<Real_v> unorm;
    Real_v r(-1.0);
    Real_v rz;
    Bool_v completelyinside, completelyoutside, onsurf;
    CheckInside<Real_v>(point, completelyinside, completelyoutside, onsurf);
    // If on the wrong side, return -1.
    Real_v wrongsidedist(-1.0);
    vecCore::MaskedAssign(distance, Bool_v(completelyinside && (!done)), wrongsidedist);
    Bool_v checked = completelyinside || done;
    if (vecCore::MaskFull(checked)) return (distance);
    // Now solve the second degree equation to find crossings
    Real_v smin[N], smax[N];
    ComputeSminSmax<Real_v>(point, dir, smin, smax);

    // now we need to analyse which of those distances is good
    // does not vectorize
    for (int i = 0; i < N; ++i) {
      crtdist = smin[i];
      // Extrapolate with hit distance candidate
      hit             = point + crtdist * dir;
      Bool_v crossing = (crtdist > -tolerance) && (Abs(hit.z()) < fDz + kTolerance);
      // Early skip surface if not in Z range
      if (!vecCore::MaskEmpty(Bool_v(crossing && (!checked)))) {
        // Compute local un-normalized outwards normal direction and hit ratio factors
        UNormal<Real_v>(hit, i, unorm, rz, r);
        // Distance have to be positive within tolerance, and crossing must be inwards
        crossing = crossing && dir.Dot(unorm) < Real_v(0.);
        // Propagated hitpoint must be on surface (rz in [0,1] checked already)
        crossing = crossing && (r >= Real_v(0.)) && (r <= Real_v(1.));
        vecCore__MaskedAssignFunc(distance, crossing && (!checked) && crtdist < distance, Max(crtdist, Real_v(0.)));
      }
      // For the particle(s) not crossing at smin, try smax
      if (!vecCore::MaskFull(Bool_v(crossing || checked))) {
        // Treat only particles not crossing at smin
        crossing = !crossing;
        crtdist  = smax[i];
        hit      = point + crtdist * dir;
        crossing = crossing && (crtdist > -tolerance) && (Abs(hit.z()) < fDz + kTolerance);
        if (vecCore::MaskEmpty(crossing)) continue;
        if (fiscurved[i])
          UNormal<Real_v>(hit, i, unorm, rz, r);
        else
          unorm = fNormals[i];
        crossing = crossing && (dir.Dot(unorm) < Real_v(0.));
        crossing = crossing && (r >= Real_v(0.)) && (r <= Real_v(1.));
        vecCore__MaskedAssignFunc(distance, crossing && (!checked) && crtdist < distance, Max(crtdist, Real_v(0.)));
      }
    }
    vecCore__MaskedAssignFunc(distance, distance < tolerance && onsurf, Real_v(0.));
    return (distance);

  } // end distanceToIn function

  //______________________________________________________________________________
  /**
   * @brief A generic function calculation for the safety to a set of curved/planar surfaces.
           Should be smaller than safmax
   * @param point Starting point inside, in the local frame
   * @param safmax current safety value
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Real_v SafetyToOut(Vector3D<Real_v> const &point, Real_v const &safmax) const
  {

    using Bool_v                = vecCore::Mask_v<Real_v>;
    VECGEOM_CONST Precision eps = 100. * kTolerance;

    Real_v safety = safmax;
    Bool_v done   = (Abs(safety) < eps);
    if (vecCore::MaskFull(done)) return (safety);
    Real_v safetyface = InfinityLength<Real_v>();

    // loop lateral surfaces
    // We can use the surface normals to get safety for non-curved surfaces
    Vertex_t va;         // vertex i of lower base
    Vector3D<Real_v> pa; // same vertex converted to backend type
    if (fisplanar) {
      for (int i = 0; i < N; ++i) {
        if (fdegenerated[i]) continue;
        va.Set(fxa[i], fya[i], -fDz);
        pa         = va;
        safetyface = (pa - point).Dot(fNormals[i]);
        vecCore::MaskedAssign(safety, (safetyface < safety) && (!done), safetyface);
      }
      vecCore__MaskedAssignFunc(safety, Abs(safety) < eps, Real_v(0.));
      return safety;
    }

    // Not fully planar - use mixed case
    safetyface = SafetyCurved<Real_v>(point, Bool_v(true));
    //  std::cerr << "safetycurved = " << safetyface << std::endl;
    vecCore::MaskedAssign(safety, (safetyface < safety) && (!done), safetyface);
    //  std::cerr << "safety = " << safety << std::endl;
    vecCore__MaskedAssignFunc(safety, safety > Real_v(0.) && safety < eps, Real_v(0.));
    return safety;

  } // end SafetyToOut

  //______________________________________________________________________________
  /**
   * @brief A generic function calculation for the safety to a set of curved/planar surfaces.
           Should be smaller than safmax
   * @param point Starting point outside, in the local frame
   * @param safmax current safety value
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Real_v SafetyToIn(Vector3D<Real_v> const &point, Real_v const &safmax) const
  {

    using Bool_v                = vecCore::Mask_v<Real_v>;
    VECGEOM_CONST Precision eps = 100. * kTolerance;

    Real_v safety = safmax;
    Bool_v done   = (Abs(safety) < eps);
    if (vecCore::MaskFull(done)) return (safety);
    Real_v safetyface = InfinityLength<Real_v>();

    // loop lateral surfaces
    // We can use the surface normals to get safety for non-curved surfaces
    Vertex_t va;         // vertex i of lower base
    Vector3D<Real_v> pa; // same vertex converted to backend type
    if (fisplanar) {
      for (int i = 0; i < N; ++i) {
        if (fdegenerated[i]) continue;
        va.Set(fxa[i], fya[i], -fDz);
        pa         = va;
        safetyface = (point - pa).Dot(fNormals[i]);
        vecCore::MaskedAssign(safety, (safetyface > safety) && (!done), safetyface);
      }
      vecCore__MaskedAssignFunc(safety, Abs(safety) < eps, Real_v(0.));
      return safety;
    }

    // Not fully planar - use mixed case
    safetyface = SafetyCurved<Real_v>(point, Bool_v(false));
    //  std::cerr << "safetycurved = " << safetyface << std::endl;
    vecCore::MaskedAssign(safety, (safetyface > safety) && (!done), safetyface);
    //  std::cerr << "safety = " << safety << std::endl;
    vecCore__MaskedAssignFunc(safety, safety > Real_v(0.) && safety < eps, Real_v(0.));
    return (safety);

  } // end SafetyToIn

  //______________________________________________________________________________
  /**
   * @brief A generic function calculation for the safety to a set of curved/planar surfaces.
           Should be smaller than safmax
   * @param point Starting point
   * @param in Inside value for starting point
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Real_v SafetyCurved(Vector3D<Real_v> const &point, vecCore::Mask_v<Real_v> in) const
  {
    using Bool_v     = vecCore::Mask_v<Real_v>;
    Real_v safety    = InfinityLength<Real_v>();
    Real_v safplanar = InfinityLength<Real_v>();
    Real_v tolerance = Real_v(100 * kTolerance);
    vecCore__MaskedAssignFunc(tolerance, !in, -tolerance);

    //  loop over edges connecting points i with i+4
    Real_v dx, dy, dpx, dpy, lsq, u;
    Real_v dx1 = Real_v(0.);
    Real_v dx2 = Real_v(0.);
    Real_v dy1 = Real_v(0.);
    Real_v dy2 = Real_v(0.);

    Bool_v completelyinside, completelyoutside, onsurf;
    CheckInside<Real_v>(point, completelyinside, completelyoutside, onsurf);
    if (vecCore::MaskFull(onsurf)) return (Real_v(0.));

    Bool_v wrong = in && (completelyoutside);
    wrong        = wrong || ((!in) && completelyinside);
    if (vecCore::MaskFull(wrong)) {
      return (Real_v(-1.));
    }
    Real_v vertexX[N];
    Real_v vertexY[N];
    Real_v dzp = fDz + point[2];
    for (int i = 0; i < N; i++) {
      // calculate x-y positions of vertex i at this z-height
      vertexX[i] = fxa[i] + ftx1[i] * dzp;
      vertexY[i] = fya[i] + fty1[i] * dzp;
    }
    Real_v umin = Real_v(0.);
    for (int i = 0; i < N; i++) {
      if (fiscurved[i] == 0) {
        // We can use the surface normals to get safety for non-curved surfaces
        Vector3D<Real_v> pa; // same vertex converted to backend type
        pa.Set(Real_v(fxa[i]), Real_v(fya[i]), Real_v(-fDz));
        Real_v sface = Abs((point - pa).Dot(fNormals[i]));
        vecCore::MaskedAssign(safplanar, sface < safplanar, sface);
        continue;
      }
      int j = (i + 1) % N;
      dx    = vertexX[j] - vertexX[i];
      dy    = vertexY[j] - vertexY[i];
      dpx   = point[0] - vertexX[i];
      dpy   = point[1] - vertexY[i];
      lsq   = dx * dx + dy * dy;
      u     = (dpx * dx + dpy * dy) / NonZero(lsq);
      vecCore__MaskedAssignFunc(dpx, u > 1, point[0] - vertexX[j]);
      vecCore__MaskedAssignFunc(dpy, u > 1, point[1] - vertexY[j]);
      vecCore__MaskedAssignFunc(dpx, u >= 0 && u <= 1, dpx - u * dx);
      vecCore__MaskedAssignFunc(dpy, u >= 0 && u <= 1, dpy - u * dy);
      Real_v ssq = dpx * dpx + dpy * dpy; // safety squared
      vecCore__MaskedAssignFunc(dx1, ssq < safety, Real_v(fxc[i] - fxa[i]));
      vecCore__MaskedAssignFunc(dx2, ssq < safety, Real_v(fxd[i] - fxb[i]));
      vecCore__MaskedAssignFunc(dy1, ssq < safety, Real_v(fyc[i] - fya[i]));
      vecCore__MaskedAssignFunc(dy2, ssq < safety, Real_v(fyd[i] - fyb[i]));
      vecCore::MaskedAssign(umin, ssq < safety, u);
      vecCore::MaskedAssign(safety, ssq < safety, ssq);
    }
    vecCore__MaskedAssignFunc(umin, umin < 0 || umin > 1, Real_v(0.));
    dx = dx1 + umin * (dx2 - dx1);
    dy = dy1 + umin * (dy2 - dy1);
    // Denominator below always positive as fDz>0
    safety *= Real_v(1.) - Real_v(4.) * fDz * fDz / (dx * dx + dy * dy + Real_v(4.) * fDz * fDz);
    safety = Sqrt(safety);
    vecCore::MaskedAssign(safety, safplanar < safety, safplanar);
    vecCore__MaskedAssignFunc(safety, wrong, -safety);
    vecCore__MaskedAssignFunc(safety, onsurf, Real_v(0.));
    return safety;
  } // end SafetyCurved

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vertex_t const *GetNormals() const { return fNormals; }

  //______________________________________________________________________________
  /**
   * @brief Computes if point on surface isurf is within surface limits
   * @param point Starting point
   * @param isurf Surface index
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  vecCore::Mask_v<Real_v> InSurfLimits(Vector3D<Real_v> const &point, int isurf) const
  {

    using Bool_v = vecCore::Mask_v<Real_v>;
    // Check first if Z is in range
    Real_v rz     = fDz2 * (point.z() + fDz);
    Bool_v insurf = (rz > Real_v(MakeMinusTolerant<true>(0.))) && (rz < Real_v(MakePlusTolerant<true>(1.)));
    if (vecCore::MaskEmpty(insurf)) return insurf;

    Real_v r     = InfinityLength<Real_v>();
    Real_v num   = (point.x() - fxa[isurf]) - rz * (fxb[isurf] - fxa[isurf]);
    Real_v denom = (fxc[isurf] - fxa[isurf]) + rz * (fxd[isurf] - fxc[isurf] - fxb[isurf] + fxa[isurf]);
    vecCore__MaskedAssignFunc(r, (Abs(denom) > Real_v(1.e-6)), num / NonZero(denom));
    num   = (point.y() - fya[isurf]) - rz * (fyb[isurf] - fya[isurf]);
    denom = (fyc[isurf] - fya[isurf]) + rz * (fyd[isurf] - fyc[isurf] - fyb[isurf] + fya[isurf]);
    vecCore__MaskedAssignFunc(r, (Abs(denom) > Real_v(1.e-6)), num / NonZero(denom));
    insurf = insurf && (r > Real_v(MakeMinusTolerant<true>(0.))) && (r < Real_v(MakePlusTolerant<true>(1.)));
    return insurf;
  }

  //______________________________________________________________________________
  /** @brief Computes un-normalized normal to surface isurf, on the input point */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void UNormal(Vector3D<Real_v> const &point, int isurf, Vector3D<Real_v> &unorm, Real_v &rz, Real_v &r) const
  {

    // unorm = (vi X hi0) + rz*(vi X vj) + r*(hi1 X hi0)
    //    where: vi, vj are the vectors (AB) and (CD) (see constructor)
    //           hi0 = (AC) and hi1 = (BD)
    //           rz = 0.5*(point.z()+dz)/dz is the vertical ratio
    //           r = ((AP)-rz*vi) / (hi0+rz(vj-vi)) is the horizontal ratio
    // Any point within the surface range should reurn r and rz in the range [0,1]
    // These can be used as surface crossing criteria
    rz = fDz2 * (point.z() + fDz);
    /*
      Vector3D<Real_v> a(fxa[isurf], fya[isurf], -fDz);
      Vector3D<Real_v> vi(fxb[isurf]-fxa[isurf], fyb[isurf]-fya[isurf], 2*fDz);
      Vector3D<Real_v> vj(fxd[isurf]-fxc[isurf], fyd[isurf]-fyc[isurf], 2*fDz);
      Vector3D<Real_v> hi0(fxc[isurf]-fxa[isurf], fyc[isurf]-fya[isurf], 0.);
    */
    Real_v num   = (point.x() - fxa[isurf]) - rz * (fxb[isurf] - fxa[isurf]);
    Real_v denom = (fxc[isurf] - fxa[isurf]) + rz * (fxd[isurf] - fxc[isurf] - fxb[isurf] + fxa[isurf]);
    vecCore__MaskedAssignFunc(r, Abs(denom) > Real_v(1.e-6), num / NonZero(denom));
    num   = (point.y() - fya[isurf]) - rz * (fyb[isurf] - fya[isurf]);
    denom = (fyc[isurf] - fya[isurf]) + rz * (fyd[isurf] - fyc[isurf] - fyb[isurf] + fya[isurf]);
    vecCore__MaskedAssignFunc(r, Abs(denom) > Real_v(1.e-6), num / NonZero(denom));

    unorm = (Vector3D<Real_v>)fViCrossHi0[isurf] + rz * (Vector3D<Real_v>)fViCrossVj[isurf] +
            r * (Vector3D<Real_v>)fHi1CrossHi0[isurf];
  } // end UNormal

  //______________________________________________________________________________
  /** @brief Solver for the second degree equation for curved surface crossing */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  /**
   * Function to compute smin and smax crossings with the N lateral surfaces.
   */
  void ComputeSminSmax(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir, Real_v smin[N], Real_v smax[N]) const
  {

    const Real_v big = InfinityLength<Real_v>();

    Real_v dzp = fDz + point[2];
    // calculate everything needed to solve the second order equation
    Real_v a[N], b[N], c[N], d[N];
    Real_v signa[N], inva[N];

    // vectorizes
    for (int i = 0; i < N; ++i) {
      Real_v xs1 = fxa[i] + ftx1[i] * dzp;
      Real_v ys1 = fya[i] + fty1[i] * dzp;
      Real_v xs2 = fxc[i] + ftx2[i] * dzp;
      Real_v ys2 = fyc[i] + fty2[i] * dzp;
      Real_v dxs = xs2 - xs1;
      Real_v dys = ys2 - ys1;
      a[i]       = (fDeltatx[i] * dir[1] - fDeltaty[i] * dir[0] + ft1crosst2[i] * dir[2]) * dir[2];
      b[i]       = dxs * dir[1] - dys * dir[0] +
             (fDeltatx[i] * point[1] - fDeltaty[i] * point[0] + fty2[i] * xs1 - fty1[i] * xs2 + ftx1[i] * ys2 -
              ftx2[i] * ys1) *
                 dir[2];
      c[i] = dxs * point[1] - dys * point[0] + xs1 * ys2 - xs2 * ys1;
      d[i] = b[i] * b[i] - 4 * a[i] * c[i];
    }

    // does not vectorize
    for (int i = 0; i < N; ++i) {
      // zero or one to start with
      signa[i] = Real_v(0.);
      vecCore__MaskedAssignFunc(signa[i], a[i] < -kTolerance, Real_v(-1.));
      vecCore__MaskedAssignFunc(signa[i], a[i] > kTolerance, Real_v(1.));
      inva[i] = c[i] / NonZero(b[i] * b[i]);
      vecCore__MaskedAssignFunc(inva[i], Abs(a[i]) > kTolerance, Real_v(1.) / NonZero(Real_v(2.) * a[i]));
    }

    // vectorizes
    for (int i = 0; i < N; ++i) {
      // treatment for curved surfaces. Invalid solutions will be excluded.
      Real_v sqrtd = signa[i] * Sqrt(Abs(d[i]));
      vecCore::MaskedAssign(sqrtd, d[i] < Real_v(0.), big);
      // what is the meaning of this??
      smin[i] = (-b[i] - sqrtd) * inva[i];
      smax[i] = (-b[i] + sqrtd) * inva[i];
    }
    // For the planar surfaces, the above may be wrong, redo the work using
    // just the normal. This does not vectorize
    for (int i = 0; i < N; ++i) {
      if (fiscurved[i]) continue;
      Vertex_t va;         // vertex i of lower base
      Vector3D<Real_v> pa; // same vertex converted to backend type
      // Point A is the current vertex on lower Z. P is the point we come from.
      pa.Set(Real_v(fxa[i]), Real_v(fya[i]), Real_v(-fDz));
      Vector3D<Real_v> vecAP = point - pa;
      // Dot product between AP vector and normal to surface has to be negative
      Real_v dotAPNorm = vecAP.Dot(fNormals[i]);
      // Dot product between direction and normal to surface has to be positive
      Real_v dotDirNorm = dir.Dot(fNormals[i]);
      smin[i]           = -dotAPNorm / NonZero(dotDirNorm);
      smax[i]           = InfinityLength<Real_v>(); // not to be checked
    }
  } // end ComputeSminSmax

}; // end class definition

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom

#endif /* SECONDORDERSURFACESHELL_H_ */
