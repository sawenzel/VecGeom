#ifndef VECGEOM_SURFACE_RINGMASK_H
#define VECGEOM_SURFACE_RINGMASK_H

#include <VecGeom/surfaces/base/CommonTypes.h>

namespace vgbrep {

/// @brief Ring masks on plane surfaces.
/// @tparam Real_t Precision used
template <typename Real_t>
struct RingMask {
  Range<Real_t> rangeR;        ///< Radius limits in the form of [Rmin, Rmax].
  bool isFullCirc;             ///< Does the phi cut exist here?
  AngleVector<Real_t> vecSPhi; ///< Cartesian coordinates of vectors that represents the start of the phi-cut.
  AngleVector<Real_t> vecEPhi; ///< Cartesian coordinates of vectors that represents the end of the phi-cut.

  RingMask() = default;
  RingMask(Real_t rmin, Real_t rmax, bool isFullCircle, Real_t sphi = Real_t{0}, Real_t ephi = Real_t{0})
      : rangeR(rmin, rmax), isFullCirc(isFullCircle)
  {
    // If there is no Phi cut, we needn't wotty about phi vectors.
    if (isFullCirc) return;
    vecSPhi.Set(vecgeom::Cos(sphi), vecgeom::Sin(sphi));
    vecEPhi.Set(vecgeom::Cos(ephi), vecgeom::Sin(ephi));
  };

  /// @brief Returns the extent of the window
  /// @param window Extent window to be filled
  void GetExtent(WindowMask<Real_t> &window) const
  {
    auto const &Rmax = rangeR[1];
    Real_t xmin{-Rmax}, xmax{Rmax}, ymin{-Rmax}, ymax{Rmax};
    // The axis vector has to be between Rmin and Rmax and cannot be unit vector anymore
    auto const Rmean = (rangeR[0] + Rmax) * 0.5;
    Vector2D<Real_t> axis{1, 0};
    if (!isFullCirc) {
      // Projections of points that delimit vertices of the phi-cut ring
      Real_t x1, x2, x3, x4, y1, y2, y3, y4;

      auto Rmin = rangeR[0];

      x1 = Rmax * axis.Dot(vecSPhi); //< (sphi, Rmax)_x
      x2 = Rmax * axis.Dot(vecEPhi); //< (ephi, Rmax)_x
      x3 = Rmin * axis.Dot(vecSPhi); //< (sphi, Rmin)_x
      x4 = Rmin * axis.Dot(vecEPhi); //< (ephi, Rmin)_x
      axis.Set(0, 1);
      y1 = Rmax * axis.Dot(vecSPhi); //< (sphi, Rmax)_x
      y2 = Rmax * axis.Dot(vecEPhi); //< (ephi, Rmax)_x
      y3 = Rmin * axis.Dot(vecSPhi); //< (sphi, Rmin)_x
      y4 = Rmin * axis.Dot(vecEPhi); //< (ephi, Rmin)_x

      xmax = vecgeom::Max(vecgeom::Max(x1, x2), vecgeom::Max(x3, x4));
      ymax = vecgeom::Max(vecgeom::Max(y1, y2), vecgeom::Max(y3, y4));
      xmin = vecgeom::Min(vecgeom::Min(x1, x2), vecgeom::Min(x3, x4));
      ymin = vecgeom::Min(vecgeom::Min(y1, y2), vecgeom::Min(y3, y4));
      // If the axes lie within the circle
      if (Inside(Vector3D<Real_t>(Rmean, 0, 0))) xmax = Rmax;
      if (Inside(Vector3D<Real_t>(0, Rmean, 0))) ymax = Rmax;
      if (Inside(Vector3D<Real_t>(-Rmean, 0, 0))) xmin = -Rmax;
      if (Inside(Vector3D<Real_t>(0, -Rmean, 0))) ymin = -Rmax;
    }

    window.rangeU.Set(xmin, xmax);
    window.rangeV.Set(ymin, ymax);
  }

  /// @brief Check if local point is in the radius range
  /// @param local Point in local coordinates
  /// @return Point inside range
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool InsideR(Vector3D<Real_t> const &local) const
  {
    Real_t rsq = local[0] * local[0] + local[1] * local[1];
    // The point must be inside the ring:
    if ((rsq < rangeR[0] * rangeR[0] + 2 * vecgeom::kToleranceSquared * rangeR[0]) ||
        (rsq > rangeR[1] * rangeR[1] - 2 * vecgeom::kToleranceSquared * rangeR[1]))
      return false;
    return true;
  }

  /// @brief Check if local point is in the phi range
  /// @param local Point in local coordinates
  /// @return Point inside phi
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool InsidePhi(Vector3D<Real_t> const &local) const
  {
    if (isFullCirc) return true;
    AngleVector<Real_t> localAngle{local[0], local[1]};
    auto convex = vecSPhi.CrossZ(vecEPhi) > Real_t(0);
    auto in1    = vecSPhi.CrossZ(localAngle) > -vecgeom::kTolerance;
    auto in2    = localAngle.CrossZ(vecEPhi) > -vecgeom::kTolerance;
    return convex ? in1 && in2 : in1 || in2;
  }

  /// @brief Checks if the point is within mask.
  /// @details For the phi part, use the cross-products of the point vector with the unit
  ///  phi vectors, representing the signed safeties with respect to these vectors. The
  ///  point must also satisfy the radius limits.
  /// @param local Local coordinates of the point
  /// @return Inside the mask or not
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool Inside(Vector3D<Real_t> const &local) const { return InsideR(local) && InsidePhi(local); }

  /// @brief Computes safe distance to the frame combining surface and frame safeties (under-estimate)
  /// @details Computes first the maximum signed distance to each edge on a single axis. Two versions
  /// @param local Projected point in local coordinates
  /// @param safetySurf Safety from non-projected point to the frame support surface.
  /// @return Safety distance to the rectangle mask.
  VECCORE_ATT_HOST_DEVICE
  Real_t Safety(Vector3D<Real_t> const &local, Real_t safetySurf, bool &valid) const
  {
    valid         = true;
    Real_t rho    = local.Perp();
    Real_t safR   = vecCore::math::Max(rangeR[0] - rho, rho - rangeR[1]);
    Real_t safety = vecCore::math::Max(safR, safetySurf);
    if (isFullCirc) return safety;

    if (!InsidePhi(local)) {
      AngleVector<Real_t> localAngle{local[0], local[1]};
      Real_t safPhi = vecCore::math::Max(localAngle.CrossZ(vecSPhi), -localAngle.CrossZ(vecEPhi));
      safety        = vecCore::math::Max(safety, safPhi);
    }

    return safety;
  }
};

} // namespace vgbrep

#endif
