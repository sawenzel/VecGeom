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

  template <typename Real_i>
  RingMask(Real_i rmin, Real_i rmax, bool isFullCircle, Real_i sphi = Real_i{0}, Real_i ephi = Real_i{0})
      : rangeR(static_cast<Real_t>(rmin), static_cast<Real_t>(rmax)), isFullCirc(isFullCircle)
  {
    // If there is no Phi cut, no need to set phi vectors.
    if (isFullCirc) return;
    vecSPhi.Set(static_cast<Real_t>(vecgeom::Cos(sphi)), static_cast<Real_t>(vecgeom::Sin(sphi)));
    vecEPhi.Set(static_cast<Real_t>(vecgeom::Cos(ephi)), static_cast<Real_t>(vecgeom::Sin(ephi)));
  }

  template <typename Real_i>
  RingMask(const RingMask<Real_i> &other)
      : rangeR(static_cast<Real_t>(other.rangeR[0]), static_cast<Real_t>(other.rangeR[1])), isFullCirc(other.isFullCirc)
  {
    // If there is no Phi cut, no need to set phi vectors.
    if (isFullCirc) return;
    vecSPhi.Set(static_cast<Real_t>(other.vecSPhi[0]), static_cast<Real_t>(other.vecSPhi[1]));
    vecEPhi.Set(static_cast<Real_t>(other.vecEPhi[0]), static_cast<Real_t>(other.vecEPhi[1]));
  }

  /// @brief Fills extents in X and Y for the ring mask
  /// @param xmin Minimum of the extent in X
  /// @param xmax Maximum of the extent in X
  /// @param ymin Minimum of the extent in Y
  /// @param ymax Maximum of the extent in Y
  void GetXYextent(Real_t &xmin, Real_t &xmax, Real_t &ymin, Real_t &ymax) const
  {
    auto const &rmax = rangeR[1];
    xmin             = -rmax;
    xmax             = rmax;
    ymin             = -rmax;
    ymax             = rmax;
    // The axis vector has to be between Rmin and Rmax and cannot be unit vector anymore
    Vector2D<Real_t> axis{1, 0};
    if (!isFullCirc) {
      // Projections of points that delimit vertices of the phi-cut ring
      Real_t x1, x2, x3, x4, y1, y2, y3, y4;

      auto const &rmin = rangeR[0];

      x1 = rmax * axis.Dot(vecSPhi); //< (sphi, Rmax)_x
      x2 = rmax * axis.Dot(vecEPhi); //< (ephi, Rmax)_x
      x3 = rmin * axis.Dot(vecSPhi); //< (sphi, Rmin)_x
      x4 = rmin * axis.Dot(vecEPhi); //< (ephi, Rmin)_x
      axis.Set(0, 1);
      y1 = rmax * axis.Dot(vecSPhi); //< (sphi, Rmax)_x
      y2 = rmax * axis.Dot(vecEPhi); //< (ephi, Rmax)_x
      y3 = rmin * axis.Dot(vecSPhi); //< (sphi, Rmin)_x
      y4 = rmin * axis.Dot(vecEPhi); //< (ephi, Rmin)_x

      xmax = vecgeom::Max(vecgeom::Max(x1, x2), vecgeom::Max(x3, x4));
      ymax = vecgeom::Max(vecgeom::Max(y1, y2), vecgeom::Max(y3, y4));
      xmin = vecgeom::Min(vecgeom::Min(x1, x2), vecgeom::Min(x3, x4));
      ymin = vecgeom::Min(vecgeom::Min(y1, y2), vecgeom::Min(y3, y4));
      // If the axes lie within the circle
      if (InsidePhi(1, 0)) xmax = rmax;
      if (InsidePhi(0, 1)) ymax = rmax;
      if (InsidePhi(-1, 0)) xmin = -rmax;
      if (InsidePhi(0, -1)) ymin = -rmax;
    }
  }

  /// @brief Fills the 3D extent of the ring
  /// @param aMin Bottom extent corner
  /// @param aMax Top extent corner
  void Extent3D(Vector3D<Real_t> &aMin, Vector3D<Real_t> &aMax) const
  {
    Real_t xmin, xmax, ymin, ymax;
    GetXYextent(xmin, xmax, ymin, ymax);
    aMin.Set(xmin, ymin, Real_t(0));
    aMax.Set(xmax, ymax, Real_t(0));
  }

  /// @brief Returns the window extent of the ring
  /// @param window Extent window to be filled
  void GetExtent(WindowMask<Real_t> &window) const
  {
    Real_t xmin, xmax, ymin, ymax;
    GetXYextent(xmin, xmax, ymin, ymax);
    window.rangeU.Set(xmin, xmax);
    window.rangeV.Set(ymin, ymax);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool HasRmin() const { return rangeR[0] > vecgeom::kToleranceDist<Real_t>; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsConvex() const { return !HasRmin() && vecSPhi.CrossZ(vecEPhi) > -vecgeom::kToleranceDist<Real_t>; }

  /// @brief Check if local point is in the radius range
  /// @param local Point in local coordinates
  /// @return Point inside range
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool InsideR(Vector3D<Real_t> const &local) const
  {
    Real_t rsq = local[0] * local[0] + local[1] * local[1];
    // The point must be inside the ring:
    if ((rsq < rangeR[0] * rangeR[0] - 2 * vecgeom::kRelTolerance<Real_t>(rangeR[0])) ||
        (rsq > rangeR[1] * rangeR[1] + 2 * vecgeom::kRelTolerance<Real_t>(rangeR[1])))
      return false;
    return true;
  }

  /// @brief Check if local point is in the phi range
  /// @param x Local x coordinate
  /// @param y Local y coordinate
  /// @return Point inside phi
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool InsidePhi(Real_t const &x, Real_t const &y) const
  {
    if (isFullCirc) return true;
    AngleVector<Real_t> localAngle{x, y};
    auto convex = vecSPhi.CrossZ(vecEPhi) > Real_t(0);
    auto in1    = vecSPhi.CrossZ(localAngle) > -vecgeom::kToleranceStrict<Real_t>;
    auto in2    = localAngle.CrossZ(vecEPhi) > -vecgeom::kToleranceStrict<Real_t>;
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
  bool Inside(Vector3D<Real_t> const &local) const { return InsideR(local) && InsidePhi(local[0], local[1]); }

  /// @brief Computes safe distance to the frame combining surface and frame safeties (under-estimate)
  /// @details Computes first the maximum signed distance to each edge on a single axis. Two versions
  /// @param local Projected point in local coordinates
  /// @param safetySurf Safety from non-projected point to the frame support surface.
  /// @return Safety distance to the rectangle mask.
  VECCORE_ATT_HOST_DEVICE
  Real_t Safety(Vector3D<Real_t> const &local, Real_t safetySurf, bool &valid) const
  {
    valid       = true;
    Real_t rho  = local.Perp(); // still a square root, but less registers/branching
    Real_t safR = HasRmin() ? vecCore::math::Max(rangeR[0] - rho, rho - rangeR[1]) : rho - rangeR[1];
    if (isFullCirc || InsidePhi(local[0], local[1]))
#ifdef SURF_ACCURATE_SAFETY
      return vecCore::math::Sqrt(safetySurf * safetySurf + safR * safR);
#else
      return vecCore::math::Max(safetySurf, safR);
#endif
    AngleVector<Real_t> localAngle{local[0], local[1]};
    Real_t safPhi = vecCore::math::Max(localAngle.CrossZ(vecSPhi), -localAngle.CrossZ(vecEPhi));
#ifdef SURF_ACCURATE_SAFETY
    return vecCore::math::Sqrt(safetySurf * safetySurf + safR * safR + safPhi * safPhi);
#else
    return vecCore::math::Max(safetySurf, safR, safPhi);
#endif
  }

  /// @brief Safe distance from a point assumed inside the window
  /// @details Used on host only for frame checks
  /// @param local Projected point in local coordinates
  /// @return Safe distance
  Real_t SafetyInside(Vector3D<Real_t> const &local) const
  {
    Real_t rho  = local.Perp();
    Real_t safR = HasRmin() ? vecCore::math::Min(rangeR[1] - rho, rho - rangeR[0]) : rangeR[1] - rho;
    safR        = vecCore::math::Max(safR, Real_t(0));
    if (isFullCirc) return safR;
    AngleVector<Real_t> localAngle{local[0], local[1]};
    auto saf1     = (vecSPhi.Dot(localAngle) > Real_t(0)) ? vecCore::math::Max(vecSPhi.CrossZ(localAngle), Real_t(0))
                                                          : vecgeom::InfinityLength<Real_t>();
    auto saf2     = (vecEPhi.Dot(localAngle) > Real_t(0)) ? vecCore::math::Max(localAngle.CrossZ(vecEPhi), Real_t(0))
                                                          : vecgeom::InfinityLength<Real_t>();
    Real_t safPhi = vecCore::math::Min(saf1, saf2);
    return vecCore::math::Min(safR, safPhi);
  }
};

} // namespace vgbrep

#endif
