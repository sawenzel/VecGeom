#ifndef VECGEOM_SURFACE_ZPHIMASK_H
#define VECGEOM_SURFACE_ZPHIMASK_H

#include <VecGeom/management/Logger.h>
#include <VecGeom/surfaces/base/CommonTypes.h>

namespace vgbrep {

/// @brief Mask in cylindrical coordinates, used on cylindrical surfaces.
/// @tparam Real_t Precision type
template <typename Real_t>
struct ZPhiMask {
  //// ZPhi format:
  //// rangeZ             -> extent along z axis
  //// isFullCirc         -> Does the phi cut exist here?
  //// vecSPhi, vecEPhi   -> Cartesian coordinates of vectors that delimit phi-cut

  using value_type = Real_t;
  Range<Real_t> rangeZ;        ///< Limits on the z-axis.
  Real_t r0;                   ///< Radius at z = 0
  Real_t invcalf;              ///< Inverse of the cosine of the surface angle with respect to z axis
  AngleVector<Real_t> vecSPhi; ///< Cartesian coordinates of vectors that represents the start of the phi-cut.
  AngleVector<Real_t> vecEPhi; ///< Cartesian coordinates of vectors that represents the end of the phi-cut.
  bool isFullCirc;             ///< Does the phi cut exist here?

  ZPhiMask() = default;
  template <typename Real_i>
  ZPhiMask(Real_i zmin, Real_i zmax, bool isFullCircle, Real_i rbottom, Real_i rtop, Real_i sphi = Real_i{0},
           Real_i ephi = Real_i{0})
      : rangeZ(static_cast<Real_t>(zmin), static_cast<Real_t>(zmax)), isFullCirc(isFullCircle)
  {
    r0       = static_cast<Real_t>(0.5 * (rbottom + rtop));
    Real_t t = static_cast<Real_t>((rtop - rbottom) / (zmax - zmin));
    invcalf  = (t == Real_t(0)) ? Real_t(1) : vecCore::math::Sqrt(Real_t(1) + t * t);
    // If there is no Phi cut, we needn't worry about phi vectors.
    if (isFullCirc) return;
    vecSPhi.Set(static_cast<Real_t>(vecgeom::Cos(sphi)), static_cast<Real_t>(vecgeom::Sin(sphi)));
    vecEPhi.Set(static_cast<Real_t>(vecgeom::Cos(ephi)), static_cast<Real_t>(vecgeom::Sin(ephi)));
  }
  template <typename Real_i>
  ZPhiMask(const ZPhiMask<Real_i> &other)
      : r0(static_cast<Real_t>(other.r0)), invcalf(static_cast<Real_t>(other.invcalf)), isFullCirc(other.isFullCirc)
  {
    rangeZ  = Range<Real_t>(other.rangeZ);
    vecSPhi = AngleVector<Real_t>(other.vecSPhi);
    vecEPhi = AngleVector<Real_t>(other.vecEPhi);
  }

  /// @brief Fills extents in X and Y for the ZPhi mask
  /// @param xmin Minimum of the extent in X
  /// @param xmax Maximum of the extent in X
  /// @param ymin Minimum of the extent in Y
  /// @param ymax Maximum of the extent in Y
  void GetXYextent(Real_t &xmin, Real_t &xmax, Real_t &ymin, Real_t &ymax) const
  {
    auto const rmax = vecCore::math::Max(Radius(rangeZ[0]), Radius(rangeZ[1]));
    xmin            = -rmax;
    xmax            = rmax;
    ymin            = -rmax;
    ymax            = rmax;
    // The axis vector has to be between Rmin and Rmax and cannot be unit vector anymore
    Vector2D<Real_t> axis{1, 0};
    if (!isFullCirc) {
      // Projections of points that delimit vertices of the phi-cut ring
      Real_t x1, x2, x3, x4, y1, y2, y3, y4;

      auto const rmin = vecCore::math::Min(Radius(rangeZ[0]), Radius(rangeZ[1]));

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

  /// @brief Fills the 3D extent of the mask
  /// @param aMin Bottom extent corner
  /// @param aMax Top extent corner
  void Extent3D(Vector3D<Real_t> &aMin, Vector3D<Real_t> &aMax) const
  {
    Real_t xmin, xmax, ymin, ymax;
    GetXYextent(xmin, xmax, ymin, ymax);
    aMin.Set(xmin, ymin, rangeZ[0]);
    aMax.Set(xmax, ymax, rangeZ[1]);
  }

  /// @brief Get radius of the mask dependent on z
  /// @param z Z value
  /// @return Radius of the mask
  Real_t Radius(Real_t z) const
  {
    Real_t t = vecCore::math::Sqrt((invcalf - Real_t(1)) * (invcalf + Real_t(1)));
    return r0 + t * z;
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
  ///  point must also satisfy the Z limits.
  /// @param local Local coordinates of the point
  /// @return Inside the mask or not
  VECCORE_ATT_HOST_DEVICE
  bool Inside(Vector3D<Real_t> const &local) const
  {
    // The point must be inside z-span:
    if (local[2] < vecgeom::MakeMinusTolerant<true, Real_t>(rangeZ[0]) ||
        local[2] > vecgeom::MakePlusTolerant<true, Real_t>(rangeZ[1]))
      return false;
    return InsidePhi(local[0], local[1]);
  }

  /// @brief Transform a ZPhi mask from a local reference defined by trans to the parent reference
  /// @param trans Transformation of the ZPhi mask with respect to the parent reference
  /// @return Transformed mask
  ZPhiMask<Real_t> InverseTransform(vecgeom::Transformation2DMP<Real_t> const &trans) const
  {
    ZPhiMask<Real_t> frame(*this);
    // Convert rangeZ
    Vector3D<Real_t> local;
    local           = trans.InverseTransform(Vector3D<Real_t>{0, 0, rangeZ[0]});
    frame.rangeZ[0] = local[2];
    local           = trans.InverseTransform(Vector3D<Real_t>{0, 0, rangeZ[1]});
    frame.rangeZ[1] = local[2];

    // Convert radius at 0
    frame.r0 = Radius(-trans.Translation()[2]);

    if (!isFullCirc) {
      // Convert phi range
      local = trans.InverseTransformDirection(Vector3D<Real_t>{vecSPhi[0], vecSPhi[1], 0});
      frame.vecSPhi.Set(local[0], local[1]);
      local = trans.InverseTransformDirection(Vector3D<Real_t>{vecEPhi[0], vecEPhi[1], 0});
      frame.vecEPhi.Set(local[0], local[1]);
    }
    return frame;
  }

  /// @brief Combine this mask (1,2) with another (3,4) and get the resulting extent.
  /// @param other Another ZPhi frame mask
  bool CombineWith(ZPhiMask<Real_t> const &other)
  {
    // Check compatibility of frames
    if (vecCore::math::Abs(invcalf - other.invcalf) > vecgeom::kToleranceDist<Real_t>) return false;
    if (vecCore::math::Abs(Radius(0) - other.Radius(0)) > vecgeom::kToleranceDist<Real_t>) return false;

    rangeZ[0] = vecCore::math::Min(rangeZ[0], other.rangeZ[0]);
    rangeZ[1] = vecCore::math::Max(rangeZ[1], other.rangeZ[1]);
    isFullCirc |= other.isFullCirc;
    if (isFullCirc) return true;
    // intersect the phi ranges
    // Matching start-start (1==3)
    if (ApproxEqualVector2(vecSPhi, other.vecSPhi)) {
      if (!InsidePhi(other.vecEPhi[0], other.vecEPhi[1])) vecEPhi = other.vecEPhi;
      return true;
    }
    // Matching end-end (2==4)
    if (ApproxEqualVector2(vecEPhi, other.vecEPhi)) {
      if (!InsidePhi(other.vecSPhi[0], other.vecSPhi[1])) vecSPhi = other.vecSPhi;
      return true;
    }
    // Matching end-start ranges (2==3)
    if (ApproxEqualVector2(vecEPhi, other.vecSPhi)) {
      if (InsidePhi(other.vecEPhi[0], other.vecEPhi[1])) {
        vecEPhi    = vecSPhi;
        isFullCirc = true;
      } else {
        vecEPhi = other.vecEPhi;
      }
      return true;
    }
    // Matching start-end ranges (1==4)
    if (ApproxEqualVector2(vecSPhi, other.vecEPhi)) {
      if (InsidePhi(other.vecSPhi[0], other.vecSPhi[1])) {
        vecEPhi    = vecSPhi;
        isFullCirc = true;
      } else {
        vecSPhi = other.vecSPhi;
      }
      return true;
    }
    // non-matching ends
    bool in1 = other.InsidePhi(vecSPhi[0], vecSPhi[1]);       // 1 inside (3,4)
    bool in2 = other.InsidePhi(vecEPhi[0], vecEPhi[1]);       // 2 inside (3,4)
    bool in3 = InsidePhi(other.vecSPhi[0], other.vecSPhi[1]); // 3 inside (1,2)
    bool in4 = InsidePhi(other.vecEPhi[0], other.vecEPhi[1]); // 4 inside (1,2)

    if (!(in1 || in2 || in3 || in4)) {
      if ((vecSPhi - other.vecEPhi).Mag2() > (other.vecSPhi - vecEPhi).Mag2())
        vecEPhi = other.vecEPhi;
      else
        vecSPhi = other.vecSPhi;
      return true;
    }

    if (in1 && in2 && in3 && in4) {
      vecEPhi    = vecSPhi;
      isFullCirc = true;
      return true;
    }

    if (!(in1 || in2)) return true;

    if (!(in3 || in4)) {
      vecSPhi = other.vecSPhi;
      vecEPhi = other.vecEPhi;
      return true;
    }

    if (!in1 && in2) {
      vecEPhi = other.vecEPhi;
      return true;
    }

    if (in1 && !in2) {
      vecSPhi = other.vecSPhi;
      return true;
    }
    VECGEOM_LOG(critical) << "CombineWith: wrong logic";
    return false;
  }

  /// @brief Computes safe distance to the frame combining surface and frame safeties
  /// @details The safety to the cylindrical surface comes as safetySurf, it is positive since only
  ///  points outside the cylindrical shell are considered.
  /// @param local Projected point in local coordinates
  /// @param safetySurf Safety from non-projected point to the frame support surface.
  /// @return Safety distance to the mask.
  VECCORE_ATT_HOST_DEVICE
  Real_t Safety(Vector3D<Real_t> const &local, Real_t safetySurf, bool &valid) const
  {
    valid = true;
    if (!InsidePhi(local[0], local[1])) {
      // If the point is not in the phi range, there are other surfaces closer than this one
      // so this frame should not be part of the minimization process.
      valid = false;
      return 0.;
    }
    auto safetyZ = vecCore::math::Max(local[2] - rangeZ[1], rangeZ[0] - local[2]);
    return vecCore::math::Max(safetySurf, safetyZ);
  }
};

} // namespace vgbrep

#endif
