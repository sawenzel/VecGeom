#ifndef VECGEOM_SURFACE_ZPHIMASK_H
#define VECGEOM_SURFACE_ZPHIMASK_H

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

  Range<Real_t> rangeZ;        ///< Limits on the z-axis.
  Real_t invcalf;              ///< Inverse of the cosine of the surface angle with respect to z axis
  AngleVector<Real_t> vecSPhi; ///< Cartesian coordinates of vectors that represents the start of the phi-cut.
  AngleVector<Real_t> vecEPhi; ///< Cartesian coordinates of vectors that represents the end of the phi-cut.
  bool isFullCirc;             ///< Does the phi cut exist here?

  ZPhiMask() = default;
  ZPhiMask(Real_t zmin, Real_t zmax, bool isFullCircle, Real_t sphi = Real_t{0}, Real_t ephi = Real_t{0},
           Real_t rbottom = Real_t{0}, Real_t rtop = Real_t{0})
      : rangeZ(zmin, zmax), isFullCirc(isFullCircle)
  {
    Real_t t = (rtop - rbottom) / (zmax - zmin);
    invcalf  = vecCore::math::Abs(vecCore::math::Sqrt(Real_t(1) + t * t));
    // If there is no Phi cut, we needn't worry about phi vectors.
    if (isFullCirc) return;
    vecSPhi.Set(vecgeom::Cos(sphi), vecgeom::Sin(sphi));
    vecEPhi.Set(vecgeom::Cos(ephi), vecgeom::Sin(ephi));
  };

  void GetMask(ZPhiMask<Real_t> &mask)
  {
    mask.rangeZ.Set(rangeZ[0], rangeZ[1]);
    mask.isFullCirc = isFullCirc;
    if (isFullCirc) return;
    mask.vecSPhi.Set(vecSPhi[0], vecSPhi[1]);
    mask.vecEPhi.Set(vecEPhi[0], vecEPhi[1]);
  }

  /// @brief Check if local point is in the phi range
  /// @param local Point in local coordinates
  /// @return Point inside phi
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool InsidePhi(Real_t x, Real_t y, Real_t tolerance = vecgeom::kTolerance) const
  {
    if (isFullCirc) return true;
    AngleVector<Real_t> localAngle{x, y};
    auto convex = vecSPhi.CrossZ(vecEPhi) > Real_t(0);
    auto in1    = vecSPhi.CrossZ(localAngle) > -tolerance;
    auto in2    = localAngle.CrossZ(vecEPhi) > -tolerance;
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
    if (local[2] < rangeZ[0] - vecgeom::kTolerance || local[2] > rangeZ[1] + vecgeom::kTolerance) return false;
    return InsidePhi(local[0], local[1]);
  }

  /// @brief Transform a ZPhi mask from a local reference defined by trans to the parent reference
  /// @param trans Transformation of the ZPhi mask with respect to the parent reference
  /// @return Transformed mask
  ZPhiMask<Real_t> InverseTransform(Transformation const &trans) const
  {
    ZPhiMask<Real_t> frame;
    // Convert rangeZ
    Vector3D<Real_t> local;
    local            = trans.InverseTransform(Vector3D<Real_t>{0, 0, rangeZ[0]});
    frame.rangeZ[0]  = local[2];
    local            = trans.InverseTransform(Vector3D<Real_t>{0, 0, rangeZ[1]});
    frame.rangeZ[1]  = local[2];
    frame.isFullCirc = isFullCirc;
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
  void CombineWith(ZPhiMask<Real_t> const &other)
  {
    rangeZ[0] = vecCore::math::Min(rangeZ[0], other.rangeZ[0]);
    rangeZ[1] = vecCore::math::Max(rangeZ[1], other.rangeZ[1]);
    isFullCirc |= other.isFullCirc;
    if (!isFullCirc) {
      // intersect the phi ranges
      // Matching start-start (1==3)
      if (ApproxEqualVector2(vecSPhi, other.vecSPhi)) {
        if (!InsidePhi(other.vecEPhi[0], other.vecEPhi[1])) vecEPhi = other.vecEPhi;
        return;
      }
      // Matching end-end (2==4)
      if (ApproxEqualVector2(vecEPhi, other.vecEPhi)) {
        if (!InsidePhi(other.vecSPhi[0], other.vecSPhi[1])) vecSPhi = other.vecSPhi;
        return;
      }
      // Matching end-start ranges (2==3)
      if (ApproxEqualVector2(vecEPhi, other.vecSPhi)) {
        if (InsidePhi(other.vecEPhi[0], other.vecEPhi[1])) {
          vecEPhi    = vecSPhi;
          isFullCirc = true;
        } else {
          vecEPhi = other.vecEPhi;
        }
        return;
      }
      // Matching start-end ranges (1==4)
      if (ApproxEqualVector2(vecSPhi, other.vecEPhi)) {
        if (InsidePhi(other.vecSPhi[0], other.vecSPhi[1])) {
          vecEPhi    = vecSPhi;
          isFullCirc = true;
        } else {
          vecSPhi = vecEPhi;
          vecEPhi = other.vecSPhi;
        }
        return;
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
        return;
      }

      if (in1 && in2 && in3 && in4) {
        vecEPhi    = vecSPhi;
        isFullCirc = true;
        return;
      }

      if (!(in1 || in2)) return;

      if (!(in3 || in4)) {
        vecSPhi = other.vecSPhi;
        vecEPhi = other.vecEPhi;
        return;
      }

      if (!in1 && in2) {
        vecEPhi = other.vecEPhi;
        return;
      }

      if (in1 && !in2) {
        vecSPhi = other.vecSPhi;
        return;
      }
      assert(0 && "wrong logic for ZPhiMask::IntersectExtent");
    }
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
      return vecgeom::InfinityLength<Real_t>();
    }
    Real_t safetyZ = vecCore::math::Max(local[2] - rangeZ[1], rangeZ[0] - local[2]);
    if (safetyZ < 0) return safetySurf;
      // Correct safetyZ by the cosine of the angle between the surface generators and the Z axis
#ifdef SURF_ACCURATE_SAFETY
    return vecCore::math::Sqrt(safetySurf * safetySurf + safetyZ * safetyZ * invcalf * invcalf);
#else
    return vecCore::math::Max(safetySurf, safetyZ * calf);
#endif
  }
};

} // namespace vgbrep

#endif
