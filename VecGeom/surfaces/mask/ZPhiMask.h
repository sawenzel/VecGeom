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
  bool isFullCirc;             ///< Does the phi cut exist here?
  AngleVector<Real_t> vecSPhi; ///< Cartesian coordinates of vectors that represents the start of the phi-cut.
  AngleVector<Real_t> vecEPhi; ///< Cartesian coordinates of vectors that represents the end of the phi-cut.

  ZPhiMask() = default;
  ZPhiMask(Real_t zmin, Real_t zmax, bool isFullCircle, Real_t sphi = Real_t{0}, Real_t ephi = Real_t{0})
      : rangeZ(zmin, zmax), isFullCirc(isFullCircle)
  {
    // If there is no Phi cut, we needn't wotty about phi vectors.
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
  ///  point must also satisfy the Z limits.
  /// @param local Local coordinates of the point
  /// @return Inside the mask or not
  VECCORE_ATT_HOST_DEVICE
  bool Inside(Vector3D<Real_t> const &local) const
  {
    // The point must be inside z-span:
    if (local[2] < rangeZ[0] - vecgeom::kTolerance || local[2] > rangeZ[1] + vecgeom::kTolerance) return false;
    return InsidePhi(local);
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
    valid          = true;
    Real_t safetyZ = vecCore::math::Max(local[2] - rangeZ[1], rangeZ[0] - local[2]);
    if (InsidePhi(local)) return vecCore::math::Max(safetySurf, safetyZ);
    // If the point is not in the phi range, there are other surfaces closer than this one
    // This frame should not be part of the minimization process.
    valid = false;
    return vecgeom::InfinityLength<Real_t>();
  }
};

} // namespace vgbrep

#endif
