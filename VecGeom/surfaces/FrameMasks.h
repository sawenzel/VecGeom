#ifndef VECGEOM_SURFACE_FRAMEMASKS_H
#define VECGEOM_SURFACE_FRAMEMASKS_H

#include <VecGeom/surfaces/CommonTypes.h>

// FIXME: This should not be here; used to pull in Make{Plus,Minus}Tolerant
#include <VecGeom/volumes/kernel/GenericKernels.h>

#define BOX_ACCURATE_SAFETY 1

namespace vgbrep {

//
//  Masks for different types of frames
//
/// TODO: Implement other masks, too.

/**
 * @brief Rectangular masks on plane surfaces.
 *
 * @tparam Real_t Scalar type for representing rectangle limit on axis.
 */

/// @brief Rectangular masks on plane surfaces.
/// @tparam Real_t Precision type
template <typename Real_t>
struct WindowMask {
  Range<Real_t> rangeU; ///< Rectangle limits on x axis.
  Range<Real_t> rangeV; ///< Rectangle limits on y axis.

  WindowMask() = default;
  WindowMask(Real_t u1, Real_t u2, Real_t v1, Real_t v2) : rangeU(u1, u2), rangeV(v1, v2){};
  WindowMask(Real_t u, Real_t v) : rangeU(-u, u), rangeV(-v, v){};

  /// @brief Returns the extent of the window
  /// @param window Extent window to be filled
  void GetExtent(WindowMask<Real_t> &window) const
  {
    window.rangeU = rangeU;
    window.rangeV = rangeV;
  }

  VECCORE_ATT_HOST_DEVICE
  bool Inside(Vector3D<Real_t> const &local) const
  {
    return (local[0] > vecgeom::MakeMinusTolerant<true>(rangeU[0]) &&
            local[0] < vecgeom::MakePlusTolerant<true>(rangeU[1]) &&
            local[1] > vecgeom::MakeMinusTolerant<true>(rangeV[0]) &&
            local[1] < vecgeom::MakePlusTolerant<true>(rangeV[1]));
  }

  /// @brief Computes safe distance to the frame combining surface and frame safeties.
  /// @details Computes first the maximum signed distance to each edge on a single axis. Two versions
  ///  are supported:
  ///  - under-estimate (default): Just use the maximum between the edge and surface components
  ///  - accurate: Zero negative safety components, then use Pythagoras of surface and edge components
  /// @param local Projected point in local coordinates
  /// @param safetySurf Safety from non-projected point to the frame support surface.
  /// @return Safety distance to the rectangle mask.
  VECCORE_ATT_HOST_DEVICE
  Real_t Safety(Vector3D<Real_t> const &local, Real_t safetySurf, bool &valid) const
  {
    valid     = true;
    Real_t sx = vecCore::math::Max(local[0] - rangeU[1], rangeU[0] - local[0]);
    Real_t sy = vecCore::math::Max(local[1] - rangeV[1], rangeV[0] - local[1]);
#ifdef BOX_ACCURATE_SAFETY
    // The following returns the accurate safety at the price of an extra square root per frame
    // meaning 6 for a box.
    sx = vecCore::math::Max(Real_t(0), sx);
    sy = vecCore::math::Max(Real_t(0), sy);
    return vecCore::math::Sqrt(sx * sx + sy * sy + safetySurf * safetySurf);
#else
    // The VecGeom box version just returns the maximum
    return vecCore::math::Max(sx, sy, safetySurf);
#endif
  }
};

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

/**
 * @brief Mask in cylindrical coordinates, used on cylindrical surfaces.
 *
 * @tparam Real_t Scalar value for representing angles and coordinates.
 */
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

/**
 * @brief Triangular masks in the XY plane.
 *
 * @tparam Real_t is data type for storing coordinates.
 */
template <typename Real_t>
struct TriangleMask {
  Point2D<Real_t> p_[3] = {Real_t(0)}; ///< 2D coordinates of the vertices.
  Point2D<Real_t> n_[3] = {Real_t(0)}; ///< 2D coordinates of the outwards normals to segments

  TriangleMask() = default;
  TriangleMask(Real_t x1, Real_t y1, Real_t x2, Real_t y2, Real_t x3, Real_t y3)
  {
    p_[0].Set(x1, y1);
    p_[1].Set(x2, y2);
    p_[2].Set(x3, y3);

    // Compute outward normals
    for (int i = 0; i < 3; ++i) {
      auto j      = (i + 1) % 3;
      auto k      = (i + 2) % 3;
      auto seg_ij = p_[j] - p_[i];
      auto seg_ik = p_[k] - p_[i];
      assert(seg_ij.Mag2() > vecgeom::kToleranceSquared);
      // normal in XY plane
      n_[i].Set(seg_ij.y(), -seg_ij.x());
      // flip the normal so point k is 'backwards'
      if (n_[i].Dot(seg_ik) > Real_t(0)) n_[i] *= Real_t(-1);
      // Normalize normal vector
      n_[i].Normalize();
    }
  }

  /// @brief Returns the extent of the triangle
  /// @param window Extent window to be filled
  void GetExtent(WindowMask<Real_t> &window) const
  {
    window.rangeU.Set(p_[0].x());
    window.rangeV.Set(p_[0].y());
    for (int i = 1; i < 3; ++i) {
      window.rangeU.Set(vecCore::math::Min(window.rangeU[0], p_[i].x()),
                        vecCore::math::Max(window.rangeU[1], p_[i].x()));
      window.rangeV.Set(vecCore::math::Min(window.rangeV[0], p_[i].y()),
                        vecCore::math::Max(window.rangeV[1], p_[i].y()));
    }
  }

  void GetMask(TriangleMask<Real_t> &mask)
  {
    for (int i = 0; i < 3; ++i) {
      mask.p_[i] = p_[i];
      mask.n_[i] = n_[i];
    }
  }

  /// @brief Checks if the point is within mask.
  /// @details The point is within the triangle if all dot products of point
  /// position relative to each vertex and corresponding segment normal are negative.
  /// @param local Local coordinates of the point
  /// @return true if the point is inside the mask.
  VECCORE_ATT_HOST_DEVICE
  bool Inside(Vector3D<Real_t> const &local) const
  {
    // TODO: Do we need a tolerance-aware version?
    Vector2D<Real_t> const local2D(local.x(), local.y());
    return (n_[0].Dot(local2D - p_[0]) < Real_t(0) && n_[1].Dot(local2D - p_[1]) < Real_t(0) &&
            n_[2].Dot(local2D - p_[2]) < Real_t(0));
  }

  /// @brief Computes the closest distance from a point in XY plane and the triangle.
  /// @param local Local coordinates of the point
  /// @return Safety from point to triangle.
  VECCORE_ATT_HOST_DEVICE
  Real_t Safety(Vector3D<Real_t> const &local, Real_t safetySurf, bool &valid) const
  {
    // The algorithm currently gives an underestimate
    valid = true;
    Vector2D<Real_t> const local2D(local.x(), local.y());
    // Compute signed safeties to segments
    Real_t safety = safetySurf;
    for (int i = 0; i < 3; ++i)
      safety = vecCore::math::Max(safety, n_[i].Dot(local2D - p_[i]));
    return safety;
  }
};

/**
 * @brief Convex quadrilateral masks on plane surfaces.
 *
 * @tparam Real_t is data type for storing coordinates.
 */
template <typename Real_t>
struct QuadrilateralMask {
  Point2D<Real_t> p_[4] = {Real_t(0)}; ///< 2D coordinates of the vertices.
  Point2D<Real_t> n_[4] = {Real_t(0)}; ///< 2D coordinates of the outwards normals to segments

  QuadrilateralMask() = default;
  /**
   * @brief Construct a new Quadrilateral Mask object.
   *
   * @param x1 is the x coordinate of the lower left corner of the quadrilateral mask.
   * @param y1 is the y coordinate of the lower left corner of the quadrilateral mask.
   * @details The rest of the points should be entered counter-clockwise
   * starting from the lower left corner.
   */
  QuadrilateralMask(Real_t x1, Real_t y1, Real_t x2, Real_t y2, Real_t x3, Real_t y3, Real_t x4, Real_t y4)
  {
    p_[0].Set(x1, y1);
    p_[1].Set(x2, y2);
    p_[2].Set(x3, y3);
    p_[3].Set(x4, y4);

    // Compute outward normals
    for (int i = 0; i < 4; ++i) {
      auto j      = (i + 1) % 4;
      auto k      = (i + 2) % 4;
      auto seg_ij = p_[j] - p_[i];
      auto seg_ik = p_[k] - p_[i];
      assert(seg_ij.Mag2() > vecgeom::kToleranceSquared);
      // normal in XY plane
      n_[i].Set(seg_ij.y(), -seg_ij.x());
      // flip the normal so point k is 'backwards'
      if (n_[i].Dot(seg_ik) > Real_t(0)) n_[i] *= Real_t(-1);
      // Normalize normal vector
      n_[i].Normalize();
    }
  }

  /// @brief Returns the extent of the quadrilateral
  /// @param window Extent window to be filled
  void GetExtent(WindowMask<Real_t> &window) const
  {
    window.rangeU.Set(p_[0].x());
    window.rangeV.Set(p_[0].y());
    for (int i = 1; i < 4; ++i) {
      window.rangeU.Set(vecCore::math::Min(window.rangeU[0], p_[i].x()),
                        vecCore::math::Max(window.rangeU[1], p_[i].x()));
      window.rangeV.Set(vecCore::math::Min(window.rangeV[0], p_[i].y()),
                        vecCore::math::Max(window.rangeV[1], p_[i].y()));
    }
  }

  /// @brief Checks if the point is within mask.
  /// @details The point is within the quadrilateral if all dot products of point
  /// position relative to each vertex and corresponding segment normal are negative.
  /// @param local Local coordinates of the point
  /// @return true if the point is inside the mask.
  VECCORE_ATT_HOST_DEVICE
  bool Inside(Vector3D<Real_t> const &local) const
  {
    // TODO: Do we need a tolerance-aware version?
    Vector2D<Real_t> const local2D(local.x(), local.y());
    return (n_[0].Dot(local2D - p_[0]) < Real_t(0) && n_[1].Dot(local2D - p_[1]) < Real_t(0) &&
            n_[2].Dot(local2D - p_[2]) < Real_t(0) && n_[3].Dot(local2D - p_[3]) < Real_t(0));
  }

  /// @brief Computes the closest distance from a point in XY plane and the triangle.
  /// @param local Local coordinates of the point
  /// @return Safety from point to triangle.
  VECCORE_ATT_HOST_DEVICE
  Real_t Safety(Vector3D<Real_t> const &local, Real_t safetySurf, bool &valid) const
  {
    // The algorithm currently gives an underestimate
    valid = true;
    Vector2D<Real_t> const local2D(local.x(), local.y());
    // Compute signed safeties to segments
    Real_t safety = safetySurf;
    for (int i = 0; i < 4; ++i)
      safety = vecCore::math::Max(safety, n_[i].Dot(local2D - p_[i]));
    return safety;
  }
};

} // namespace vgbrep

#endif
