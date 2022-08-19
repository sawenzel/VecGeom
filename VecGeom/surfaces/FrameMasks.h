#ifndef VECGEOM_SURFACE_FRAMEMASKS_H
#define VECGEOM_SURFACE_FRAMEMASKS_H

namespace vgbrep {

/**
 * @brief Vector in 2D plane.
 *
 * @tparam Real_t is a type parameter of vector elements.
 */
template <typename Real_t>
struct Vec2D {
  Real_t components[2]{0};

  Vec2D() = default;
  Vec2D(Real_t x1, Real_t x2)
  {
    components[0] = x1;
    components[1] = x2;
  }
  Real_t operator[](int i) const { return components[i]; }
  Real_t &operator[](int i) { return components[i]; }
  void Set(Real_t min, Real_t max)
  {
    components[0] = min;
    components[1] = max;
  }

  /**
   * @brief Calculates the z-component of cross product of two 2D vectors in xy plane.
   *
   * @param other The second vector in cross product.
   * @return Real_t z-component of cross product, signed.
   */
  Real_t crossProd2D(Vec2D<Real_t> other) const { return components[0] * other[1] - components[1] * other[0]; };
};

// Aliases for different usages of Vec2D.
template <typename Real_t>
using Range = Vec2D<Real_t>;

template <typename Real_t>
using AngleVector = Vec2D<Real_t>;

template <typename Real_t>
using Point2D = Vec2D<Real_t>;

//
//  Masks for different types of frames
//
/// TODO: Implement other masks, too.

/**
 * @brief Rectangular masks on plane surfaces.
 *
 * @tparam Real_t Scalar type for representing rectangle limit on axis.
 */
template <typename Real_t>
struct WindowMask {
  Range<Real_t> rangeU; ///< Rectangle limits on x axis.
  Range<Real_t> rangeV; ///< Rectangle limits on y axis.

  WindowMask() = default;
  WindowMask(Real_t u1, Real_t u2, Real_t v1, Real_t v2) : rangeU(u1, u2), rangeV(v1, v2){};
  WindowMask(Real_t u, Real_t v) : rangeU(-u, u), rangeV(-v, v){};

  void GetMask(WindowMask<Real_t> &mask)
  {
    mask.rangeU.Set(rangeU[0], rangeU[1]);
    mask.rangeV.Set(rangeV[0], rangeV[1]);
  }

  bool Inside(Vector3D<Real_t> const &local) const
  {
    return (local[0] > vecgeom::MakeMinusTolerant<true>(rangeU[0]) &&
            local[0] < vecgeom::MakePlusTolerant<true>(rangeU[1]) &&
            local[1] > vecgeom::MakeMinusTolerant<true>(rangeV[0]) &&
            local[1] < vecgeom::MakePlusTolerant<true>(rangeV[1]));
  }
};

/**
 * @brief Ring masks on plane surfaces.
 *
 * @tparam Real_t
 */
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

  void GetMask(RingMask<Real_t> &mask)
  {
    mask.rangeR.Set(rangeR[0], rangeR[1]);
    mask.isFullCirc = isFullCirc;
    if (isFullCirc) return;
    mask.vecSPhi.Set(vecSPhi[0], vecSPhi[1]);
    mask.vecEPhi.Set(vecEPhi[0], vecEPhi[1]);
  }

  /**
   * @brief Checks if the point is within mask.
   * @details In barycentric coordinate system, where the base vectors are xaxis and vvec,
   *  the point lies in the convex part of the plane if both of its coordinates are
   *  greater than zero. If vectors that delimit the phi-cut form an angle less than
   *  180 degrees, all the points in the convex part of the plane are inside. Otherwise,
   *  if phi-cut is greater than 180 degrees, all the points in the concave part are inside.
   *  The point must also satisfy the radius limits.
   *
   * @param local Local coordinates of the point
   * @return true if the point is inside the mask.
   * @return false if the point is outside the mask.
   */
  bool Inside(Vector3D<Real_t> const &local) const
  {
    Real_t rsq = local[0] * local[0] + local[1] * local[1];

    // The point must be inside the ring:
    if ((rsq < rangeR[0] * rangeR[0] + 2 * vecgeom::kToleranceSquared * rangeR[0]) ||
        (rsq > rangeR[1] * rangeR[1] - 2 * vecgeom::kToleranceSquared * rangeR[1]))
      return false;

    // If it's a full circle:
    if (isFullCirc) return true;

    //

    AngleVector<Real_t> localAngle{local[0], local[1]};
    auto sysdet  = vecSPhi.crossProd2D(vecEPhi);
    auto divider = 1 / sysdet;
    auto d1      = localAngle.crossProd2D(vecEPhi) * divider;
    auto d2      = vecSPhi.crossProd2D(localAngle) * divider;
    // If limiting vectors are close, we want convex solutions, and concave otherwise.
    bool convexity = (sysdet > 0);

    // TODO: Check these tolerances.
    return (d1 > -vecgeom::kTolerance && d2 > -vecgeom::kTolerance) == convexity;
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

  /**
   * @brief Checks if the point is within mask.
   * @details In barycentric coordinate system, where the base vectors are xaxis and vvec,
   *  the point lies in the convex part of the plane if both of its coordinates are
   *  greater than zero. If vectors that delimit the phi-cut form an angle less than
   *  180 degrees, all the points in the convex part of the plane are inside. Otherwise,
   *  if phi-cut is greater than 180 degrees, all the points in the concave part are inside.
   *  The point must also satisfy the z limits.
   *
   * @param local Local coordinates of the point
   * @return true if the point is inside the mask.
   * @return false if the point is outside the mask.
   */
  bool Inside(Vector3D<Real_t> const &local) const
  {
    // The point must be inside z-span:
    if (local[2] < rangeZ[0] - vecgeom::kTolerance || local[2] > rangeZ[1] + vecgeom::kTolerance) return false;

    // If it's a full circle:
    if (isFullCirc) return true;

    AngleVector<Real_t> localAngle{local[0], local[1]};
    auto sysdet  = vecSPhi.crossProd2D(vecEPhi);
    auto divider = 1 / sysdet;
    auto d1      = localAngle.crossProd2D(vecEPhi) * divider;
    auto d2      = vecSPhi.crossProd2D(localAngle) * divider;
    // If limiting vectors are close, we want convex solutions, and concave otherwise.
    bool convexity = (sysdet > 0);

    // TODO: Check these tolerances.
    return (d1 > -vecgeom::kTolerance && d2 > -vecgeom::kTolerance) == convexity;
  }
};

/**
 * @brief Triangular masks on plane surfaces.
 *
 * @tparam Real_t
 */
template <typename Real_t>
struct TriangleMask {
  // TODO: Check if the points themselves are useful
  Point2D<Real_t> p1, p2, p3; ///< 2D coordinates of the triangle vertices.
  Point2D<Real_t> bp2, bp3;   ///< 2D coordinates of vertices 2 and 3 relative to vertex 1.

  TriangleMask() = default;
  TriangleMask(Real_t x1, Real_t y1, Real_t x2, Real_t y2, Real_t x3, Real_t y3)
      : p1(x1, y1), p2(x2, y2), p3(x3, y3), bp2(x2 - x1, y2 - y1), bp3(x3 - x1, y3 - y1){};

  void GetMask(TriangleMask<Real_t> &mask)
  {
    mask.p1.Set(p1[0], p1[1]);
    mask.p2.Set(p2[0], p2[1]);
    mask.p3.Set(p3[0], p3[1]);
    mask.bp2.Set(bp2[0], bp2[1]);
    mask.bp3.Set(bp3[0], bp3[1]);
  }

  /**
   * @brief Checks if the point is within mask.
   * @details The point is within the triangle if its barymetric coordinates d1, d2
   *  satisfy (d1 > 0 && d2 > 0 && d1 + d2 < 1).
   *
   * @param local Local coordinates of the point
   * @return true if the point is inside the mask.
   * @return false if the point is outside the mask.
   */
  bool Inside(Vector3D<Real_t> const &local) const
  {
    // Barycentric coordinates with respect to triangle:
    Point2D<Real_t> blocal{local[0] - p1[0], local[1] - p1[1]};

    auto divider = 1 / bp2.crossProd2D(bp3);
    auto d1      = blocal.crossProd2D(bp3) * divider;
    auto d2      = bp2.crossProd2D(blocal) * divider;

    // TODO: Tolerances.
    return (d1 > 0 && d2 > 0 && d1 + d2 < 1);
  }
};
} // namespace vgbrep

#endif