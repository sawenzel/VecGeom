#ifndef VECGEOM_SURFACE_QUADMASK_H
#define VECGEOM_SURFACE_QUADMASK_H

#ifndef QUAD_ACCURATE_SAFETY
#define QUAD_ACCURATE_SAFETY 0
#endif

#include <VecGeom/surfaces/base/CommonTypes.h>

namespace vgbrep {

/// @brief Convex quadrilateral masks on plane surfaces.
/// @tparam Real_t Precision type
template <typename Real_t>
struct QuadrilateralMask {
  using value_type = Real_t;
  Point2D<Real_t> p_[4] = {Real_t(0)}; ///< 2D coordinates of the vertices.
  Point2D<Real_t> n_[4] = {Real_t(0)}; ///< 2D coordinates of the outwards normals to segments

  QuadrilateralMask() = default;
  /**
   * @brief Construct a new Quadrilateral Mask object.
   *
   * @tparam Real_i Precision type of inputs
   * @param x1 is the x coordinate of the lower left corner of the quadrilateral mask.
   * @param y1 is the y coordinate of the lower left corner of the quadrilateral mask.
   * @details The rest of the points should be entered counter-clockwise
   * starting from the lower left corner.
   */
  template <typename Real_i>
  QuadrilateralMask(Real_i x1, Real_i y1, Real_i x2, Real_i y2, Real_i x3, Real_i y3, Real_i x4, Real_i y4)
  {
    // temporary points and normals in vecgeom Precision for calculation and degeneracy check
    Point2D<vecgeom::Precision> p[4] = {vecgeom::Precision(0)};
    Point2D<vecgeom::Precision> n[4] = {vecgeom::Precision(0)};
    p[0].Set(x1, y1);
    p[1].Set(x2, y2);
    p[2].Set(x3, y3);
    p[3].Set(x4, y4);

    // Compute outward normals
    for (int i = 0; i < 4; ++i) {
      auto j      = (i + 1) % 4;
      auto k      = (i + 2) % 4;
      auto seg_ij = p[j] - p[i];
      auto seg_ik = p[k] - p[i];
      VECGEOM_ASSERT(seg_ij.Mag2() >
                     vecgeom::kToleranceSquared); // use vecgeom tolerance since p is in vecgeom precision
      // normal in XY plane
      n[i].Set(seg_ij.y(), -seg_ij.x());
      // flip the normal so point k is 'backwards'
      if (n[i].Dot(seg_ik) > Real_t(0)) n[i] *= Real_t(-1);
      // Normalize normal vector
      n[i].Normalize();
      // set stored normal n_ in mixed precision
      n_[i].Set(static_cast<Real_t>(n[i].x()), static_cast<Real_t>(n[i].y()));
    }
    // static cast to mixed precision after normal calculation
    p_[0].Set(static_cast<Real_t>(x1), static_cast<Real_t>(y1));
    p_[1].Set(static_cast<Real_t>(x2), static_cast<Real_t>(y2));
    p_[2].Set(static_cast<Real_t>(x3), static_cast<Real_t>(y3));
    p_[3].Set(static_cast<Real_t>(x4), static_cast<Real_t>(y4));
  }
  template <typename Real_i>
  QuadrilateralMask(const QuadrilateralMask<Real_i> &other)
  {
    for (int i = 0; i < 4; ++i) {
      p_[i] = Point2D<Real_t>(other.p_[i]);
      n_[i] = Point2D<Real_t>(other.n_[i]);
    }
  }

  /// @brief Fills the 3D extent of the quadrilateral
  /// @param window Extent window to be filled
  /// @param aMin Bottom extent corner
  /// @param aMax Top extent corner
  void Extent3D(Vector3D<Real_t> &aMin, Vector3D<Real_t> &aMax) const
  {
    aMin.Set(vecgeom::InfinityLength<Real_t>());
    aMax.Set(-vecgeom::InfinityLength<Real_t>());
    aMin.z() = aMax.z() = Real_t(0);
    for (int i = 0; i < 4; ++i) {
      aMin.x() = vecCore::math::Min(aMin.x(), p_[i].x());
      aMax.x() = vecCore::math::Max(aMax.x(), p_[i].x());
      aMin.y() = vecCore::math::Min(aMin.y(), p_[i].y());
      aMax.y() = vecCore::math::Max(aMax.y(), p_[i].y());
    }
  }

  /// @brief Returns the extent of the quadrilateral
  /// @param window Extent window to be filled
  void GetExtent(WindowMask<Real_t> &window) const
  {
    window.rangeU.Set(p_[0].x());
    window.rangeV.Set(p_[0].y());
    for (int i = 0; i < 4; ++i) {
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
    Vector2D<Real_t> const local2D(local.x(), local.y());
    return (n_[0].Dot(local2D - p_[0]) < vecgeom::kToleranceDist<Real_t> &&
            n_[1].Dot(local2D - p_[1]) < vecgeom::kToleranceDist<Real_t> &&
            n_[2].Dot(local2D - p_[2]) < vecgeom::kToleranceDist<Real_t> &&
            n_[3].Dot(local2D - p_[3]) < vecgeom::kToleranceDist<Real_t>);
  }

  /// @brief Computes the closest distance from a point in XY plane and the triangle.
  /// @param local Local coordinates of the point
  /// @return Safety from point to triangle.
  VECCORE_ATT_HOST_DEVICE
  Real_t Safety(Vector3D<Real_t> const &local, Real_t safetySurf, bool &valid) const
  {
    valid = true;
    Vector2D<Real_t> const local2D(local.x(), local.y());
    Real_t safety = safetySurf;
    bool withinBound[4];
    for (int i = 0; i < 4; ++i) {
      withinBound[i] = n_[i].Dot(local2D - p_[i]) <= 0;
    }
    if (withinBound[0] && withinBound[1] && withinBound[2] && withinBound[3]) return safetySurf;

    Real_t dseg_squared = vecgeom::InfinityLength<Real_t>();
    for (int i = 0; i < 4; ++i) {
      if (!withinBound[i])
        dseg_squared = vecCore::math::Min(dseg_squared, DistanceToSegmentSquared(local2D, p_[i], p_[(i + 1) % 4]));
    }
    safety = vecCore::math::Sqrt(dseg_squared + safetySurf * safetySurf);

    return safety;
  }

  /// @brief Safe distance from a point assumed inside the quadrilateral
  /// @details Used on host only for frame checks
  /// @param local Projected point in local coordinates
  /// @return Safe distance
  Real_t SafetyInside(Vector3D<Real_t> const &local) const
  {
    if (Inside(local) == false) return Real_t(0);
    Vector2D<Real_t> const local2D(local.x(), local.y());
    Real_t safety_squared = vecgeom::InfinityLength<Real_t>();
    for (int i = 0; i < 4; ++i) {
      safety_squared = vecCore::math::Min(safety_squared, DistanceToSegmentSquared(local2D, p_[i], p_[(i + 1) % 4]));
    }
    return vecCore::math::Sqrt(safety_squared);
  }

  /// @brief Safe distance from a point assumed outside the quadrilateral
  /// @details Used on host only for frame checks
  /// @param local Projected point in local coordinates
  /// @return Safe distance
  Real_t SafetyOutside(Vector3D<Real_t> const &local) const
  {
    if (Inside(local) == true) return Real_t(0);
    Vector2D<Real_t> const local2D(local.x(), local.y());
    Real_t safety_squared = vecgeom::InfinityLength<Real_t>();
    for (int i = 0; i < 3; ++i) {
      safety_squared = vecCore::math::Min(safety_squared, DistanceToSegmentSquared(local2D, p_[i], p_[(i + 1) % 4]));
    }
    return vecCore::math::Sqrt(safety_squared);
  }
};

} // namespace vgbrep

#endif
