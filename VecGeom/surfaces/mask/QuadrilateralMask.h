#ifndef VECGEOM_SURFACE_QUADMASK_H
#define VECGEOM_SURFACE_QUADMASK_H

#include <VecGeom/surfaces/base/CommonTypes.h>

namespace vgbrep {

/// @brief Convex quadrilateral masks on plane surfaces.
/// @tparam Real_t Precision type
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
    valid = true;
    Vector2D<Real_t> const local2D(local.x(), local.y());
    Real_t safety = safetySurf;
#ifdef QUAD_ACCURATE_SAFETY
    // lambda to compute distance to segment i
    auto distanceToSegmentSquared = [&](int i) {
      int j     = (i + 1) % 4;
      auto line = p_[j] - p_[i];
      auto pvec = local2D - p_[i];
      auto dot0 = line.Dot(pvec);
      if (dot0 <= 0) return pvec.Mag2();
      auto dot1 = line.Mag2();
      if (dot1 <= dot0) return (local2D - p_[j]).Mag2();
      return ((dot0 / dot1) * line - pvec).Mag2();
    };

    bool withinBound[4];
    for (int i = 0; i < 4; ++i) {
      withinBound[i] = n_[i].Dot(local2D - p_[i]) <= 0;
    }
    if (withinBound[0] && withinBound[1] && withinBound[2] && withinBound[3]) return safetySurf;

    Precision dseg_squared = vecgeom::InfinityLength<Real_t>();
    for (int i = 0; i < 4; ++i) {
      if (!withinBound[i]) {
        dseg_squared = vecCore::math::Min(dseg_squared, distanceToSegmentSquared(i));
      }
    }
    safety = vecCore::math::Sqrt(dseg_squared + safetySurf * safetySurf);
#else
    // Compute signed safeties to segments
    for (int i = 0; i < 4; ++i)
      safety = vecCore::math::Max(safety, n_[i].Dot(local2D - p_[i]));
#endif
    return safety;
  }
};

} // namespace vgbrep

#endif
