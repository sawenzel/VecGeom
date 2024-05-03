#ifndef VECGEOM_SURFACE_TRIANGLEMASK_H
#define VECGEOM_SURFACE_TRIANGLEMASK_H

#include <VecGeom/surfaces/base/CommonTypes.h>

namespace vgbrep {

/// @brief Triangular masks in the XY plane.
/// @tparam Real_t Precision type
template <typename Real_t>
struct TriangleMask {
  Point2D<Real_t> p_[3] = {Real_t(0)}; ///< 2D coordinates of the vertices.
  Point2D<Real_t> n_[3] = {Real_t(0)}; ///< 2D coordinates of the outwards normals to segments

  TriangleMask() = default;
  template <typename Real_i>
  TriangleMask(Real_i x1, Real_i y1, Real_i x2, Real_i y2, Real_i x3, Real_i y3)
  {
    p_[0].Set(static_cast<Real_t>(x1), static_cast<Real_t>(y1));
    p_[1].Set(static_cast<Real_t>(x2), static_cast<Real_t>(y2));
    p_[2].Set(static_cast<Real_t>(x3), static_cast<Real_t>(y3));

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

  /// @brief Fills the 3D extent of the quadrilateral
  /// @param window Extent window to be filled
  /// @param aMin Bottom extent corner
  /// @param aMax Top extent corner
  void Extent3D(Vector3D<Real_t> &aMin, Vector3D<Real_t> &aMax) const
  {
    aMin.Set(vecgeom::InfinityLength<Real_t>());
    aMax.Set(-vecgeom::InfinityLength<Real_t>());
    aMin.z() = aMax.z() = Real_t(0);
    for (int i = 0; i < 3; ++i) {
      aMin.x() = vecCore::math::Min(aMin.x(), p_[i].x());
      aMax.x() = vecCore::math::Max(aMax.x(), p_[i].x());
      aMin.y() = vecCore::math::Min(aMin.y(), p_[i].y());
      aMax.y() = vecCore::math::Max(aMax.y(), p_[i].y());
    }
  }

  /// @brief Returns the extent of the triangle
  /// @param window Extent window to be filled
  void GetExtent(WindowMask<Real_t> &window) const
  {
    window.rangeU.Set(p_[0].x());
    window.rangeV.Set(p_[0].y());
    for (int i = 0; i < 3; ++i) {
      window.rangeU.Set(vecCore::math::Min(window.rangeU[0], p_[i].x()),
                        vecCore::math::Max(window.rangeU[1], p_[i].x()));
      window.rangeV.Set(vecCore::math::Min(window.rangeV[0], p_[i].y()),
                        vecCore::math::Max(window.rangeV[1], p_[i].y()));
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
    Vector2D<Real_t> const local2D(local.x(), local.y());
    return (n_[0].Dot(local2D - p_[0]) < vecgeom::kToleranceDist<Real_t> &&
            n_[1].Dot(local2D - p_[1]) < vecgeom::kToleranceDist<Real_t> &&
            n_[2].Dot(local2D - p_[2]) < vecgeom::kToleranceDist<Real_t>);
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

    // lambda to compute distance to segment i (common to triangles and quads, to be moved?)
    auto distanceToSegmentSquared = [&](int i) {
      int j     = (i + 1) % 3;
      auto line = p_[j] - p_[i];
      auto pvec = local2D - p_[i];
      auto dot0 = line.Dot(pvec);
      if (dot0 <= 0) return pvec.Mag2();
      auto dot1 = line.Mag2();
      if (dot1 <= dot0) return (local2D - p_[j]).Mag2();
      return ((dot0 / dot1) * line - pvec).Mag2();
    };

    bool withinBound[3];
    for (int i = 0; i < 3; ++i) {
      withinBound[i] = n_[i].Dot(local2D - p_[i]) <= 0;
    }
    if (withinBound[0] && withinBound[1] && withinBound[2]) return safetySurf;

    Real_t dseg_squared = vecgeom::InfinityLength<Real_t>();
    for (int i = 0; i < 3; ++i) {
      if (!withinBound[i]) {
        dseg_squared = vecCore::math::Min(dseg_squared, distanceToSegmentSquared(i));
      }
    }
    safety = vecCore::math::Sqrt(dseg_squared + safetySurf * safetySurf);

    return safety;
  }

  /// @brief Safe distance from a point assumed inside the window
  /// @details Used on host only for frame checks
  /// @param local Projected point in local coordinates
  /// @return Safe distance
  Real_t SafetyInside(Vector3D<Real_t> const &local) const
  {
    if (Inside(local) == false) return -vecgeom::InfinityLength<Real_t>();

    Vector2D<Real_t> const local2D(local.x(), local.y());

    // lambda to compute distance to segment i (common to triangles and quads, to be moved?)
    auto distanceToSegmentSquared = [&](int i) {
      int j     = (i + 1) % 3;
      auto line = p_[j] - p_[i];
      auto pvec = local2D - p_[i];
      auto dot0 = line.Dot(pvec);
      if (dot0 <= 0) return pvec.Mag2();
      auto dot1 = line.Mag2();
      if (dot1 <= dot0) return (local2D - p_[j]).Mag2();
      return ((dot0 / dot1) * line - pvec).Mag2();
    };

    Real_t safety_squared = vecgeom::InfinityLength<Real_t>();
    for (int i = 0; i < 3; ++i) {
      safety_squared = vecCore::math::Min(safety_squared, distanceToSegmentSquared(i));
    }
    return vecCore::math::Sqrt(safety_squared);
  }
};

} // namespace vgbrep

#endif
