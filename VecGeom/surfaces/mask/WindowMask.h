#ifndef VECGEOM_SURFACE_WINDOWMASK_H
#define VECGEOM_SURFACE_WINDOWMASK_H

#ifndef WINDOW_ACCURATE_SAFETY
#define WINDOW_ACCURATE_SAFETY 0
#endif

#include <VecGeom/surfaces/base/CommonTypes.h>

namespace vgbrep {

/// @brief Rectangular masks on plane surfaces.
/// @tparam Real_t Precision type
template <typename Real_t>
struct WindowMask {
  Range<Real_t> rangeU; ///< Rectangle limits on x axis.
  Range<Real_t> rangeV; ///< Rectangle limits on y axis.

  WindowMask() = default;
  WindowMask(Real_t u1, Real_t u2, Real_t v1, Real_t v2) : rangeU(u1, u2), rangeV(v1, v2){};
  WindowMask(Real_t u, Real_t v) : rangeU(-u, u), rangeV(-v, v){};

  /// @brief Fills the 3D extent of the window
  /// @param aMin Bottom extent corner
  /// @param aMax Top extent corner
  void Extent3D(Vector3D<Real_t> &aMin, Vector3D<Real_t> &aMax) const
  {
    aMin.Set(rangeU[0], rangeV[0], Real_t(0));
    aMax.Set(rangeU[1], rangeV[1], Real_t(0));
  }

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
    if (WINDOW_ACCURATE_SAFETY > 0) {
      // The following returns the accurate safety at the price of an extra square root per frame
      // meaning 6 for a box.
      sx = vecCore::math::Max(Real_t(0), sx);
      sy = vecCore::math::Max(Real_t(0), sy);
      return vecCore::math::Sqrt(sx * sx + sy * sy + safetySurf * safetySurf);
    } else {
      // The VecGeom box version just returns the maximum
      return vecCore::math::Max(sx, sy, safetySurf);
    }
  }
};

} // namespace vgbrep

#endif
