/// @file SurfaceHitView.h
/// @brief Optional surface hit information returned by distance queries.

#ifndef VECGEOM_VOLUMES_SURFACEHITVIEW_H_
#define VECGEOM_VOLUMES_SURFACEHITVIEW_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"

#include <cstdint>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

using SurfaceCode = uint32_t; ///< Shape-local opaque surface identifier.

inline constexpr SurfaceCode kNoSurfaceCode = 0; ///< No surface information was provided.

/// @brief Optional surface information associated with a scalar distance result.
/// @details The surface code is owned by the view and opaque outside the shape
/// family producing it. Normal storage is supplied by the caller only when
/// needed, avoiding the cost of reserving three coordinates for surface-only
/// queries. Safety queries intentionally do not use this view because their
/// closest surface is not a boundary hit in the tracking direction.
template <typename Real_t = Precision>
struct SurfaceHitView {
  SurfaceCode fSurface      = kNoSurfaceCode; ///< Opaque shape-local surface code for the returned value.
  Vector3D<Real_t> *fNormal = nullptr;        ///< Optional destination for the outward normal in query-local frame.

  /// @brief Check whether normal output was requested.
  /// @return True when `fNormal` points to writable storage.
  VECCORE_ATT_HOST_DEVICE
  bool WantsNormal() const { return fNormal != nullptr; }

  /// @brief Clear outputs to the no-information state.
  VECCORE_ATT_HOST_DEVICE
  void Clear()
  {
    fSurface = kNoSurfaceCode;
    if (fNormal) fNormal->Set(Real_t(0));
  }

  /// @brief Store a shape-local opaque surface code.
  /// @param surface Shape-local code owned by the producing solid.
  VECCORE_ATT_HOST_DEVICE
  void SetSurface(SurfaceCode surface) { fSurface = surface; }

  /// @brief Store an outward normal if requested.
  /// @param normal Outward normal in the query-local coordinate frame.
  VECCORE_ATT_HOST_DEVICE
  void SetNormal(Vector3D<Real_t> const &normal) const
  {
    if (fNormal) *fNormal = normal;
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SURFACEHITVIEW_H_
