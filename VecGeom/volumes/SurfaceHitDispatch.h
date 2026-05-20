/// @file SurfaceHitDispatch.h
/// @brief Compile-time dispatch for optional surface-hit-aware kernels.

#ifndef VECGEOM_VOLUMES_SURFACEHITDISPATCH_H_
#define VECGEOM_VOLUMES_SURFACEHITDISPATCH_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/SurfaceHitView.h"

#include <type_traits>
#include <utility>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

namespace SurfaceHitDispatch {

template <typename Impl, typename Struct, typename Real_t, typename = void>
struct HasDistanceToInWithHit : std::false_type {};

template <typename Impl, typename Struct, typename Real_t>
struct HasDistanceToInWithHit<
    Impl, Struct, Real_t,
    std::void_t<decltype(Impl::DistanceToIn(std::declval<Struct const &>(), std::declval<Vector3D<Real_t> const &>(),
                                            std::declval<Vector3D<Real_t> const &>(), std::declval<Real_t const &>(),
                                            std::declval<Real_t &>(), std::declval<SurfaceHitView<Real_t> *>()))>>
    : std::true_type {};

template <typename Impl, typename Struct, typename Real_t, typename = void>
struct HasDistanceToOutWithHit : std::false_type {};

template <typename Impl, typename Struct, typename Real_t>
struct HasDistanceToOutWithHit<
    Impl, Struct, Real_t,
    std::void_t<decltype(Impl::DistanceToOut(std::declval<Struct const &>(), std::declval<Vector3D<Real_t> const &>(),
                                             std::declval<Vector3D<Real_t> const &>(), std::declval<Real_t const &>(),
                                             std::declval<Real_t &>(), std::declval<SurfaceHitView<Real_t> *>()))>>
    : std::true_type {};

/// @brief Call `DistanceToIn` with optional surface-hit output when supported.
template <typename Impl, typename Struct, typename Real_t>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void DistanceToIn(Struct const &shape, Vector3D<Real_t> const &point,
                                                               Vector3D<Real_t> const &dir, Real_t const &stepMax,
                                                               Real_t &distance, SurfaceHitView<Real_t> *hit_info)
{
  if constexpr (HasDistanceToInWithHit<Impl, Struct, Real_t>::value) {
    Impl::DistanceToIn(shape, point, dir, stepMax, distance, hit_info);
  } else {
    Impl::DistanceToIn(shape, point, dir, stepMax, distance);
    if (hit_info) hit_info->Clear();
  }
}

/// @brief Call `DistanceToOut` with optional surface-hit output when supported.
template <typename Impl, typename Struct, typename Real_t>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void DistanceToOut(Struct const &shape, Vector3D<Real_t> const &point,
                                                                Vector3D<Real_t> const &dir, Real_t const &stepMax,
                                                                Real_t &distance, SurfaceHitView<Real_t> *hit_info)
{
  if constexpr (HasDistanceToOutWithHit<Impl, Struct, Real_t>::value) {
    Impl::DistanceToOut(shape, point, dir, stepMax, distance, hit_info);
  } else {
    Impl::DistanceToOut(shape, point, dir, stepMax, distance);
    if (hit_info) hit_info->Clear();
  }
}

} // namespace SurfaceHitDispatch

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SURFACEHITDISPATCH_H_
