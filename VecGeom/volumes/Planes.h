/// \file Planes.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLANES_H_
#define VECGEOM_VOLUMES_PLANES_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Array.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"

#if defined(VECGEOM_VC) && defined(VECGEOM_QUADRILATERALS_VC)
#include <Vc/Vc>
typedef Vc::Vector<vecgeom::Precision> VcPrecision;
typedef Vc::Vector<vecgeom::Precision>::Mask VcBool;
constexpr int kVectorSize = VcPrecision::Size;
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class Planes;);
VECGEOM_DEVICE_DECLARE_CONV(class, Planes);

inline namespace VECGEOM_IMPL_NAMESPACE {

class Planes : public AlignedBase {

private:
  SOA3D<Precision> fNormals;   ///< Normalized normals of the planes.
  Array<Precision> fDistances; ///< Distance from plane to origin (0, 0, 0).
  bool fConvex{true};          ///< Convexity of the planes array (drives the inside reduction)

public:
  VECCORE_ATT_HOST_DEVICE
  Planes(int size, bool convex = true);

  VECCORE_ATT_HOST_DEVICE
  ~Planes();

  VECCORE_ATT_HOST_DEVICE
  Planes &operator=(Planes const &rhs);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision const *operator[](int index) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int size() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void reserve(size_t newsize)
  {
    fNormals.reserve(newsize);
    fDistances.Allocate(newsize);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  SOA3D<Precision> const &GetNormals() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> GetNormal(int i) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Array<Precision> const &GetDistances() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDistance(int i) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsConvex() const { return fConvex; }

  VECCORE_ATT_HOST_DEVICE
  void Set(int index, Vector3D<Precision> const &normal, Vector3D<Precision> const &origin);

  VECCORE_ATT_HOST_DEVICE
  void Set(int index, Vector3D<Precision> const &normal, Precision distance);

  /// Flip the sign of the normal and distance at the specified index
  VECCORE_ATT_HOST_DEVICE
  void FlipSign(int index);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE vecCore::Mask_v<Real_v> Contains(Vector3D<Real_v> const &point) const;

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Inside_v Inside(Vector3D<Real_v> const &point) const;

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Inside_v Inside(Vector3D<Real_v> const &point, int i) const;

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v Distance(Vector3D<Real_v> const &point,
                                                               Vector3D<Real_v> const &direction) const;

  template <bool pointInsideT, typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v Distance(Vector3D<Real_v> const &point) const;
};

std::ostream &operator<<(std::ostream &os, Planes const &planes);

VECCORE_ATT_HOST_DEVICE
Precision const *Planes::operator[](int i) const
{
  if (i == 0) return fNormals.x();
  if (i == 1) return fNormals.y();
  if (i == 2) return fNormals.z();
  if (i == 3) return &fDistances[0];
  return NULL;
}

VECCORE_ATT_HOST_DEVICE
int Planes::size() const
{
  return fNormals.size();
}

VECCORE_ATT_HOST_DEVICE
SOA3D<Precision> const &Planes::GetNormals() const
{
  return fNormals;
}

VECCORE_ATT_HOST_DEVICE
Vector3D<Precision> Planes::GetNormal(int i) const
{
  return fNormals[i];
}

VECCORE_ATT_HOST_DEVICE
Array<Precision> const &Planes::GetDistances() const
{
  return fDistances;
}

VECCORE_ATT_HOST_DEVICE
Precision Planes::GetDistance(int i) const
{
  return fDistances[i];
}

namespace {

template <typename Real_v, bool = true>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void AcceleratedContains(int & i, const int,
                                                                      SOA3D<Precision> const &,
                                                                      Array<Precision> const &,
                                                                      Vector3D<Real_v> const &,
                                                                      vecCore::Mask_v<Real_v> &)
{
  return;
}

#if defined(VECGEOM_VC) && defined(VECGEOM_QUADRILATERALS_VC)
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void AcceleratedContains<Precision, true>(
    int &i, const int n, SOA3D<Precision> const &normals, Array<Precision> const &distances,
    Vector3D<Precision> const &point, vecCore::Mask_v<double> &result)
{
  for (; i < n - kVectorSize; i += kVectorSize) {
    VcBool inside = VcPrecision(normals.x() + i) * point[0] + VcPrecision(normals.y() + i) * point[1] +
                        VcPrecision(normals.z() + i) * point[2] + VcPrecision(&distances[0] + i) <
                    0;
    // Early return if not inside all planes (convex case)
    result = vecCore::MaskFull(inside);
    if (!result) {
      i = n;
      break;
    }
  }
}

template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void AcceleratedContains<Precision, false>(
    int &i, const int n, SOA3D<Precision> const &normals, Array<Precision> const &distances,
    Vector3D<Precision> const &point, vecCore::Mask_v<double> &result)
{
  for (; i < n - kVectorSize; i += kVectorSize) {
    VcBool inside = VcPrecision(normals.x() + i) * point[0] + VcPrecision(normals.y() + i) * point[1] +
                        VcPrecision(normals.z() + i) * point[2] + VcPrecision(&distances[0] + i) <
                    0;
    // Early return ifinside any planes (non-convex case)
    result = !vecCore::MaskEmpty(inside);
    if (result) {
      i = n;
      break;
    }
  }
}
#endif

} // End anonymous namespace

template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE vecCore::Mask_v<Real_v> Planes::Contains(
    Vector3D<Real_v> const &point) const
{

  vecCore::Mask_v<Real_v> result = vecCore::Mask_v<Real_v>(true);

  int i       = 0;
  const int n = size();
  if (fConvex) {
    AcceleratedContains<Real_v, true>(i, n, fNormals, fDistances, point, result);
    for (; i < n; ++i) {
      result &= point.Dot(fNormals[i]) + fDistances[i] <= 0;
    }
  } else {
    AcceleratedContains<Real_v, false>(i, n, fNormals, fDistances, point, result);
    for (; i < n; ++i) {
      result |= point.Dot(fNormals[i]) + fDistances[i] <= 0;
    }
  }

  return result;
}

namespace {

template <typename Real_v, typename Inside_v, bool = true>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void AcceleratedInside(int & /*i*/, const int /*n*/,
                                                                    SOA3D<Precision> const & /*normals*/,
                                                                    Array<Precision> const & /*distances*/,
                                                                    Vector3D<Real_v> const & /*point*/,
                                                                    Inside_v & /*result*/)
{
  return;
}

#if defined(VECGEOM_VC) and defined(VECGEOM_QUADRILATERALS_VC)
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void AcceleratedInside<Precision, Inside_t, true>(
    int &i, const int n, SOA3D<Precision> const &normals, Array<Precision> const &distances,
    Vector3D<Precision> const &point, Inside_t &result)
{
  for (; i < n - kVectorSize; i += kVectorSize) {
    VcPrecision distance = VcPrecision(normals.x() + i) * point[0] + VcPrecision(normals.y() + i) * point[1] +
                           VcPrecision(normals.z() + i) * point[2] + VcPrecision(&distances[0] + i);
    // If point is outside tolerance of any plane, it is safe to return
    if (!vecCore::MaskEmpty(distance > kTolerance)) {
      result = EInside::kOutside;
      i      = n;
      break;
    }
    // If point is inside tolerance of all planes, keep looking
    if (vecCore::MaskFull(distance < -kTolerance)) continue;
    // Otherwise point must be on a surface, but could still be outside
    result = EInside::kSurface;
  }
}

template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void AcceleratedInside<Precision, Inside_t, false>(
    int &i, const int n, SOA3D<Precision> const &normals, Array<Precision> const &distances,
    Vector3D<Precision> const &point, Inside_t &result)
{
  for (; i < n - kVectorSize; i += kVectorSize) {
    VcPrecision distance = VcPrecision(normals.x() + i) * point[0] + VcPrecision(normals.y() + i) * point[1] +
                           VcPrecision(normals.z() + i) * point[2] + VcPrecision(&distances[0] + i);
    // If point is inside tolerance of any plane, it is safe to return
    if (!vecCore::MaskEmpty(distance < -kTolerance)) {
      result = EInside::kInside;
      i      = n;
      break;
    }
    // If point is outside tolerance of all planes, keep looking
    if (vecCore::MaskFull(distance > kTolerance)) continue;
    // Otherwise point must be on a surface, but could still be outside
    result = EInside::kSurface;
  }
}
#endif

} // End anonymous namespace

template <typename Real_v, typename Inside_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Inside_v Planes::Inside(Vector3D<Real_v> const &point) const
{
  Inside_v result = fConvex ? Inside_v(EInside::kInside) : Inside_v(EInside::kOutside);

  int i       = 0;
  const int n = size();
  if (fConvex) {
    AcceleratedInside<Real_v, Inside_v, true>(i, n, fNormals, fDistances, point, result);
    for (; i < n; ++i) {
      Real_v distanceResult =
          fNormals.x(i) * point[0] + fNormals.y(i) * point[1] + fNormals.z(i) * point[2] + fDistances[i];
      vecCore::MaskedAssign(result, distanceResult > Real_v(kTolerance), EInside::kOutside);
      vecCore::MaskedAssign(result, result == Inside_v(EInside::kInside) && distanceResult > Real_v(-kTolerance),
                            Inside_v(EInside::kSurface));
      if (vecCore::MaskFull(result == Inside_v(EInside::kOutside))) break;
    }
  } else {
    AcceleratedInside<Real_v, Inside_v, false>(i, n, fNormals, fDistances, point, result);
    for (; i < n; ++i) {
      Real_v distanceResult =
          fNormals.x(i) * point[0] + fNormals.y(i) * point[1] + fNormals.z(i) * point[2] + fDistances[i];
      vecCore::MaskedAssign(result, distanceResult < Real_v(-kTolerance), EInside::kInside);
      vecCore::MaskedAssign(result, result == Inside_v(EInside::kOutside) && distanceResult < Real_v(kTolerance),
                            Inside_v(EInside::kSurface));
      if (vecCore::MaskFull(result == Inside_v(EInside::kInside))) break;
    }
  }

  return result;
}

template <typename Real_v, typename Inside_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Inside_v Planes::Inside(Vector3D<Real_v> const &point, int i) const
{

  Inside_v result = Inside_v(EInside::kInside);
  Real_v distanceResult =
      fNormals.x(i) * point[0] + fNormals.y(i) * point[1] + fNormals.z(i) * point[2] + fDistances[i];
  vecCore::MaskedAssign(result, distanceResult > Real_v(kTolerance), Inside_v(EInside::kOutside));
  vecCore::MaskedAssign(result, result == Inside_v(EInside::kInside) && distanceResult > Real_v(-kTolerance),
                        Inside_v(EInside::kSurface));

  return result;
}

template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v Planes::Distance(Vector3D<Real_v> const &point,
                                                                     Vector3D<Real_v> const &direction) const
{

  Real_v bestDistance = InfinityLength<Real_v>();
  for (int i = 0, iMax = size(); i < iMax; ++i) {
    Vector3D<Precision> normal = fNormals[i];
    Real_v distance            = -(point.Dot(normal) + fDistances[i]) / direction.Dot(normal);
    vecCore::MaskedAssign(bestDistance, distance >= Real_v(0) && distance < bestDistance, distance);
  }

  return bestDistance;
}

template <bool pointInsideT, typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v Planes::Distance(Vector3D<Real_v> const &point) const
{

  Real_v bestDistance = InfinityLength<Real_v>();
  for (int i = 0, iMax = size(); i < iMax; ++i) {
    Real_v distance = Flip<!pointInsideT>::FlipSign(point.Dot(fNormals[i]) + fDistances[i]);
    vecCore::MaskedAssign(bestDistance, distance >= Real_v(0) && distance < bestDistance, distance);
  }
  return bestDistance;
}

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLANES_H_
