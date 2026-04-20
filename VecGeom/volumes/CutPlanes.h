/// \file CutPlanes.h
/// \author mgheata

#ifndef VECGEOM_VOLUMES_CUTPLANES_H_
#define VECGEOM_VOLUMES_CUTPLANES_H_

#include "VecGeom/volumes/Plane.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class CutPlanes;);
VECGEOM_DEVICE_DECLARE_CONV(class, CutPlanes);

inline namespace VECGEOM_IMPL_NAMESPACE {

class CutPlanes : public AlignedBase {

private:
  Plane fCutPlanes[2]; ///< Two cut planes

public:
  VECCORE_ATT_HOST_DEVICE
  CutPlanes() {}

  VECCORE_ATT_HOST_DEVICE
  ~CutPlanes() {}

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Plane const &GetCutPlane(int i) const;

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GetNormal(int i) const;

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Precision GetDistance(int i) const;

  VECCORE_ATT_HOST_DEVICE
  void Set(int index, Vector3D<Precision> const &normal, Vector3D<Precision> const &origin);

  VECCORE_ATT_HOST_DEVICE
  void Set(int index, Vector3D<Precision> const &normal, Precision distance);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Contains(Vector3D<Real_v> const &point, bool &inside) const;

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Inside(Vector3D<Real_v> const &point, Inside_t &inside) const;

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void DistanceToIn(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, Real_v &distance) const;

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, Real_v &distance) const;

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SafetyToIn(Vector3D<Real_v> const &point, Real_v &distance) const;

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SafetyToOut(Vector3D<Real_v> const &point, Real_v &distance) const;
}; // class CutPlanes

std::ostream &operator<<(std::ostream &os, CutPlanes const &planes);

//______________________________________________________________________________
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
Plane const &CutPlanes::GetCutPlane(int i) const
{
  return fCutPlanes[i];
}

//______________________________________________________________________________
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
Vector3D<Precision> CutPlanes::GetNormal(int i) const
{
  return fCutPlanes[i].GetNormal();
}

//______________________________________________________________________________
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
Precision CutPlanes::GetDistance(int i) const
{
  return fCutPlanes[i].GetDistance();
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
void CutPlanes::Contains(Vector3D<Real_v> const &point, bool &inside) const
{
  inside = fCutPlanes[0].DistPlane(point) < Real_v(0.) && fCutPlanes[1].DistPlane(point) < Real_v(0.);
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
void CutPlanes::Inside(Vector3D<Real_v> const &point, Inside_t &inside) const
{
  Real_v d0 = fCutPlanes[0].DistPlane(point);
  Real_v d1 = fCutPlanes[1].DistPlane(point);

  inside = (d0 < Real_v(0.0) && d1 < Real_v(0.0)) ? EInside::kInside : EInside::kOutside;
  if (vecCore::math::Abs(d0) < Real_v(kTolerance) || vecCore::math::Abs(d1) < Real_v(kTolerance)) {
    inside = EInside::kSurface;
  }
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
void CutPlanes::DistanceToIn(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, Real_v &distance) const
{
  // The function returns a negative distance for points coming from the
  // region in between the two planes, or from points outside going outwards
  // If a particle has valid crossings withe the 2 planes, the maximum distance
  // has to be taken
  Real_v d0, d1;
  fCutPlanes[0].DistanceToIn(point, direction, d0);
  if (direction.Dot(Vector3D<Real_v>(fCutPlanes[0].GetNormal())) > Real_v(0.)) d0 = Real_v(-kInfLength);
  fCutPlanes[1].DistanceToIn(point, direction, d1);
  if (direction.Dot(Vector3D<Real_v>(fCutPlanes[1].GetNormal())) > Real_v(0.)) d1 = Real_v(-kInfLength);
  distance = vecCore::math::Max(d0, d1);
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
void CutPlanes::DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &direction, Real_v &distance) const
{
  Real_v d0, d1;
  fCutPlanes[0].DistanceToOut(point, direction, d0);
  fCutPlanes[1].DistanceToOut(point, direction, d1);
  distance = vecCore::math::Min(d0, d1);
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
void CutPlanes::SafetyToIn(Vector3D<Real_v> const &point, Real_v &distance) const
{
  Real_v d0, d1;
  fCutPlanes[0].SafetyToIn(point, d0);
  fCutPlanes[1].SafetyToIn(point, d1);
  distance = vecCore::math::Max(d0, d1);
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
void CutPlanes::SafetyToOut(Vector3D<Real_v> const &point, Real_v &distance) const
{
  Real_v d0, d1;
  fCutPlanes[0].SafetyToOut(point, d0);
  fCutPlanes[1].SafetyToOut(point, d1);
  distance = vecCore::math::Min(d0, d1);
}

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom

#endif // VECGEOM_VOLUMES_CUTPLANES_H_
