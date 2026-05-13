/// @file CoaxialConesImplementation.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_COAXIALCONESIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_COAXIALCONESIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/CoaxialConesStruct.h"
#include "VecGeom/volumes/kernel/ConeImplementation.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct CoaxialConesImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, CoaxialConesImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedCoaxialCones;
template <typename T>
struct CoaxialConesStruct;
class UnplacedCoaxialCones;

struct CoaxialConesImplementation {

  using PlacedShape_t    = PlacedCoaxialCones;
  using UnplacedStruct_t = CoaxialConesStruct<Precision>;
  using UnplacedVolume_t = UnplacedCoaxialCones;

  template <typename Real_v, bool ForLowerZ>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOnRing(UnplacedStruct_t const &coaxialcones,
                                                                    Vector3D<Real_v> const &point)
  {
    for (unsigned int i = 0; i < coaxialcones.fConeStructVector.size(); i++) {
      if (ConeImplementation<ConeTypes::UniversalCone>::template IsOnRing<Real_v, true, ForLowerZ>(
              *coaxialcones.fConeStructVector[i], point) ||
          ConeImplementation<ConeTypes::UniversalCone>::template IsOnRing<Real_v, false, ForLowerZ>(
              *coaxialcones.fConeStructVector[i], point)) {
        return true;
      }
    }

    return false;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &coaxialcones,
                                                                    Vector3D<Real_v> const &point, bool &inside)
  {
    bool unused(false), outside(false);
    GenericKernelForContainsAndInside<Real_v, false>(coaxialcones, point, unused, outside);
    inside = !outside;
  }

  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &coaxialcones,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    bool completelyinside(false), completelyoutside(false);
    GenericKernelForContainsAndInside<Real_v, true>(coaxialcones, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    if (completelyoutside) inside = EInside::kOutside;
    if (completelyinside) inside = EInside::kInside;
  }

  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &coaxialcones, Vector3D<Real_v> const &point, bool &completelyinside,
      bool &completelyoutside)
  {
    /* TODO : Logic to check where the point is inside or not.
    **
    ** if ForInside is false then it will only check if the point is outside,
    ** and is used by Contains function
    **
    ** if ForInside is true then it will check whether the point is inside or outside,
    ** and if neither inside nor outside then it is on the surface.
    ** and is used by Inside function
    */
    completelyinside  = false;
    completelyoutside = true;

    for (unsigned int i = 0; i < coaxialcones.fConeStructVector.size(); i++) {
      bool compIn(false);
      bool compOut(false);

      ConeHelpers<Real_v, ConeTypes::UniversalCone>::template GenericKernelForContainsAndInside<ForInside>(
          *coaxialcones.fConeStructVector[i], point, compIn, compOut);
      if (ForInside) {
        if (compIn) {
          completelyinside  = true;
          completelyoutside = false;
          return;
        }

        if (!compOut) {
          completelyoutside = false;
          return;
        }
      } else {

        completelyoutside &= compOut;
      }
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &coaxialcones,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    /* TODO :  Logic to calculate Distance from outside point to the CoaxialCones surface */

    distance = kInfLength;
    for (unsigned int i = 0; i < coaxialcones.fConeStructVector.size(); i++) {
      Real_v dist(kInfLength);
      ConeImplementation<ConeTypes::UniversalCone>::template DistanceToIn<Real_v>(*coaxialcones.fConeStructVector[i],
                                                                                  point, direction, stepMax, dist);

      if (dist < distance) distance = dist;
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &coaxialcones,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    /* TODO :  Logic to calculate Distance from inside point to the CoaxialCones surface */
    distance = -1.;
    for (unsigned int i = 0; i < coaxialcones.fConeStructVector.size(); i++) {
      Real_v dist(kInfLength);
      ConeImplementation<ConeTypes::UniversalCone>::template DistanceToOut<Real_v>(*coaxialcones.fConeStructVector[i],
                                                                                   point, direction, stepMax, dist);

      if (dist > distance) distance = dist;
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &coaxialcones,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    /* TODO :  Logic to calculate Safety from outside point to the CoaxialCones surface */
    safety = kInfLength;
    for (unsigned int i = 0; i < coaxialcones.fConeStructVector.size(); i++) {
      Real_v safeDist(kInfLength);
      ConeImplementation<ConeTypes::UniversalCone>::template SafetyToIn<Real_v>(*coaxialcones.fConeStructVector[i],
                                                                                point, safeDist);

      if (safeDist < safety) safety = safeDist;
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &coaxialcones,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    /* TODO :  Logic to calculate Safety from inside point to the CoaxialCones surface */
    safety = Real_v(-1.);
    for (unsigned int i = 0; i < coaxialcones.fConeStructVector.size(); i++) {
      Real_v safeDist(kInfLength);
      ConeImplementation<ConeTypes::UniversalCone>::template SafetyToOut<Real_v>(*coaxialcones.fConeStructVector[i],
                                                                                 point, safeDist);

      if (safeDist > safety) safety = safeDist;
    }
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_COAXIALCONESIMPLEMENTATION_H_
