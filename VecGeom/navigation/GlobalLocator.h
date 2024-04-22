/*
 * GlobalLocator.h
 *
 *  Created on: Aug 27, 2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_GLOBALLOCATOR_H_
#define NAVIGATION_GLOBALLOCATOR_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/navigation/VLevelLocator.h"
#include "VecGeom/navigation/NavigationState.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class LogicalVolume;
class VPlacedVolume;

//! basic algorithms to locate a global point

//! basic generic algorithm that locates a global point in a geometry hierarchy
//! dispatches to specific LevelLocators
//! its a namespace rather than a class since it offers only a static function
namespace GlobalLocator {

// modified version using different LevelLocatorInterface
// this function is a generic variant which can pick from each volume
// the best (or default) LevelLocator
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
VPlacedVolume const *LocateGlobalPoint(VPlacedVolume const *vol, Vector3D<Precision> const &point,
                                       NavigationState &path, bool top)
{
  VPlacedVolume const *candvolume = vol;
  Vector3D<Precision> currentpoint(point);
  if (top) {
    assert(vol != nullptr);
    candvolume = (vol->UnplacedContains(point)) ? vol : nullptr;
  }
  if (candvolume) {
    path.Push(candvolume);
    LogicalVolume const *lvol         = candvolume->GetLogicalVolume();
    Vector<Daughter> const *daughters = lvol->GetDaughtersp();

    bool godeeper = true;
    while (daughters->size() > 0 && godeeper) {
      // returns nextvolume; and transformedpoint; modified path
      VLevelLocator const *locator = lvol->GetLevelLocator();
      if (locator != nullptr) { // if specialized/optimized technique attached to logical volume
        Vector3D<Precision> transformedpoint;
        godeeper = locator->LevelLocate(lvol, currentpoint, path, transformedpoint);
        if (godeeper) {
          candvolume   = path.Top();
          lvol         = candvolume->GetLogicalVolume();
          daughters    = lvol->GetDaughtersp();
          currentpoint = transformedpoint;
        }
      } else { // otherwise do a default implementation
        // GL: this code adapted from LocateGlobalPointExclVolume(...) function below
        godeeper = true;
        for (size_t i = 0; i < daughters->size() && godeeper; ++i) {
          VPlacedVolume const *nextvolume = (*daughters)[i];
          Vector3D<Precision> transformedpoint;
          if (nextvolume->Contains(currentpoint, transformedpoint)) {
            path.Push(nextvolume);
            currentpoint = transformedpoint;
            candvolume   = nextvolume;
            daughters    = candvolume->GetLogicalVolume()->GetDaughtersp();
            godeeper     = true;
            break;
          }
        }
        godeeper = false;
      }
    }
  }
  return candvolume ? path.Top() : nullptr;
}

// special version of locate point function that excludes searching a given volume
// (useful when we know that a particle must have traversed a boundary)
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
VPlacedVolume const *LocateGlobalPointExclVolume(VPlacedVolume const *vol, VPlacedVolume const *excludedvolume,
                                                 Vector3D<Precision> const &point, NavigationState &path, bool top)
{
  VPlacedVolume const *candvolume = vol;
  Vector3D<Precision> currentpoint(point);
  if (top) {
    assert(vol != nullptr);
    candvolume = (vol->UnplacedContains(point)) ? vol : nullptr;
  }
  if (candvolume) {
    path.Push(candvolume);
    LogicalVolume const *lvol         = candvolume->GetLogicalVolume();
    Vector<Daughter> const *daughters = lvol->GetDaughtersp();

    bool godeeper = true;
    while (daughters->size() > 0 && godeeper) {
      godeeper = false;
      // returns nextvolume; and transformedpoint; modified path
      VLevelLocator const *locator = lvol->GetLevelLocator();
      if (locator != nullptr) { // if specialized/optimized technique attached to logical volume
        Vector3D<Precision> transformedpoint;
        godeeper = locator->LevelLocateExclVol(lvol, excludedvolume, currentpoint, candvolume, transformedpoint);
        if (godeeper) {
          lvol         = candvolume->GetLogicalVolume();
          daughters    = lvol->GetDaughtersp();
          currentpoint = transformedpoint;
          path.Push(candvolume);
        }
      } else { // otherwise do a default implementation
        for (size_t i = 0; i < daughters->size(); ++i) {
          VPlacedVolume const *nextvolume = (*daughters)[i];
          if (nextvolume != excludedvolume) {
            Vector3D<Precision> transformedpoint;
            if (nextvolume->Contains(currentpoint, transformedpoint)) {
              path.Push(nextvolume);
              currentpoint = transformedpoint;
              candvolume   = nextvolume;
              daughters    = candvolume->GetLogicalVolume()->GetDaughtersp();
              godeeper     = true;
              break;
            }
          } // end if excludedvolume
        }
      }
    }
  }
  return candvolume;
}

VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
static VPlacedVolume const *RelocatePointFromPath(Vector3D<Precision> const &localpoint, NavigationState &path)
{
  // idea: do the following:
  // ----- is localpoint still in current mother ? : then go down
  // if not: have to go up until we reach a volume that contains the
  // localpoint and then go down again (neglecting the volumes currently stored in the path)
  VPlacedVolume const *currentmother = path.Top();
  if (currentmother != nullptr) {
    Vector3D<Precision> tmp = localpoint;
    while (currentmother) {
      if (currentmother->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly() ||
          !currentmother->UnplacedContains(tmp)) {
        path.Pop();
        Vector3D<Precision> pointhigherup = currentmother->GetTransformation()->InverseTransform(tmp);
        tmp                               = pointhigherup;
        currentmother                     = path.Top();
      } else {
        break;
      }
    }

    if (currentmother) {
      path.Pop();
      return LocateGlobalPoint(currentmother, tmp, path, false);
    }
  }
  return currentmother;
}

//  a version for relocation when we know that new state must be different
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
static VPlacedVolume const *RelocatePointFromPathForceDifferent(Vector3D<Precision> const &localpoint,
                                                                NavigationState &path)
{
  // idea: do the following:
  // ----- is localpoint still in current mother ? : then go down
  // if not: have to go up until we reach a volume that contains the
  // localpoint and then go down again (neglecting the volumes currently stored in the path)
  VPlacedVolume const *currentmother = path.Top();
  VPlacedVolume const *entryvol      = currentmother;
  if (currentmother != nullptr) {
    Vector3D<Precision> tmp = localpoint;
    while (currentmother) {
      if (currentmother == entryvol || currentmother->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly() ||
          !currentmother->UnplacedContains(tmp)) {
        path.Pop();
        Vector3D<Precision> pointhigherup = currentmother->GetTransformation()->InverseTransform(tmp);
        tmp                               = pointhigherup;
        currentmother                     = path.Top();
      } else {
        break;
      }
    }

    if (currentmother) {
      path.Pop();
      return LocateGlobalPointExclVolume(currentmother, entryvol, tmp, path, false);
    }
  }
  return currentmother;
}

VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
bool HasSamePath(Vector3D<Precision> const &globalpoint, Transformation3D const &globaltransf,
                 NavigationState const &currentstate, NavigationState &newstate)
{
  Vector3D<Precision> localpoint = globaltransf.Transform(globalpoint);
  currentstate.CopyTo(&newstate);
  RelocatePointFromPath(localpoint, newstate);
  return currentstate.HasSamePathAsOther(newstate);
}

/**
 * function to check whether global point has same path as given by currentstate
 * input:  A global point
 *         the path itself
 *         a new path object which is filled; Initially a copy of currentstate will be made
 * output: yes or no
 * side effects: modifies newstate to be path of globalpoint
 *
 */
VECCORE_ATT_HOST_DEVICE
inline bool HasSamePath(Vector3D<Precision> const &globalpoint, NavigationState const &currentstate,
                        NavigationState &newstate)
{
  Transformation3D m;
  currentstate.TopMatrix(m);
  return HasSamePath(globalpoint, m, currentstate, newstate);
}

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE inline bool HasSamePath(Vector3D<Real_t> const &globalpoint,
                                                NavigationState const &currentstate, NavigationState &newstate)
{
  Transformation3DMP<Real_t> m;
  currentstate.TopMatrix(m);
  return HasSamePath(globalpoint, m, currentstate, newstate);
}

} // namespace GlobalLocator
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVIGATION_GLOBALLOCATOR_H_ */
