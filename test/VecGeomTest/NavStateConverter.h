/// \file NavStateConverter.h

#pragma once

#include "VecGeom/navigation/NavigationState.h"
#include "VecGeomTest/RootGeoManager.h"
#include "TGeoBranchArray.h"
#include "TGeoNode.h"
#include "TGeoManager.h"

namespace NavStateConverter {
/**
 * Convert an input navigation state to `TGeoBranchArray`.
 *
 * The helper is representation-agnostic. It only requires the common
 * navigation-state path interface:
 * - `GetMaxLevel()`
 * - `GetCurrentLevel()`
 * - `At(level)`
 *
 * Consequently it works for the current VecGeom navigation-state
 * implementations, including `NavStateIndex` and `NavStateTuple`.
 *
 * Caller takes ownership of the returned pointer.
 */
template <typename NavState>
TGeoBranchArray *ToTGeoBranchArray(NavState const &nsp)
{
  // TGeo stores level 0 in the first slot, while VecGeom exposes the number of
  // filled levels via GetCurrentLevel().
  TGeoBranchArray *tmp = TGeoBranchArray::MakeInstance(nsp.GetMaxLevel());

  TGeoNode **array            = tmp->GetArray();
  vecgeom::RootGeoManager &mg = vecgeom::RootGeoManager::Instance();
  TGeoNavigator *nav          = gGeoManager->GetCurrentNavigator();
  tmp->InitFromNavigator(nav);

  for (int i = 0; i < nsp.GetCurrentLevel(); ++i)
    array[i] = const_cast<TGeoNode *>(mg.tgeonode(nsp.At(i)));

  return tmp;
}
} // namespace NavStateConverter
