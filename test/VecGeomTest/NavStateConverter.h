/// \file NavStatePathConverter.h

#pragma once

#include "VecGeom/navigation/NavStatePath.h"
#include "VecGeom/navigation/NavStateIndex.h"
#include "VecGeomTest/RootGeoManager.h"
#include "TGeoBranchArray.h"
#include "TGeoNode.h"
#include "TGeoManager.h"

namespace NavStateConverter {
/// Convert input NavStatePath to TGeoBranchArray
///
/// Caller is takes ownership of returned pointer
template <typename NavState>
TGeoBranchArray *ToTGeoBranchArray(NavState const &nsp)
{
  // attention: the counting of levels is different: fLevel=0 means that
  // this is a branch which is filled at level zero

  // my counting is a bit different: it tells the NUMBER of levels which are filled
  TGeoBranchArray *tmp = TGeoBranchArray::MakeInstance(nsp.GetMaxLevel());

  // gain access to array
  TGeoNode **array            = tmp->GetArray();
  vecgeom::RootGeoManager &mg = vecgeom::RootGeoManager::Instance();
  TGeoNavigator *nav          = gGeoManager->GetCurrentNavigator();
  tmp->InitFromNavigator(nav);

  // tmp->
  for (int i = 0; i < nsp.GetCurrentLevel(); ++i)
    array[i] = const_cast<TGeoNode *>(mg.tgeonode(nsp.At(i)));

  return tmp;
}
} // namespace NavStateConverter
