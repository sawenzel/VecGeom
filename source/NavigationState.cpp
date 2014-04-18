/*
 * NavigationState.cpp
 *
 *  Created on: Apr 17, 2014
 *      Author: swenzel
 */

#include "navigation/navigationstate.h"
#include "management/rootgeo_manager.h"

#include "TGeoBranchArray.h"
#include "TGeoNode.h"
#include <cassert>
#include <iostream>

namespace VECGEOM_NAMESPACE
{
  TGeoBranchArray * NavigationState::ToTGeoBranchArray() const
  {
    // attention: the counting of levels is different: fLevel=0 means that
    // this is a branch which is filled at level zero

    // my counting is a bit different: it tells the NUMBER of levels which are filled
    TGeoBranchArray * tmp = new TGeoBranchArray( currentlevel_-1 );
    // gain access to array
    TGeoNode ** array = tmp->GetArray();
    RootGeoManager & mg=RootGeoManager::Instance();
    
    for(int i=0;i<currentlevel_;++i)
      array[i]=const_cast<TGeoNode *>(mg.tgeonode( path_[i] ));
    std::cout << "currentlevel_ " << currentlevel_ << std::endl;
    // tmp->Print();
    assert(tmp->GetCurrentNode() == mg.tgeonode( Top() ));
    
    return tmp;
  }
};

