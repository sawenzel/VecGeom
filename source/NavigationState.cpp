/// \file NavigationState.cpp
/// \author Sandro Wenzel (sandro.wenzel@cern.ch)
/// \date 17.04.2014

#include "navigation/NavigationState.h"
 
#include <cassert>
#include <iostream>

#ifdef VECGEOM_ROOT
#include "management/RootGeoManager.h"
#include "TGeoBranchArray.h"
#include "TGeoNode.h"
#endif

namespace VECGEOM_NAMESPACE
{

#ifdef VECGEOM_ROOT

  TGeoBranchArray * NavigationState::ToTGeoBranchArray() const
  {
    // attention: the counting of levels is different: fLevel=0 means that
    // this is a branch which is filled at level zero

    // my counting is a bit different: it tells the NUMBER of levels which are filled
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,34,23)
    TGeoBranchArray * tmp = TGeoBranchArray::MakeInstance( currentlevel_-1, 0 );
#else
    TGeoBranchArray * tmp = new TGeoBranchArray( currentlevel_-1 );
#endif

    // gain access to array
    TGeoNode ** array = tmp->GetArray();
    RootGeoManager & mg=RootGeoManager::Instance();
    
    for(int i=0;i<currentlevel_;++i)
      array[i]=const_cast<TGeoNode *>(mg.tgeonode( path_[i] ));
    // assert( tmp->GetCurrentNode() == mg.tgeonode( Top() ));
    
    return tmp;
  }

  NavigationState & NavigationState::operator=( TGeoBranchArray const & other )
   {
     // attention: the counting of levels is different: fLevel=0 means that
     // this is a branch which is filled at level zero
	 this->currentlevel_=other.GetLevel()+1;
	 assert(currentlevel_ <= maxlevel_);

     RootGeoManager & mg=RootGeoManager::Instance();

     for(int i=0;i<currentlevel_;++i)
       path_[i]=mg.GetPlacedVolume( other.GetNode(i) );

     //other things like onboundary I don't care
     onboundary_=false;

     return *this;
   }

#endif

};

