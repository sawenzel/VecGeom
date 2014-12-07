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

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifdef VECGEOM_ROOT

  TGeoBranchArray * NavigationState::ToTGeoBranchArray() const
  {
    // attention: the counting of levels is different: fLevel=0 means that
    // this is a branch which is filled at level zero

    // my counting is a bit different: it tells the NUMBER of levels which are filled
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,34,23)
    TGeoBranchArray * tmp = TGeoBranchArray::MakeInstance( fMaxlevel );
#else
    TGeoBranchArray * tmp = new TGeoBranchArray( fMaxlevel );
#endif
    // gain access to array
    TGeoNode ** array = tmp->GetArray();
    RootGeoManager & mg=RootGeoManager::Instance();
    
    for(int i=0;i<fCurrentLevel;++i)
      array[i]=const_cast<TGeoNode *>(mg.tgeonode( fPath[i] ));
    // assert( tmp->GetCurrentNode() == mg.tgeonode( Top() ));
    
    return tmp;
  }

  NavigationState & NavigationState::operator=( TGeoBranchArray const & other )
   {
     // attention: the counting of levels is different: fLevel=0 means that
     // this is a branch which is filled at level zero
	 this->fCurrentLevel=other.GetLevel()+1;
	 assert(fCurrentLevel <= fMaxlevel);

     RootGeoManager & mg=RootGeoManager::Instance();

     for(int i=0;i<fCurrentLevel;++i)
       fPath[i]=mg.GetPlacedVolume( other.GetNode(i) );

     //other things like onboundary I don't care
     fOnBoundary=false;

     return *this;
   }

#endif

} } // End global namespace

