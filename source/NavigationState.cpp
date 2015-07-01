/// \file NavigationState.cpp
/// \author Sandro Wenzel (sandro.wenzel@cern.ch)
/// \date 17.04.2014

#include "navigation/NavigationState.h"
 
#include <cassert>
#include <iostream>
#include <list>

#ifdef VECGEOM_ROOT
#include "management/RootGeoManager.h"
#include "TGeoBranchArray.h"
#include "TGeoNode.h"
#include "TGeoManager.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision>
NavigationState::GlobalToLocal(Vector3D<Precision> const & globalpoint, int tolevel) const{
    Vector3D<Precision> tmp=globalpoint;
    Vector3D<Precision> current;
    for(int level=0;level<tolevel;++level)
    {
      Transformation3D const *m = fPath[level]->GetTransformation();
      current = m->Transform( tmp );
      tmp = current;
    }
    return tmp;

}

VECGEOM_CUDA_HEADER_BOTH
void
NavigationState::TopMatrix( int tolevel, Transformation3D & global_matrix ) const {
    for(int i=1;i<tolevel;++i)
    {
       global_matrix.MultiplyFromRight( *(fPath[i]->GetTransformation()) );
    }
}


/**
 * function that transforms a global point to local point in reference frame of deepest volume in current navigation state
 * ( equivalent to using a global matrix )
 */
VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision>
NavigationState::GlobalToLocal(Vector3D<Precision> const & globalpoint) const
{
   Vector3D<Precision> tmp=globalpoint;
   Vector3D<Precision> current;
   for(int level=0;level<fCurrentLevel;++level)
   {
      Transformation3D const *m = fPath[level]->GetTransformation();
      current = m->Transform( tmp );
      tmp = current;
   }
   return tmp;
}

  uint FindIndexWithinMother( VPlacedVolume const * mother, VPlacedVolume const * daughter )
  {
    for( auto d = 0; d<mother->GetDaughters().size(); ++d)
    {
        if ( mother->GetDaughters()[d] == daughter) return d;
    }
    return -1;
  }

  VPlacedVolume const * GetDaughterWithinMother( VPlacedVolume const * mother, uint index )
    {
      if( index < (uint) mother->GetDaughters().size() )
          return mother->GetDaughters()[index];

      return NULL;
    }

  void NavigationState::GetPathAsListOfIndices(  std::list<uint> & indices ) const {
    indices.clear();
    if( IsOutside() ) return;
    for( uint level = fCurrentLevel-1; level>0; --level){
        indices.push_front( FindIndexWithinMother( At(level-1), At(level) ) );
    }
    indices.push_front(0);
  }

  void NavigationState::ResetPathFromListOfIndices( VPlacedVolume const * world, std::list<uint> const & indices ){
    // clear current nav state
    fCurrentLevel =  indices.size();
    if( indices.size() > 0 ) {
        fPath[0] = world;
        // have to disregard first one;
        // then iterate through list
        int counter=0;
        for( auto x : indices ){
            if(counter>0)
                fPath[counter] = GetDaughterWithinMother( At(counter-1), x );
            counter++;
        }
    }
  }

#ifdef VECGEOM_ROOT
  TGeoBranchArray * NavigationState::ToTGeoBranchArray() const
  {
    // attention: the counting of levels is different: fLevel=0 means that
    // this is a branch which is filled at level zero

    // my counting is a bit different: it tells the NUMBER of levels which are filled
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,34,23)
    TGeoBranchArray * tmp = TGeoBranchArray::MakeInstance( GetMaxLevel() );
#else
    TGeoBranchArray * tmp = new TGeoBranchArray( GetMaxLevel() );
#endif
    // gain access to array
    TGeoNode ** array = tmp->GetArray();
    RootGeoManager & mg=RootGeoManager::Instance();
    TGeoNavigator * nav = gGeoManager->GetCurrentNavigator();
    tmp->InitFromNavigator( nav );

    //tmp->
    for(int i=0;i<fCurrentLevel;++i)
          array[i]=const_cast<TGeoNode *>(mg.tgeonode( fPath[i] ));
    // assert( tmp->GetCurrentNode() == mg.tgeonode( Top() ));

    /*
    std::list<uint> ilist;
    GetPathAsListOfIndices( ilist );
    int counter=0;
    for( auto x : ilist ) {
        if(counter>0)
            tmp->AddLevel(x);
        counter++;
    }
    */
    
    return tmp;
  }

  NavigationState & NavigationState::operator=( TGeoBranchArray const & other )
   {
     // attention: the counting of levels is different: fLevel=0 means that
     // this is a branch which is filled at level zero
	 this->fCurrentLevel=other.GetLevel()+1;
	 assert(fCurrentLevel <= GetMaxLevel());

     RootGeoManager & mg=RootGeoManager::Instance();

     for(int i=0;i<fCurrentLevel;++i)
       fPath[i]=mg.GetPlacedVolume( other.GetNode(i) );

     //other things like onboundary I don't care
     fOnBoundary=false;

     return *this;
   }

#endif

} } // End global namespace

