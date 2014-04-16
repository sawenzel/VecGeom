/*
 * simple_navigator.cpp
 *
 *  Created on: Apr 16, 2014
 *      Author: swenzel
 */


#include "base/global.h"
#include "volumes/placed_volume.h"
#include "base/vector3d.h"
#include "navigation/navigationstate.h"
#include "navigation/simple_navigator.h"
#include "management/geo_manager.h"

namespace VECGEOM_NAMESPACE
{

#ifdef VECGEOM_ROOT
void SimpleNavigator::InspectEnvironmentForPointAndDirection
   (   Vector3D<Precision> const & globalpoint,
      Vector3D<Precision> const & globaldir,
      NavigationState const & state ) const
{
   TransformationMatrix m = const_cast<NavigationState &>(state).TopMatrix();
   Vector3D<Precision> localpoint = m.Transform( globalpoint );
   Vector3D<Precision> localdir = m.TransformDirection( globaldir );

   // check that everything is consistent
   {
      NavigationState tmpstate( state );
      tmpstate.Clear();
      assert( LocatePoint( GeoManager::Instance().world(),
              globalpoint, tmpstate, true ) == state.Top() );
   }

   // now check mother and daughters
   VPlacedVolume const * currentvolume = state.Top();
   std::cout << "############################################ " << "\n";
   std::cout << "Navigating in placed volume : " << RootGeoManager::Instance().GetName( currentvolume ) << "\n";

   int nexthitvolume = -1; // means mother
   double step = currentvolume->DistanceToOut( localpoint, localdir );

   std::cout << "DistanceToOutMother : " << step << "\n";

   // iterate over all the daughters
   Vector<Daughter> const * daughters = currentvolume->logical_volume()->daughtersp();

   std::cout << "ITERATING OVER " << daughters->size() << " DAUGHTER VOLUMES " << "\n";
   for(int d = 0; d<daughters->size(); ++d)
   {
      VPlacedVolume const * daughter = daughters->operator [](d);
      //    previous distance becomes step estimate, distance to daughter returned in workspace
      Precision ddistance = daughter->DistanceToIn( localpoint, localdir, step );

      std::cout << "DistanceToDaughter : " << RootGeoManager::Instance().GetName( daughter ) << " " << ddistance << "\n";

      nexthitvolume = (ddistance < step) ? d : nexthitvolume;
      step      = (ddistance < step) ? ddistance  : step;
   }
   std::cout << "DECIDED FOR NEXTVOLUME " << nexthitvolume << "\n";

   // same information from ROOT
   TGeoNode const * currentRootNode = RootGeoManager::Instance().tgeonode( currentvolume );
   double lp[3]={localpoint[0],localpoint[1],localpoint[2]};
   double ld[3]={localdir[0],localdir[1],localdir[2]};
   double rootstep =  currentRootNode->GetVolume()->GetShape()->DistFromInside( lp, ld, 3, 1E30, 0 );
   std::cout << "---------------- CMP WITH ROOT ---------------------" << "\n";
   std::cout << "DistanceToOutMother ROOT : " << rootstep << "\n";
   std::cout << "ITERATING OVER " << currentRootNode->GetNdaughters() << " DAUGHTER VOLUMES " << "\n";
   for( int d=0; d<currentRootNode->GetNdaughters();++d )
   {
      TGeoMatrix const * m = currentRootNode->GetMatrix();
      double llp[3], lld[3];
      m->MasterToLocal(lp, llp);
      m->MasterToLocalVect(ld, lld);
      TGeoNode const * daughter=currentRootNode->GetDaughter(d);
      Precision ddistance = daughter->GetVolume()->GetShape()->DistFromOutside(llp,llp,3,1E30,0);

      std::cout << "DistanceToDaughter ROOT : " << daughter->GetName() << " " << ddistance << "\n";
   }
}
#endif


}
