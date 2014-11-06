/// \file SimpleNavigator.cpp
/// \author Sandro Wenzel (sandro.wenzel@cern.ch)
/// \date 16.04.2014

#include "navigation/SimpleNavigator.h"

#include "base/Vector3D.h"
#include "management/GeoManager.h"
#include "navigation/NavigationState.h"
#include "volumes/PlacedVolume.h"

#ifdef VECGEOM_ROOT
#include "TGeoManager.h"
#endif

namespace VECGEOM_NAMESPACE
{

#ifdef VECGEOM_ROOT

void SimpleNavigator::InspectEnvironmentForPointAndDirection
   (   Vector3D<Precision> const & globalpoint,
      Vector3D<Precision> const & globaldir,
      NavigationState const & state ) const
{
   Transformation3D m = const_cast<NavigationState &>(state).TopMatrix();
   Vector3D<Precision> localpoint = m.Transform( globalpoint );
   Vector3D<Precision> localdir = m.TransformDirection( globaldir );

   // check that everything is consistent
   {
      NavigationState tmpstate( state );
      tmpstate.Clear();
      assert( LocatePoint( GeoManager::Instance().GetWorld(),
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

void SimpleNavigator::InspectSafetyForPoint
   (   Vector3D<Precision> const & globalpoint,
      NavigationState const & state ) const
{
   Transformation3D m = const_cast<NavigationState &>(state).TopMatrix();
   Vector3D<Precision> localpoint = m.Transform( globalpoint );

   // check that everything is consistent
   {
      NavigationState tmpstate( state );
      tmpstate.Clear();
      assert( LocatePoint( GeoManager::Instance().GetWorld(),
              globalpoint, tmpstate, true ) == state.Top() );
   }

   std::cout << "############################################ " << "\n";
   // safety to mother
   VPlacedVolume const * currentvol = state.Top();
   double safety = currentvol->SafetyToOut( localpoint );
   std::cout << "Safety in placed volume : " << RootGeoManager::Instance().GetName( currentvol ) << "\n";
   std::cout << "Safety to Mother : " << safety << "\n";

   //assert( safety > 0 );

   // safety to daughters
   Vector<Daughter> const * daughters = currentvol->logical_volume()->daughtersp();
   int numberdaughters = daughters->size();
   for(int d = 0; d<numberdaughters; ++d)
   {
	   VPlacedVolume const * daughter = daughters->operator [](d);
	   double tmp = daughter->SafetyToIn( localpoint );
       std::cout << "Safety to Daughter " << tmp << "\n";
	   safety = Min(safety, tmp);
   }
   std::cout << "Would return" << safety << "\n";

   // same information from ROOT
   TGeoNode const * currentRootNode = RootGeoManager::Instance().tgeonode( currentvol );
   double lp[3]={localpoint[0],localpoint[1],localpoint[2]};
   double rootsafe =  currentRootNode->GetVolume()->GetShape()->Safety( lp, kTRUE );
   std::cout << "---------------- CMP WITH ROOT ---------------------" << "\n";
   std::cout << "SafetyToOutMother ROOT : " << rootsafe << "\n";
   std::cout << "ITERATING OVER " << currentRootNode->GetNdaughters() << " DAUGHTER VOLUMES " << "\n";
   for( int d=0; d<currentRootNode->GetNdaughters();++d )
   {
      TGeoMatrix const * m = currentRootNode->GetMatrix();
      double llp[3];
      m->MasterToLocal(lp, llp);
      TGeoNode const * daughter=currentRootNode->GetDaughter(d);
      Precision ddistance = daughter->GetVolume()->GetShape()->Safety(llp, kFALSE);
      std::cout << "Safety ToDaughter ROOT : " << daughter->GetName() << " " << ddistance << "\n";
   }
}

#endif // VECGEOM_ROOT

}
