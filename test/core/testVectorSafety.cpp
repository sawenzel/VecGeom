/*
 * testVectorSafety.cpp
 *
 *  Created on: Jun 25, 2014
 *      Author: swenzel
 */

#include "volumes/utilities/VolumeUtilities.h"
#include "volumes/Box.h"
#include "base/Transformation3D.h"
#include "base/SOA3D.h"
#include "navigation/NavigationState.h"
#include "navigation/SimpleNavigator.h"
#include "management/GeoManager.h"
#include "base/Global.h"

using namespace vecgeom;


VPlacedVolume* SetupBoxGeometry() {
  UnplacedBox *worldUnplaced = new UnplacedBox(10, 10, 10);
  UnplacedBox *boxUnplaced = new UnplacedBox(0.5, 0.5, 0.5);
  Transformation3D *placement1 = new Transformation3D( 2,  2,  2,  0,  0,  0);
  Transformation3D *placement2 = new Transformation3D(-2,  2,  2, 45,  0,  0);
  Transformation3D *placement3 = new Transformation3D( 2, -2,  2,  0, 45,  0);
  Transformation3D *placement4 = new Transformation3D( 2,  2, -2,  0,  0, 45);
  Transformation3D *placement5 = new Transformation3D(-2, -2,  2, 45, 45,  0);
  Transformation3D *placement6 = new Transformation3D(-2,  2, -2, 45,  0, 45);
  Transformation3D *placement7 = new Transformation3D( 2, -2, -2,  0, 45, 45);
  Transformation3D *placement8 = new Transformation3D(-2, -2, -2, 45, 45, 45);
  LogicalVolume *world = new LogicalVolume(worldUnplaced);
  LogicalVolume *box = new LogicalVolume(boxUnplaced);
  world->PlaceDaughter(box, placement1);
  world->PlaceDaughter(box, placement2);
  world->PlaceDaughter(box, placement3);
  world->PlaceDaughter(box, placement4);
  world->PlaceDaughter(box, placement5);
  world->PlaceDaughter(box, placement6);
  world->PlaceDaughter(box, placement7);
  world->PlaceDaughter(box, placement8);
  VPlacedVolume  * w = world->Place();
  GeoManager::Instance().SetWorld(w);
  GeoManager::Instance().CloseGeometry();
  return w;
}

// function to test safety
void testVectorSafety( VPlacedVolume* world ){
   SOA3D<Precision> points(1024);
   SOA3D<Precision> workspace(1024);
   Precision * safeties = (Precision *) _mm_malloc(sizeof(Precision)*1024,32);
   vecgeom::volumeUtilities::FillUncontainedPoints( *world, points );

   // now setup all the navigation states
   NavigationState ** states = new NavigationState*[1024];
   vecgeom::SimpleNavigator nav;
   for (int i=0;i<1024;++i){
       states[i]=NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );
       nav.LocatePoint( world, points[i], *states[i], true);
   }

    // calculate safeties with vector interface
    nav.GetSafeties(points, states, workspace, safeties );

    // verify against serial interface
    for (int i=0;i<1024;++i){
        vecgeom::Assert( safeties[i] == nav.GetSafety( points[i], *states[i] ), ""
                " Problem in VectorSafety (in SimpleNavigator)" );
    }
    std::cout << "Safety test passed\n";
   _mm_free(safeties);
}

// function to test vector navigator
void testVectorNavigator( VPlacedVolume* world ){
   int np=100000;
   SOA3D<Precision> points(np);
   SOA3D<Precision> dirs(np);
   SOA3D<Precision> workspace1(np);
   SOA3D<Precision> workspace2(np);

   Precision * steps = (Precision *) _mm_malloc(sizeof(Precision)*np,32);
   Precision * pSteps = (Precision *) _mm_malloc(sizeof(Precision)*np,32);
   Precision * safeties = (Precision *) _mm_malloc(sizeof(Precision)*np,32);

   int * intworkspace = (int *) _mm_malloc(sizeof(int)*np,32);

   vecgeom::volumeUtilities::FillUncontainedPoints( *world, points );
   vecgeom::volumeUtilities::FillRandomDirections( dirs );

   // now setup all the navigation states
   NavigationState ** states = new NavigationState*[np];
   NavigationState ** newstates = new NavigationState*[np];

   vecgeom::SimpleNavigator nav;
   for (int i=0;i<np;++i){
      // pSteps[i] = kInfinity;
     pSteps[i] = (i%2)? 1 : VECGEOM_NAMESPACE::kInfinity;
       states[i] =  NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );
       newstates[i] = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );
       nav.LocatePoint( world, points[i], *states[i], true);
   }

   // calculate steps with vector interface
   nav.FindNextBoundaryAndStep( points, dirs, workspace1, workspace2,
           states, newstates, pSteps, safeties, steps, intworkspace );

   // verify against serial interface
   for (int i=0;i<np;++i) {
       Precision s=0;
       NavigationState * cmp = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );
       cmp->Clear();
       nav.FindNextBoundaryAndStep( points[i], dirs[i], *states[i],
               *cmp, pSteps[i], s );
       vecgeom::Assert( steps[i] == s ,
               " Problem in VectorNavigation (steps) (in SimpleNavigator)" );
       vecgeom::Assert( cmp->Top() == newstates[i]->Top() ,
                      " Problem in VectorNavigation (states) (in SimpleNavigator)" );
       vecgeom::Assert( cmp->IsOnBoundary() == newstates[i]->IsOnBoundary(),
                      " Problem in VectorNavigation (boundary) (in SimpleNavigator)" );

       vecgeom::Assert( safeties[i] == nav.GetSafety( points[i], *states[i] ),
               " Problem with safety " );
   }

    std::cout << "Navigation test passed\n";
   _mm_free(steps);
   _mm_free(intworkspace);
   _mm_free(pSteps);
}


int main()
{
    VPlacedVolume *w;
    testVectorSafety(w=SetupBoxGeometry());
    testVectorNavigator(w);
}
