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
  UnplacedBox *boxUnplaced = new UnplacedBox(2.5, 2.5, 2.5);
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
  return world->Place();
}

// function
void testVectorSafety( VPlacedVolume* world ){
   SOA3D<Precision> points(1024);
   SOA3D<Precision> workspace(1024);
   Precision * safeties = (Precision *) _mm_malloc(sizeof(Precision)*1024,32);
   vecgeom::volumeUtilities::FillUncontainedPoints( *world, points );

   // now setup all the navigation states
   NavigationState ** states = new NavigationState*[1024];
   vecgeom::SimpleNavigator nav;
   for (int i=0;i<1024;++i){
       states[i]=new NavigationState( GeoManager::Instance().getMaxDepth() );
       nav.LocatePoint( world, points[i], *states[i], true);
   }

    // calculate safeties with vector interface
    nav.GetSafeties(points, states, workspace, safeties );

    // verify against serial interface
    for (int i=0;i<1024;++i){
        vecgeom::Assert( safeties[i] == nav.GetSafety( points[i], *states[i] ), ""
                " Problem in VectorSafety (in SimpleNavigator)" );
    }
   _mm_free(safeties);
}


int main()
{
    testVectorSafety(SetupBoxGeometry());
}
