/*
 * TestExportToROOT.cpp
 *
 *  Created on: 28.10.2014
 *      Author: swenzel
 */

#include "volumes/PlacedVolume.h"
#include "base/Global.h"
#include "base/Transformation3D.h"
#include "volumes/UnplacedBox.h"
#include "volumes/UnplacedTube.h"
#include "volumes/UnplacedCone.h"
#include "volumes/UnplacedTube.h"
#include "volumes/UnplacedTrd.h"
#include "volumes/UnplacedOrb.h"
#include "volumes/UnplacedParaboloid.h"
#include "volumes/UnplacedParallelepiped.h"
#include "volumes/UnplacedTorus.h"
#include "volumes/UnplacedTrapezoid.h"
#include "management/RootGeoManager.h"
#include "management/GeoManager.h"
#include "TGeoManager.h"
#include <iostream>

using namespace vecgeom;


// create a VecGeom geometry

VPlacedVolume* SetupGeometry() {
  UnplacedBox *worldUnplaced = new UnplacedBox(10, 10, 10);
  UnplacedBox *boxUnplaced = new UnplacedBox(2.5, 2.5, 2.5);

  UnplacedTube *tube1Unplaced = new UnplacedTube( 0.5, 1., 0.5, 0., kTwoPi);
  UnplacedTube *tube2Unplaced = new UnplacedTube( 0.5, 1., 0.5, 0., kPi);
  UnplacedCone *cone1Unplaced = new UnplacedCone( 0.5, 1., 0.6, 1.2, 0.5, 0., kTwoPi);
  UnplacedCone *cone2Unplaced = new UnplacedCone( 0.5, 1., 0.6, 1.2,0.5, kPi/4., kPi);

  UnplacedTrd *trdUnplaced = new UnplacedTrd( 0.1, 0.2, 0.15, 0.05 );
  UnplacedTrapezoid *trapUnplaced = new UnplacedTrapezoid(0.2,0.,0.,0.1,0.08,0.12,0.,0.15,0.12,0.18,0.);

  UnplacedOrb *orbUnplaced = new UnplacedOrb( 0.1 );
  UnplacedParaboloid *paraUnplaced = new UnplacedParaboloid( 0.1, 0.2, 0.1 );
  UnplacedParallelepiped *epipedUnplaced =  new UnplacedParallelepiped( 0.1, 0.05, 0.05, 0.2, 0.4, 0.1 );

  UnplacedTorus *torusUnplaced = new UnplacedTorus(0.1,2.1, 10,0,kTwoPi);

  Transformation3D *placement1 = new Transformation3D( 5,  5,  5,  0,  0,  0);
  Transformation3D *placement2 = new Transformation3D(-5,  5,  5, 45,  0,  0);
  Transformation3D *placement3 = new Transformation3D( 5, -5,  5,  0, 45,  0);
  Transformation3D *placement4 = new Transformation3D( 5,  5, -5,  0,  0, 45);
  Transformation3D *placement5 = new Transformation3D(-5, -5,  5, 45, 45,  0);
  Transformation3D *placement6 = new Transformation3D(-5,  5, -5, 45,  0, 45);
  Transformation3D *placement7 = new Transformation3D( 5, -5, -5,  0, 45, 45);
  Transformation3D *placement8 = new Transformation3D(-5, -5, -5, 45, 45, 45);

  Transformation3D *placement9 = new Transformation3D(-0.5,-0.5,-0.5,0,0,0);
  Transformation3D *placement10 = new Transformation3D(0.5,0.5,0.5,0,45,0);
  Transformation3D *idendity    = new Transformation3D();

  LogicalVolume *world = new LogicalVolume("world",worldUnplaced);
  LogicalVolume *box =   new LogicalVolume("lbox1",boxUnplaced);
  LogicalVolume *tube1 = new LogicalVolume("ltube1", tube1Unplaced);
  LogicalVolume *tube2 = new LogicalVolume("ltube2", tube2Unplaced);
  LogicalVolume *cone1 = new LogicalVolume("lcone1", cone1Unplaced);
  LogicalVolume *cone2 = new LogicalVolume("lcone2", cone2Unplaced);

  LogicalVolume *trd1  = new LogicalVolume("ltrd", trdUnplaced);
  LogicalVolume *trap1 = new LogicalVolume("ltrap", trapUnplaced);

  LogicalVolume *orb1 = new LogicalVolume("lorb1", orbUnplaced);
  LogicalVolume *parab1 = new LogicalVolume("lparab1", paraUnplaced);
  LogicalVolume *epip1 = new LogicalVolume("lepip1", epipedUnplaced);
  LogicalVolume *tor1 = new LogicalVolume("torus1", torusUnplaced);


  world->PlaceDaughter(orb1, idendity);
  trd1->PlaceDaughter(parab1, idendity);
  world->PlaceDaughter(epip1, idendity);

  tube1->PlaceDaughter( trd1, idendity );
  tube2->PlaceDaughter( trap1, idendity );
  box->PlaceDaughter( tube1, placement9 );
  box->PlaceDaughter( tube2, placement10 );

  world->PlaceDaughter(box, placement1);
  world->PlaceDaughter(box, placement2);
  world->PlaceDaughter(box, placement3);
  world->PlaceDaughter(box, placement4);
  world->PlaceDaughter(box, placement5);
  world->PlaceDaughter(box, placement6);
  world->PlaceDaughter(box, placement7);
  world->PlaceDaughter(box, placement8);

  cone1->PlaceDaughter(trap1, idendity);
  world->PlaceDaughter(cone1, new Transformation3D(8,0,0,0,0,0));
  world->PlaceDaughter(cone2, new Transformation3D(-8,0,0,0,0,0));
  // might not be fully in world ( need to rotate ... )
  world->PlaceDaughter(tor1, new Transformation3D(-9,0,0,0,0,0));


  return world->Place();
}


int main()
{
    VPlacedVolume const * world = SetupGeometry();
    GeoManager::Instance().SetWorld(world);
    GeoManager::Instance().CloseGeometry();
    int md1 = GeoManager::Instance().getMaxDepth();
    int mpv1 = GeoManager::Instance().GetPlacedVolumesCount();
    int mlv1 = GeoManager::Instance().GetLogicalVolumesCount();
    int ntotalnodes1 = GeoManager::Instance().GetTotalNodeCount();

    // exporting to ROOT file
    RootGeoManager::Instance().ExportToROOTGeometry( world, "geom1.root" );

    assert( ::gGeoManager->GetNNodes() == ntotalnodes1 );
    assert( ::gGeoManager->GetListOfVolumes()->GetEntries() == mlv1 );

    //
    RootGeoManager::Instance().Clear();

    // now try to read back in
    RootGeoManager::Instance().set_verbose(1);
    RootGeoManager::Instance().LoadRootGeometry("geom1.root");

    //// see if everything was restored
    // RootGeoManager::Instance().world()->logical_volume()->PrintContent(0);

    int md2 = GeoManager::Instance().getMaxDepth();
    int mpv2 = GeoManager::Instance().GetPlacedVolumesCount();
    int mlv2 = GeoManager::Instance().GetLogicalVolumesCount();
    int ntotalnodes2 = GeoManager::Instance().GetTotalNodeCount();

    assert( md2 == md1 );
    assert( mpv2 == mpv1 );
    assert( mlv2 == mlv1 );
    assert( mpv2 > 0);
    assert( mlv2 > 0);
    assert( ntotalnodes1 == ntotalnodes2 );

    return 0;
}
