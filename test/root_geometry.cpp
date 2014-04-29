#include <iostream>

#include "management/rootgeo_manager.h"
#include "volumes/placed_volume.h"

#include "TGeoBBox.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoTube.h"
#include "TGeoVolume.h"

using namespace VECGEOM_NAMESPACE;

int main() {

  const double l = 10.;
  const double lz = 10.;
  const double sqrt2 = sqrt(2.);

  TGeoVolume *world = ::gGeoManager->MakeBox("world",0, l, l, lz );
  TGeoVolume *boxLevel1 = ::gGeoManager->MakeBox("b1", 0, l/2., l/2., lz);
  TGeoVolume *boxLevel2 = ::gGeoManager->MakeBox("b2", 0, sqrt2*l/2./2.,
                                                 sqrt2*l/2./2., lz);
  TGeoVolume *boxLevel3 = ::gGeoManager->MakeBox("b3", 0, l/2./2., l/2./2.,
                                                 lz);

  boxLevel2->AddNode(boxLevel3, 0, new TGeoRotation("rot1",0,0,45));
  boxLevel1->AddNode(boxLevel2, 0, new TGeoRotation("rot2",0,0,-45));
  world->AddNode(boxLevel1, 0, new TGeoTranslation(-l/2.,0,0));
  world->AddNode(boxLevel1, 1, new TGeoTranslation(+l/2.,0,0));

  ::gGeoManager->SetTopVolume(world);
  ::gGeoManager->CloseGeometry();

  RootGeoManager::Instance().set_verbose(1);
  RootGeoManager::Instance().LoadRootGeometry();
  RootGeoManager::Instance().world()->PrintContent();

  return 0;
}
