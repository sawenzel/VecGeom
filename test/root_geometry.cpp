#include <iostream>

#include "management/root_manager.h"
#include "volumes/placed_volume.h"

#include "TGeoManager.h"
#include "TGeoBBox.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"

using namespace vecgeom;

int main() {

  TGeoShape *box1 = new TGeoBBox(20., 20., 20.);
  TGeoShape *box2 = new TGeoBBox(2., 7., 3.);
  TGeoVolume *vol1 = new TGeoVolume("", box1, NULL);
  TGeoVolume *vol2 = new TGeoVolume("", box2, NULL); 
  TGeoMatrix *matrix1 =
      new TGeoCombiTrans(0., -5., 6., new TGeoRotation("", 1., 2., 3.));
  TGeoMatrix *matrix2 =
      new TGeoCombiTrans(3., 3., 3., new TGeoRotation("", 3., 2., 1.));
  vol1->AddNode(vol2, 0, matrix1);
  vol1->AddNode(vol2, 1, matrix2);
  ::gGeoManager->SetTopVolume(vol1);

  RootManager::Instance().LoadRootGeometry();
  RootManager::Instance().world()->logical_volume()->PrintContent();

  return 0;
}
