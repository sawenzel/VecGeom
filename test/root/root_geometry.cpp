#include "management/GeoManager.h"
#include "management/RootGeoManager.h"
#include "navigation/NavigationState.h"
#include "navigation/SimpleNavigator.h"
#include "volumes/PlacedVolume.h"

#include "TGeoBBox.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoTube.h"
#include "TGeoVolume.h"

#include <cassert>

using namespace VECGEOM_NAMESPACE;

int main() {

  TGeoVolume *world_root = gGeoManager->MakeBox("world", NULL, 5., 5., 10.);
  TGeoVolume *tube_root  = gGeoManager->MakeTube("tube", NULL, 1., 5., 10.);

  world_root->AddNode(tube_root, 0, new TGeoTranslation(0, 0, 0));

  gGeoManager->SetTopVolume(world_root);
  gGeoManager->CloseGeometry();

  RootGeoManager::Instance().set_verbose(1);
  RootGeoManager::Instance().LoadRootGeometry();
  RootGeoManager::Instance().world()->PrintContent();

  VPlacedVolume const *const world = GeoManager::Instance().GetWorld();
  VPlacedVolume const *const tube = *world->daughters().begin();

  SimpleNavigator navigator;

  auto CheckPoint = [&] (const Precision x, const Precision y,
                         const Precision z, VPlacedVolume const *const volume) {
    Vector3D<Precision> const point = Vector3D<Precision>(x, y, z);
    NavigationState * path = NavigationState::MakeInstance(2);
    assert(navigator.LocatePoint(world, point, *path, true) == volume);
  };

  CheckPoint(0, 0, 0, world);
  CheckPoint(4, 4, -9, world);
  CheckPoint(4, 0, 3, tube);
  CheckPoint(0, 3, -5, tube);
  CheckPoint(0, 3, -11, NULL);

  printf("\nAll tests successfully passed.\n");

  return 0;
}
