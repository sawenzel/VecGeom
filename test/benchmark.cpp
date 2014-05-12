#include "volumes/logical_volume.h"
#include "volumes/box.h"
#include "volumes/Parallelepiped.h"
#include "benchmarking/ToInBenchmarker.h"
#include "management/geo_manager.h"

using namespace vecgeom;

int main() {

  // const double l = 10;
  // const double sqrt2 = sqrt(2.);

  // UnplacedBox worldUnplaced = UnplacedBox(l, l, l);
  // UnplacedBox boxLevel1Unplaced = UnplacedBox(0.5*l, 0.5*l, l);
  // UnplacedBox boxLevel2Unplaced = UnplacedBox(sqrt2*l/2./2., sqrt2*l/2./2., l);
  // UnplacedBox boxLevel3Unplaced = UnplacedBox(l/2/2, l/2/2, l);

  // Transformation3D placement1 = Transformation3D(-l/2, 0, 0);
  // Transformation3D placement2 = Transformation3D( l/2, 0, 0);
  // Transformation3D placement3 = Transformation3D(0, 0, 0, 0, 0, 45);
  // Transformation3D placement4 = Transformation3D(0, 0, 0, 0, 0, -45);

  // LogicalVolume world = LogicalVolume(&worldUnplaced);
  // LogicalVolume boxLevel1 = LogicalVolume(&boxLevel1Unplaced);
  // LogicalVolume boxLevel2 = LogicalVolume(&boxLevel2Unplaced);
  // LogicalVolume boxLevel3 = LogicalVolume(&boxLevel3Unplaced);
  // world.PlaceDaughter(&boxLevel1, &placement1);
  // world.PlaceDaughter(&boxLevel1, &placement2);
  // boxLevel1.PlaceDaughter(&boxLevel2, &placement4);
  // boxLevel2.PlaceDaughter(&boxLevel3, &placement3);

  UnplacedBox worldUnplaced = UnplacedBox(10., 10., 10.);
  UnplacedParallelepiped paraUnplaced =
      UnplacedParallelepiped(3., 3., 3., 45, 10, 15);
  LogicalVolume world = LogicalVolume(&worldUnplaced);
  LogicalVolume para = LogicalVolume(&paraUnplaced);
  world.PlaceDaughter(&para, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  ToInBenchmarker tester(GeoManager::Instance().world());
  tester.SetVerbose(2);
  tester.SetPointCount(1<<13);
  tester.BenchmarkAll();

  return 0;
}