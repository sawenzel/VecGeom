#include "volumes/logical_volume.h"
#include "volumes/box.h"
#include "benchmarking/Benchmarker.h"
#include "management/geo_manager.h"

using namespace vecgeom;

int main() {

  UnplacedBox worldUnplaced = UnplacedBox(10., 10., 10.);
  UnplacedBox boxUnplaced =
      UnplacedBox(2.5, 2.5, 2.5);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume box = LogicalVolume("box", &boxUnplaced);

  Transformation3D placement1( 5,  5,  5,  0,  0,  0);
  Transformation3D placement2(-5,  5,  5, 45,  0,  0);
  Transformation3D placement3( 5, -5,  5,  0, 45,  0);
  Transformation3D placement4( 5,  5, -5,  0,  0, 45);
  Transformation3D placement5(-5, -5,  5, 45, 45,  0);
  Transformation3D placement6(-5,  5, -5, 45,  0, 45);
  Transformation3D placement7( 5, -5, -5,  0, 45, 45);
  Transformation3D placement8(-5, -5, -5, 45, 45, 45);

  world.PlaceDaughter("box1", &box, &placement1);
  world.PlaceDaughter("box2", &box, &placement2);
  world.PlaceDaughter("box3", &box, &placement3);
  world.PlaceDaughter("box4", &box, &placement4);
  world.PlaceDaughter("box5", &box, &placement5);
  world.PlaceDaughter("box6", &box, &placement6);
  world.PlaceDaughter("box7", &box, &placement7);
  world.PlaceDaughter("box8", &box, &placement8);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetPointCount(8);
  tester.RunBenchmark();

  return 0;
}