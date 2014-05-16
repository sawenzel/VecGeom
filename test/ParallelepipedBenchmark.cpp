#include "volumes/logical_volume.h"
#include "volumes/box.h"
#include "volumes/Parallelepiped.h"
#include "benchmarking/Benchmarker.h"
#include "management/geo_manager.h"

using namespace vecgeom;

int main() {

  UnplacedBox worldUnplaced = UnplacedBox(10., 10., 10.);
  UnplacedParallelepiped paraUnplaced =
      UnplacedParallelepiped(3., 3., 3., 14.9, 39, 3.22);
  LogicalVolume world = LogicalVolume("w0rld", &worldUnplaced);
  LogicalVolume para = LogicalVolume("p4r4", &paraUnplaced);
  world.PlaceDaughter(&para, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetPointCount(1<<13);
  tester.RunBenchmark();

  return 0;
}