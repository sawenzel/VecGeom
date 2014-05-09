#include "volumes/logical_volume.h"
#include "volumes/box.h"
#include "benchmarking/ToInBenchmarker.h"
#include "management/geo_manager.h"

using namespace vecgeom;

int main() {

  UnplacedBox world_params = UnplacedBox(4., 4., 4.);
  UnplacedBox largebox_params = UnplacedBox(1.5, 1.5, 1.5);
  UnplacedBox smallbox_params = UnplacedBox(0.5, 0.5, 0.5);

  LogicalVolume worldl = LogicalVolume(&world_params);

  LogicalVolume largebox = LogicalVolume(&largebox_params);
  LogicalVolume smallbox = LogicalVolume(&smallbox_params);

  Transformation3D origin = Transformation3D();
  Transformation3D placement1 = Transformation3D( 2,  2,  2);
  Transformation3D placement2 = Transformation3D(-2,  2,  2);
  Transformation3D placement3 = Transformation3D( 2, -2,  2);
  Transformation3D placement4 = Transformation3D( 2,  2, -2);
  Transformation3D placement5 = Transformation3D(-2, -2,  2);
  Transformation3D placement6 = Transformation3D(-2,  2, -2);
  Transformation3D placement7 = Transformation3D(2, -2, -2);
  Transformation3D placement8 = Transformation3D(-2, -2, -2);

  largebox.PlaceDaughter(&smallbox, &origin);
  worldl.PlaceDaughter(&largebox, &placement1);
  worldl.PlaceDaughter(&largebox, &placement2);
  worldl.PlaceDaughter(&largebox, &placement3);
  worldl.PlaceDaughter(&largebox, &placement4);
  worldl.PlaceDaughter(&largebox, &placement5);
  worldl.PlaceDaughter(&largebox, &placement6);
  worldl.PlaceDaughter(&largebox, &placement7);
  worldl.PlaceDaughter(&largebox, &placement8);

  VPlacedVolume *world_placed = worldl.Place();

  GeoManager::Instance().set_world(world_placed);

  ToInBenchmarker tester(GeoManager::Instance().world());
  tester.SetVerbose(2);
  tester.SetPointCount(1<<10);
  tester.BenchmarkAll();

  return 0;
}