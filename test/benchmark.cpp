#include "volumes/logical_volume.h"
#include "volumes/box.h"
#include "benchmarking/distance_to_in.h"
#include "management/geo_manager.h"

using namespace vecgeom;

int main() {

  UnplacedBox world_params = UnplacedBox(4., 4., 4.);
  UnplacedBox largebox_params = UnplacedBox(1.5, 1.5, 1.5);
  UnplacedBox smallbox_params = UnplacedBox(0.5, 0.5, 0.5);

  LogicalVolume worldl = LogicalVolume(&world_params);

  LogicalVolume largebox = LogicalVolume(&largebox_params);
  LogicalVolume smallbox = LogicalVolume(&smallbox_params);

  TransformationMatrix origin = TransformationMatrix();
  TransformationMatrix placement1 = TransformationMatrix( 2,  2,  2, 45, 0, 0);
  TransformationMatrix placement2 = TransformationMatrix(-2,  2,  2, 0, 45, 0);
  TransformationMatrix placement3 = TransformationMatrix( 2, -2,  2, 0, 0, 45);
  TransformationMatrix placement4 = TransformationMatrix( 2,  2, -2, 45, 45, 0);
  TransformationMatrix placement5 = TransformationMatrix(-2, -2,  2, 45, 0, 45);
  TransformationMatrix placement6 = TransformationMatrix(-2,  2, -2, 0, 45, 45);
  TransformationMatrix placement7 = TransformationMatrix(2, -2, -2, 45, 45, 45);
  TransformationMatrix placement8 = TransformationMatrix(-2, -2, -2, 0, 0, 0);

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

  DistanceToInBenchmarker tester(GeoManager::Instance().world());
  tester.set_verbose(2);
  tester.set_repetitions(1<<10);
  tester.set_n_points(1<<10);
  tester.BenchmarkAll();

  return 0;
}