#include "volumes/logical_volume.h"
#include "volumes/box.h"
#include "benchmarking/distance_to_in.h"

using namespace VECGEOM_NAMESPACE;

int main() {

  UnplacedBox world_params = UnplacedBox(4., 4., 4.);
  UnplacedBox largebox_params = UnplacedBox(1.5, 1.5, 1.5);
  UnplacedBox smallbox_params = UnplacedBox(0.5, 0.5, 0.5);

  LogicalVolume world = LogicalVolume(&world_params);
  VPlacedVolume *world_placed = world.Place();
  LogicalVolume largebox = LogicalVolume(&largebox_params);
  LogicalVolume smallbox = LogicalVolume(&smallbox_params);

  TransformationMatrix origin = TransformationMatrix();
  TransformationMatrix placement1 = TransformationMatrix( 2,  2,  2);
  TransformationMatrix placement2 = TransformationMatrix(-2,  2,  2);
  TransformationMatrix placement3 = TransformationMatrix( 2, -2,  2);
  TransformationMatrix placement4 = TransformationMatrix( 2,  2, -2);
  TransformationMatrix placement5 = TransformationMatrix(-2, -2,  2);
  TransformationMatrix placement6 = TransformationMatrix(-2,  2, -2);
  TransformationMatrix placement7 = TransformationMatrix( 2, -2, -2);
  TransformationMatrix placement8 = TransformationMatrix(-2, -2, -2);

  largebox.PlaceDaughter(&smallbox, &origin);
  world.PlaceDaughter(&largebox, &placement1);
  world.PlaceDaughter(&largebox, &placement2);
  world.PlaceDaughter(&largebox, &placement3);
  world.PlaceDaughter(&largebox, &placement4);
  world.PlaceDaughter(&largebox, &placement5);
  world.PlaceDaughter(&largebox, &placement6);
  world.PlaceDaughter(&largebox, &placement7);
  world.PlaceDaughter(&largebox, &placement8);

  DistanceToIn tester(world_placed);
  tester.set_verbose(2);
  tester.BenchmarkAll();

  delete world_placed;

  return 0;
}