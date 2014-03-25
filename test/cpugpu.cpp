#include "base/stopwatch.h"
#include "management/cuda_manager.h"
#include "navigation/navigationstate.h"
#include "navigation/simple_navigator.h"
#include "volumes/box.h"
#include "volumes/utilities/volume_utilities.h"

using namespace vecgeom;

int main() {

  UnplacedBox world_params = UnplacedBox(4., 4., 4.);
  UnplacedBox largebox_params = UnplacedBox(1.5, 1.5, 1.5);
  UnplacedBox smallbox_params = UnplacedBox(0.5, 0.5, 0.5);

  LogicalVolume world = LogicalVolume(&world_params);

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

  VPlacedVolume *const world_placed = world.Place();

  CudaManager::Instance().LoadGeometry(world_placed);
  CudaManager::Instance().Synchronize();
  CudaManager::Instance().PrintGeometry();

  const int n = 1<<16;
  const int depth = 3;

  SOA3D<Precision> points(n);
  volumeutilities::FillRandomPoints(*world_placed, points);
  int *const results = new int[n]; 
  int *const results_gpu = new int[n]; 

  SimpleNavigator navigator;
  Stopwatch sw;
  sw.Start();
  for (int i = 0; i < n; ++i) {
    NavigationState path(depth);
    results[i] =
        navigator.LocatePoint(world_placed, points[i], path, true)->id();
  }
  const double cpu = sw.Stop();
  std::cout << "Points located on CPU in " << cpu << "s.\n";

  sw.Start();
  CudaManager::Instance().LocatePoints(points, depth, results_gpu);
  const double gpu = sw.Stop();
  std::cout << "Points located on GPU in " << gpu
            << "s (including memory transfer).\n";

  // Compare output
  for (int i = 0; i < n; ++i) {
    // std::cout << results[i] << " vs. " << results_gpu[i] << std::endl;
    assert(results[i] == results_gpu[i]);
  }
  std::cout << "All points located within same volume on CPU and GPU.\n";

  return 0;
}