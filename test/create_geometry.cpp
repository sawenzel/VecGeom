#include "management/cuda_manager.h"
#include "volumes/logical_volume.h"
#include "volumes/box.h"

using namespace vecgeom;

void CudaCopy(LogicalVolume const *const world);

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

  std::cerr << "Printing world content:\n";
  world.PrintContent();

  #ifdef VECGEOM_CUDA
  CudaCopy(&world);
  #endif

  return 0;
}

#ifdef VECGEOM_CUDA
__global__
void CudaContent(LogicalVolume const *world) {
  printf("Inside CUDA kernel.\n");
  world->PrintContent();
}

void CudaCopy(LogicalVolume const *const world) {
  CudaManager::Instance().set_verbose(3);
  CudaManager::Instance().LoadGeometry(world);
  CudaManager::Instance().Synchronize();
  LogicalVolume const *const world_gpu = CudaManager::Instance().world_gpu();
  CudaContent<<<1, 1>>>(world_gpu);
  cudaDeviceSynchronize(); // Necessary to print output
  CudaAssertError();
}
#endif
