#include "management/cuda_manager.h"
#include "volumes/logical_volume.h"
#include "volumes/box.h"
#ifdef VECGEOM_CUDA
#include "backend/cuda_backend.cuh"
#endif

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
  TransformationMatrix box1 = TransformationMatrix( 2,  2,  2);
  TransformationMatrix box2 = TransformationMatrix(-2,  2,  2);
  TransformationMatrix box3 = TransformationMatrix( 2, -2,  2);
  TransformationMatrix box4 = TransformationMatrix( 2,  2, -2);
  TransformationMatrix box5 = TransformationMatrix(-2, -2,  2);
  TransformationMatrix box6 = TransformationMatrix(-2,  2, -2);
  TransformationMatrix box7 = TransformationMatrix( 2, -2, -2);
  TransformationMatrix box8 = TransformationMatrix(-2, -2, -2);

  largebox.PlaceDaughter(&smallbox, &origin);
  world.PlaceDaughter(&largebox, &box1);
  world.PlaceDaughter(&largebox, &box2);
  world.PlaceDaughter(&largebox, &box3);
  world.PlaceDaughter(&largebox, &box4);
  world.PlaceDaughter(&largebox, &box5);
  world.PlaceDaughter(&largebox, &box6);
  world.PlaceDaughter(&largebox, &box7);
  world.PlaceDaughter(&largebox, &box8);

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
