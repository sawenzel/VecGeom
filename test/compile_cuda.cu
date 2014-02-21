#include "base/soa3d.h"
#include "base/specialized_matrix.h"
#include "backend/cuda_backend.cuh"
#include "volumes/kernel/box_kernel.h"
#include "volumes/logical_volume.h"
#include "volumes/box.h"
#include "management/cuda_manager.h"

using namespace vecgeom;

__host__
int main() {
  Vector3D<double> scalar_v;
  SOA3D<double> soa;
  TransformationMatrix matrix;
  UnplacedBox world_unplaced = UnplacedBox(scalar_v);
  UnplacedBox box_unplaced = UnplacedBox(scalar_v);
  LogicalVolume world = LogicalVolume(world_unplaced);
  LogicalVolume box = LogicalVolume(box_unplaced);
  world.PlaceDaughter(box, matrix);
  CudaManager::Instance().LoadGeometry(world);
  CudaManager::Instance().PrintContent();
  return 0;
}