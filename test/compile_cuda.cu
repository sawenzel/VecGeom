#include "base/soa3d.h"
#include "backend/cuda_backend.cuh"
#include "volumes/kernel/box_kernel.h"
#include "volumes/logical_volume.h"
#include "volumes/box.h"
#include "management/cuda_manager.h"

using namespace vecgeom;

__device__
void compile_test() {
  CudaPrecision scalar;
  Vector3D<double> scalar_v;
  Vector3D<CudaPrecision> vector_v;
  SOA3D<CudaPrecision> soa;
  TransformationMatrix matrix;
  CudaBool output_inside;
  CudaPrecision output_distance;
  BoxInside<translation::kOrigin, rotation::kIdentity, kCuda>(
    scalar_v, matrix, vector_v, &output_inside
  );
  BoxDistanceToIn<translation::kOrigin, rotation::kIdentity, kCuda>(
    scalar_v, matrix, vector_v, vector_v, scalar, &output_distance
  );
}

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