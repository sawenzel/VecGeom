#include "base/vector3d.h"
#include "base/soa3d.h"
#include "base/specialized_matrix.h"
#include "backend/cilk_backend.h"
#include "volumes/kernel/box_kernel.h"
#include "volumes/logical_volume.h"
#include "volumes/box.h"

using namespace vecgeom;

int main() {
  CilkPrecision scalar;
  Vector3D<double> scalar_v;
  Vector3D<CilkPrecision> vector_v;
  SOA3D<CilkPrecision> soa;
  TransformationMatrix matrix;
  CilkBool output_inside;
  CilkPrecision output_distance;
  UnplacedBox world_unplaced = UnplacedBox(scalar_v);
  UnplacedBox box_unplaced = UnplacedBox(scalar_v);
  LogicalVolume world = LogicalVolume(world_unplaced);
  LogicalVolume box = LogicalVolume(box_unplaced);
  world.PlaceDaughter(box, matrix);
  return 0;
}