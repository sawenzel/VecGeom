#include "base/vector3d.h"
#include "base/soa3d.h"
#include "base/transformation_matrix.h"
#include "backend/vc_backend.h"
#include "volumes/kernel/box_kernel.h"
#include "volumes/logical_volume.h"
#include "volumes/box.h"

using namespace vecgeom;

void foo() {
  VcPrecision scalar;
  Vector3D<double> scalar_v;
  Vector3D<VcPrecision> vector_v;
  SOA3D<VcPrecision> soa;
  TransformationMatrix matrix;
  VcBool output_inside;
  VcPrecision output_distance;
  UnplacedBox world_unplaced = UnplacedBox(scalar_v);
  UnplacedBox box_unplaced = UnplacedBox(scalar_v);
  VLogicalVolume world = VLogicalVolume(world_unplaced);
  VLogicalVolume box = VLogicalVolume(box_unplaced);
  world.PlaceDaughter(box, matrix);
}