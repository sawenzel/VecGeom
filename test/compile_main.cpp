#include "base/soa3d.h"
#include "backend/scalar_backend.h"
#include "volumes/box.h"
#include "volumes/logical_volume.h"

using namespace vecgeom;

int main() {
  Vector3D<double> scalar_v;
  SOA3D<double> soa;
  TransformationMatrix matrix;
  UnplacedBox world_unplaced = UnplacedBox(scalar_v);
  UnplacedBox box_unplaced = UnplacedBox(scalar_v);
  VLogicalVolume world = VLogicalVolume(world_unplaced);
  VLogicalVolume box = VLogicalVolume(box_unplaced);
  world.PlaceDaughter(box, matrix);
  return 0;
}