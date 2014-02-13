#include "base/soa3d.h"
#include "backend/scalar_backend.h"
#include "volumes/box.h"
#include "volumes/logical_volume.h"

using namespace vecgeom;

int main() {
  double scalar = 0;
  Vector3D<double> scalar_v;
  Vector3D<double> vector_v;
  SOA3D<double> soa;
  TransformationMatrix<double> matrix;
  ScalarBool output_inside;
  double output_distance;
  UnplacedBox<double> world_box = UnplacedBox<double>(scalar_v);
  UnplacedBox<double> box = UnplacedBox<double>(scalar_v);
  VLogicalVolume<double> world = VLogicalVolume<double>(world_box);
  world.PlaceDaughter(box, matrix);
  BoxInside<translation::kOrigin, rotation::kIdentity, kScalar>(
    scalar_v, matrix, vector_v, &output_inside
  );
  BoxDistanceToIn<translation::kOrigin, rotation::kIdentity, kScalar>(
    scalar_v, matrix, vector_v, vector_v, scalar, &output_distance
  );
  return 0;
}