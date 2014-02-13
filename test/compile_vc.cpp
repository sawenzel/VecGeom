#include "base/vector3d.h"
#include "base/soa3d.h"
#include "base/transformation_matrix.h"
#include "backend/vc_backend.h"
#include "volumes/kernel/box_kernel.h"
#include "volumes/logical_volume.h"
#include "volumes/box.h"

using namespace vecgeom;

void foo() {
  VcDouble scalar;
  Vector3D<double> scalar_v;
  Vector3D<VcDouble> vector_v;
  SOA3D<VcDouble> soa;
  TransformationMatrix<double> matrix;
  VcBool output_inside;
  VcDouble output_distance;
  UnplacedBox<double> world_box = UnplacedBox<double>(scalar_v);
  UnplacedBox<double> box = UnplacedBox<double>(scalar_v);
  VLogicalVolume<double> world = VLogicalVolume<double>(world_box);
  world.PlaceDaughter(box, matrix);
  BoxInside<translation::kOrigin, rotation::kIdentity, kVc>(
    scalar_v, matrix, vector_v, &output_inside
  );
  BoxDistanceToIn<translation::kOrigin, rotation::kIdentity, kVc>(
    scalar_v, matrix, vector_v, vector_v, scalar, &output_distance
  );
}