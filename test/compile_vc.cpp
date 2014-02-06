#include "base/vector3d.h"
#include "base/soa3d.h"
#include "base/trans_matrix.h"
#include "backend/vc_backend.h"
#include "volumes/kernel/box_kernel.h"

using namespace vecgeom;

void foo() {
  VcDouble scalar;
  Vector3D<double> scalar_v;
  Vector3D<VcDouble> vector_v;
  SOA3D<VcDouble> soa;
  TransMatrix<double> matrix;
  VcBool output_inside;
  VcDouble output_distance;
  BoxInside<kVc>(scalar_v, matrix, vector_v, &output_inside);
  BoxDistanceToIn<kVc>(scalar_v, matrix, vector_v, vector_v, scalar,
                       &output_distance);
}