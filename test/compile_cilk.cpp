#include "base/vector3d.h"
#include "base/soa3d.h"
#include "base/trans_matrix.h"
#include "backend/cilk_backend.h"
#include "volumes/kernel/box_kernel.h"

using namespace vecgeom;

void foo() {
  CilkDouble scalar;
  Vector3D<double> scalar_v;
  Vector3D<CilkDouble> vector_v;
  SOA3D<CilkDouble> soa;
  TransMatrix matrix;
  CilkBool output_inside;
  CilkDouble output_distance;
  BoxInside<kCilk>(scalar_v, matrix, vector_v, &output_inside);
  BoxDistanceToIn<kCilk>(scalar_v, matrix, vector_v, vector_v, scalar,
                         &output_distance);
}