#include "base/vector3d.h"
#include "base/soa3d.h"
#include "base/transformation_matrix.h"
#include "backend/cilk_backend.h"
#include "volumes/kernel/box_kernel.h"
#include "base/vc_vector3d.h"

using namespace vecgeom;

void compile_cilk() {
  CilkDouble scalar;
  Vector3D<double> scalar_v;
  Vector3D<CilkDouble> vector_v;
  SOA3D<CilkDouble> soa;
  TransformationMatrix<double> matrix;
  CilkBool output_inside;
  CilkDouble output_distance;
  BoxInside<kCilk>(scalar_v, matrix, vector_v, &output_inside);
  BoxDistanceToIn<kCilk>(scalar_v, matrix, vector_v, vector_v, scalar,
                         &output_distance);
}