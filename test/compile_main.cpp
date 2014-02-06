#include "base/soa3d.h"
#include "base/vector3d.h"
#include "base/transformation_matrix.h"
#include "backend/scalar_backend.h"
#include "volumes/box.h"
#include "volumes/kernel/box_kernel.h"

using namespace vecgeom;

int main() {
  double scalar = 0;
  Vector3D<double> scalar_v;
  Vector3D<double> vector_v;
  SOA3D<double> soa;
  TransformationMatrix<double> matrix;
  ScalarBool output_inside;
  double output_distance;
  BoxInside<kScalar>(scalar_v, matrix, vector_v, &output_inside);
  BoxDistanceToIn<kScalar>(scalar_v, matrix, vector_v, vector_v, scalar,
                           &output_distance);
  return 0;
}