#include "base/transformation_matrix.h"
#include "base/specialized_matrix.h"
#ifdef VECGEOM_CUDA
#include "backend/cuda_backend.cuh"
#endif

namespace vecgeom {

const TransformationMatrix TransformationMatrix::kIdentity =
    SpecializedMatrix<translation::kOrigin, rotation::kIdentity>();

TransformationMatrix::TransformationMatrix() {
  SetTranslation(0, 0, 0);
  SetRotation(1, 0, 0, 0, 1, 0, 0, 0, 1);
}

TransformationMatrix::TransformationMatrix(const Precision tx,
                                           const Precision ty,
                                           const Precision tz) {
  SetTranslation(tx, ty, tz);
  SetRotation(1, 0, 0, 0, 1, 0, 0, 0, 1);
}

TransformationMatrix::TransformationMatrix(
    const Precision tx, const Precision ty,
    const Precision tz, const Precision phi,
    const Precision theta, const Precision psi) {
  SetTranslation(tx, ty, tz);
  SetRotation(phi, theta, psi);
}

VECGEOM_CUDA_HEADER_BOTH
TransformationMatrix::TransformationMatrix(TransformationMatrix const &other) {
  SetTranslation(other.Translation(0), other.Translation(1),
                 other.Translation(2));
  SetRotation(other.Rotation(0), other.Rotation(1), other.Rotation(2),
              other.Rotation(3), other.Rotation(4), other.Rotation(5),
              other.Rotation(6), other.Rotation(7), other.Rotation(8));
}

VECGEOM_CUDA_HEADER_BOTH
void TransformationMatrix::SetTranslation(const Precision tx,
                                          const Precision ty,
                                          const Precision tz) {
  trans[0] = tx;
  trans[1] = ty;
  trans[2] = tz;
  SetProperties();
}

VECGEOM_CUDA_HEADER_BOTH
void TransformationMatrix::SetTranslation(Vector3D<Precision> const &vec) {
  SetTranslation(vec[0], vec[1], vec[2]);
}

VECGEOM_CUDA_HEADER_BOTH
void TransformationMatrix::SetProperties() {
  has_translation = (
    fabs(trans[0]) > kNearZero ||
    fabs(trans[1]) > kNearZero ||
    fabs(trans[2]) > kNearZero
  ) ? true : false;
  has_rotation = (GenerateRotationCode() == rotation::kIdentity)
                 ? false : true;
  identity = !has_translation && !has_rotation;
}

VECGEOM_CUDA_HEADER_BOTH
void TransformationMatrix::SetRotation(const Precision phi,
                                       const Precision theta,
                                       const Precision psi) {

  const Precision sinphi = sin(kDegToRad*phi);
  const Precision cosphi = cos(kDegToRad*phi);
  const Precision sinthe = sin(kDegToRad*theta);
  const Precision costhe = cos(kDegToRad*theta);
  const Precision sinpsi = sin(kDegToRad*psi);
  const Precision cospsi = cos(kDegToRad*psi);

  rot[0] =  cospsi*cosphi - costhe*sinphi*sinpsi;
  rot[1] = -sinpsi*cosphi - costhe*sinphi*cospsi;
  rot[2] =  sinthe*sinphi;
  rot[3] =  cospsi*sinphi + costhe*cosphi*sinpsi;
  rot[4] = -sinpsi*sinphi + costhe*cosphi*cospsi;
  rot[5] = -sinthe*cosphi;
  rot[6] =  sinpsi*sinthe;
  rot[7] =  cospsi*sinthe;
  rot[8] =  costhe;

  SetProperties();
}

VECGEOM_CUDA_HEADER_BOTH
void TransformationMatrix::SetRotation(Vector3D<Precision> const &vec) {
  SetRotation(vec[0], vec[1], vec[2]);
}

VECGEOM_CUDA_HEADER_BOTH
void TransformationMatrix::SetRotation(
    const Precision rot0, const Precision rot1, const Precision rot2,
    const Precision rot3, const Precision rot4, const Precision rot5,
    const Precision rot6, const Precision rot7, const Precision rot8) {

  rot[0] = rot0;
  rot[1] = rot1;
  rot[2] = rot2;
  rot[3] = rot3;
  rot[4] = rot4;
  rot[5] = rot5;
  rot[6] = rot6;
  rot[7] = rot7;
  rot[8] = rot8;

  SetProperties();
}

VECGEOM_CUDA_HEADER_BOTH
RotationCode TransformationMatrix::GenerateRotationCode() const {
  int code = 0;
  for (int i = 0; i < 9; ++i) {
    // Assign each bit
    code |= (1<<i) * (fabs(rot[i]) > kNearZero);
  }
  if (code == rotation::kDiagonal
      && (rot[0] == 1. && rot[4] == 1. && rot[8] == 1.)) {
    code = rotation::kIdentity;
  }
  return code;
}

/**
 * Very simple translation code. Kept as an integer in case other cases are to
 * be implemented in the future.
 * /return The matrix' translation code, which is 0 for matrices without
 * translation and 1 otherwise.
 */
VECGEOM_CUDA_HEADER_BOTH
TranslationCode TransformationMatrix::GenerateTranslationCode() const {
  return (has_translation) ? translation::kTranslation : translation::kOrigin;
}

VECGEOM_CUDA_HEADER_HOST
std::ostream& operator<<(std::ostream& os, TransformationMatrix const &matrix) {
  os << "Matrix {" << matrix.Translation() << ", "
     << "("  << matrix.Rotation(0) << ", " << matrix.Rotation(1)
     << ", " << matrix.Rotation(2) << ", " << matrix.Rotation(3)
     << ", " << matrix.Rotation(4) << ", " << matrix.Rotation(5)
     << ", " << matrix.Rotation(6) << ", " << matrix.Rotation(7)
     << ", " << matrix.Rotation(8) << ")}";
  return os;
}

#ifdef VECGEOM_CUDA

namespace {

__global__
void ConstructOnGpu(const TransformationMatrix matrix,
                    TransformationMatrix *const gpu_ptr) {
  new(gpu_ptr) TransformationMatrix(matrix);
}

} // End anonymous namespace

TransformationMatrix* TransformationMatrix::CopyToGpu(
    TransformationMatrix *const gpu_ptr) const {

  ConstructOnGpu<<<1, 1>>>(*this, gpu_ptr);
  CudaAssertError();
  return gpu_ptr;

}

TransformationMatrix* TransformationMatrix::CopyToGpu() const {

  TransformationMatrix *const gpu_ptr = AllocateOnGpu<TransformationMatrix>();
  return CopyToGpu(gpu_ptr);

}

#endif

} // End namespace vecgeom