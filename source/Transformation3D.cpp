/**
 * @file transformation3d.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include "backend/backend.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "backend/cuda/interface.h"
#endif
#include "base/transformation3d.h"
#include "base/specialized_transformation3d.h"

#include <stdio.h>
#include <sstream>

namespace VECGEOM_NAMESPACE {

const Transformation3D Transformation3D::kIdentity =
    SpecializedTransformation3D<translation::kIdentity, rotation::kIdentity>();

Transformation3D::Transformation3D() {
  SetTranslation(0, 0, 0);
  SetRotation(1, 0, 0, 0, 1, 0, 0, 0, 1);
  SetProperties();
}

Transformation3D::Transformation3D(const Precision tx,
                                   const Precision ty,
                                   const Precision tz) {
  SetTranslation(tx, ty, tz);
  SetRotation(1, 0, 0, 0, 1, 0, 0, 0, 1);
  SetProperties();
}

Transformation3D::Transformation3D(
    const Precision tx, const Precision ty,
    const Precision tz, const Precision phi,
    const Precision theta, const Precision psi) {
  SetTranslation(tx, ty, tz);
  SetRotation(phi, theta, psi);
  SetProperties();
}

Transformation3D::Transformation3D(
    const Precision tx, const Precision ty, const Precision tz,
    const Precision r0, const Precision r1, const Precision r2,
    const Precision r3, const Precision r4, const Precision r5,
    const Precision r6, const Precision r7, const Precision r8) {
  SetTranslation(tx, ty, tz);
  SetRotation(r0, r1, r2, r3, r4, r5, r6, r7, r8);
  SetProperties();
}

VECGEOM_CUDA_HEADER_BOTH
Transformation3D::Transformation3D(Transformation3D const &other) {
  SetTranslation(other.Translation(0), other.Translation(1),
                 other.Translation(2));
  SetRotation(other.Rotation(0), other.Rotation(1), other.Rotation(2),
              other.Rotation(3), other.Rotation(4), other.Rotation(5),
              other.Rotation(6), other.Rotation(7), other.Rotation(8));
  SetProperties();
}

VECGEOM_CUDA_HEADER_BOTH
void Transformation3D::Print() const {
  printf("Transformation3D {{%.2f, %.2f, %.2f}, ", trans[0], trans[1],
         trans[2]);
  printf("{%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f}}", rot[0],
         rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8]);
}

VECGEOM_CUDA_HEADER_BOTH
void Transformation3D::SetTranslation(const Precision tx,
                                          const Precision ty,
                                          const Precision tz) {
  trans[0] = tx;
  trans[1] = ty;
  trans[2] = tz;
}

VECGEOM_CUDA_HEADER_BOTH
void Transformation3D::SetTranslation(Vector3D<Precision> const &vec) {
  SetTranslation(vec[0], vec[1], vec[2]);
}

VECGEOM_CUDA_HEADER_BOTH
void Transformation3D::SetProperties() {
  has_translation = (
    fabs(trans[0]) > kTolerance ||
    fabs(trans[1]) > kTolerance ||
    fabs(trans[2]) > kTolerance
  ) ? true : false;
  has_rotation = (GenerateRotationCode() == rotation::kIdentity)
                 ? false : true;
  identity = !has_translation && !has_rotation;
}


VECGEOM_CUDA_HEADER_BOTH
void Transformation3D::SetRotation(const Precision phi,
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
}

VECGEOM_CUDA_HEADER_BOTH
void Transformation3D::SetRotation(Vector3D<Precision> const &vec) {
  SetRotation(vec[0], vec[1], vec[2]);
}

VECGEOM_CUDA_HEADER_BOTH
void Transformation3D::SetRotation(
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
}

VECGEOM_CUDA_HEADER_BOTH
RotationCode Transformation3D::GenerateRotationCode() const {
  int code = 0;
  for (int i = 0; i < 9; ++i) {
    // Assign each bit
    code |= (1<<i) * (fabs(rot[i]) > kTolerance);
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
 * /return The transformation's translation code, which is 0 for transformations
 *         without translation and 1 otherwise.
 */
VECGEOM_CUDA_HEADER_BOTH
TranslationCode Transformation3D::GenerateTranslationCode() const {
  return (has_translation) ? translation::kGeneric : translation::kIdentity;
}

std::ostream& operator<<(std::ostream& os,
                         Transformation3D const &transformation) {
  os << "Transformation {" << transformation.Translation() << ", "
     << "("  << transformation.Rotation(0) << ", " << transformation.Rotation(1)
     << ", " << transformation.Rotation(2) << ", " << transformation.Rotation(3)
     << ", " << transformation.Rotation(4) << ", " << transformation.Rotation(5)
     << ", " << transformation.Rotation(6) << ", " << transformation.Rotation(7)
     << ", " << transformation.Rotation(8) << ")}"
     << "; identity(" << transformation.IsIdentity() << "); rotation("
     << transformation.HasRotation() << ")";
  return os;
}

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void Transformation3D_CopyToGpu(
  const Precision tx, const Precision ty, const Precision tz,
  const Precision r0, const Precision r1, const Precision r2,
  const Precision r3, const Precision r4, const Precision r5,
  const Precision r6, const Precision r7, const Precision r8,
  Transformation3D *const gpu_ptr);

Transformation3D* Transformation3D::CopyToGpu(
    Transformation3D *const gpu_ptr) const {

  Transformation3D_CopyToGpu(trans[0], trans[1], trans[2], rot[0], rot[1],
                             rot[2], rot[3], rot[4], rot[5], rot[6],
                             rot[7], rot[8], gpu_ptr);
  vecgeom::CudaAssertError();
  return gpu_ptr;

}

Transformation3D* Transformation3D::CopyToGpu() const {

  Transformation3D *const gpu_ptr =
      vecgeom::AllocateOnGpu<Transformation3D>();
  return this->CopyToGpu(gpu_ptr);

}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class Transformation3D;

__global__
void ConstructOnGpu(const Precision tx, const Precision ty, const Precision tz,
                    const Precision r0, const Precision r1, const Precision r2,
                    const Precision r3, const Precision r4, const Precision r5,
                    const Precision r6, const Precision r7, const Precision r8,
                    Transformation3D *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::Transformation3D(tx, ty, tz, r0, r1, r2,
                                                  r3, r4, r5, r6, r7, r8);
}

void Transformation3D_CopyToGpu(
    const Precision tx, const Precision ty, const Precision tz,
    const Precision r0, const Precision r1, const Precision r2,
    const Precision r3, const Precision r4, const Precision r5,
    const Precision r6, const Precision r7, const Precision r8,
    Transformation3D *const gpu_ptr) {
  ConstructOnGpu<<<1, 1>>>(tx, ty, tz, r0, r1, r2, r3, r4, r5,
                           r6, r7, r8, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom
