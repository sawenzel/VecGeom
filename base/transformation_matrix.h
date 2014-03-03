#ifndef VECGEOM_BASE_TRANSMATRIX_H_
#define VECGEOM_BASE_TRANSMATRIX_H_

#include <cmath>
#include "base/types.h"
#include "base/vector3d.h"

namespace vecgeom {

// enum MatrixEntry {
//   k00 = 0x001, k01 = 0x002, k02 = 0x004,
//   k10 = 0x008, k11 = 0x010, k12 = 0x020,
//   k20 = 0x040, k21 = 0x080, k22 = 0x100
// };

typedef int RotationCode;
typedef int TranslationCode;
namespace rotation {
  enum RotationId { kDiagonal = 0x111, kIdentity = 0x200 };
}
namespace translation {
  enum TranslationId { kOrigin = 0, kTranslation = 1 };
}

class TransformationMatrix {

private:

  Precision trans[3];
  Precision rot[9];
  bool identity;
  bool has_rotation;
  bool has_translation;

public:

  TransformationMatrix();

  TransformationMatrix(const Precision tx, const Precision ty,
                       const Precision tz);

  TransformationMatrix(const Precision tx, const Precision ty,
                       const Precision tz, const Precision phi,
                       const Precision theta, const Precision psi);

  VECGEOM_CUDA_HEADER_BOTH
  TransformationMatrix(TransformationMatrix const &other);

  // Accessors

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> Translation() const {
    return Vector3D<Precision>(trans[0], trans[1], trans[2]);
  }

  /**
   * No safety against faulty indexing.
   * \param index Index of translation entry in the range [0-2].
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Translation(const int index) const { return trans[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision const* Rotation() const { return rot; }

  /**
   * No safety against faulty indexing.
   * \param index Index of rotation entry in the range [0-8].
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Rotation(const int index) const { return rot[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool IsIdentity() const { return identity; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool HasRotation() const { return has_rotation; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool HasTranslation() const { return has_translation; }

  // Mutators

  VECGEOM_CUDA_HEADER_BOTH
  void SetTranslation(const Precision tx, const Precision ty,
                      const Precision tz);

  VECGEOM_CUDA_HEADER_BOTH
  void SetTranslation(Vector3D<Precision> const &vec);

  VECGEOM_CUDA_HEADER_BOTH
  void SetProperties();

  VECGEOM_CUDA_HEADER_BOTH
  void SetRotation(const Precision phi, const Precision theta,
                   const Precision psi);

  VECGEOM_CUDA_HEADER_BOTH
  void SetRotation(Vector3D<Precision> const &vec);

  VECGEOM_CUDA_HEADER_BOTH
  void SetRotation(const Precision rot0, const Precision rot1,
                   const Precision rot2, const Precision rot3,
                   const Precision rot4, const Precision rot5,
                   const Precision rot6, const Precision rot7,
                   const Precision rot8);

  // Generation of template parameter codes

  VECGEOM_CUDA_HEADER_BOTH
  RotationCode GenerateRotationCode() const;

  VECGEOM_CUDA_HEADER_BOTH
  TranslationCode GenerateTranslationCode() const;

private:

  // Templated rotation and translation methods which inline and compile to
  // optimized versions.

  template <RotationCode code, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void DoRotation(Vector3D<InputType> const &master,
                  Vector3D<InputType> *const local) const;

  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void DoTranslation(Vector3D<InputType> const &master,
                     Vector3D<InputType> *const local) const;

public:

  // Transformation interface

  template <TranslationCode trans_code, RotationCode rot_code,
            typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Transform(Vector3D<InputType> const &master,
                 Vector3D<InputType> *const local) const;

  template <TranslationCode trans_code, RotationCode rot_code,
            typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<InputType> Transform(Vector3D<InputType> const &master) const;

  template <RotationCode code, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void TransformRotation(Vector3D<InputType> const &master,
                         Vector3D<InputType> *const local) const;

  template <RotationCode code, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<InputType> TransformRotation(
      Vector3D<InputType> const &master) const;

  // Utility and CUDA

  VECGEOM_CUDA_HEADER_HOST
  friend std::ostream& operator<<(std::ostream& os,
                                  TransformationMatrix const &v);

  #ifdef VECGEOM_CUDA
  TransformationMatrix* CopyToGpu() const;
  TransformationMatrix* CopyToGpu(TransformationMatrix *const gpu_ptr) const;
  #endif

}; // End class TransformationMatrix


/**
 * Rotates a vector to this matrix' frame of reference. Templates on the
 * RotationCode generated by GenerateTranslationCode() to perform specialized 
 * rotation.
 * \sa GenerateTranslationCode()
 * \param master Vector in original frame of reference.
 * \param local Output vector rotated to the new frame of reference.
 */
template <RotationCode code, typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void TransformationMatrix::DoRotation(Vector3D<InputType> const &master,
                                      Vector3D<InputType> *const local) const {

  if (code == 0x1B1) {
    (*local)[0] = master[0]*rot[0];
    (*local)[1] = master[1]*rot[4] + master[2]*rot[7];
    (*local)[2] = master[1]*rot[5] + master[2]*rot[8];
    return;
  }
  if (code == 0x18E) {
    (*local)[0] = master[1]*rot[3];
    (*local)[1] = master[0]*rot[1] + master[2]*rot[7];
    (*local)[2] = master[0]*rot[2] + master[2]*rot[8];
    return;
  }
  if (code == 0x076){
    (*local)[0] = master[2]*rot[6];
    (*local)[1] = master[0]*rot[1] + master[1]*rot[4];
    (*local)[2] = master[0]*rot[2] + master[1]*rot[5];
    return;
  }
  if (code == 0x16A) {
    (*local)[0] = master[1]*rot[3] + master[2]*rot[6];
    (*local)[1] = master[0]*rot[1];
    (*local)[2] = master[2]*rot[5] + master[2]*rot[8];
    return;
  }
  if (code == 0x155) {
    (*local)[0] = master[0]*rot[0] + master[2]*rot[6];
    (*local)[1] = master[1]*rot[4];
    (*local)[2] = master[0]*rot[2] + master[2]*rot[8];
    return;
  }
  if (code == 0x0AD){
    (*local)[0] = master[0]*rot[0] + master[1]*rot[3];
    (*local)[1] = master[2]*rot[7];
    (*local)[2] = master[0]*rot[2] + master[1]*rot[5];
    return;
  }
  if (code == 0x0DC){
    (*local)[0] = master[1]*rot[3] + master[2]*rot[6];
    (*local)[1] = master[1]*rot[4] + master[2]*rot[7];
    (*local)[2] = master[0]*rot[2];
    return;
  }
  if (code == 0x0E3) {
    (*local)[0] = master[0]*rot[0] + master[2]*rot[6];
    (*local)[1] = master[0]*rot[1] + master[2]*rot[7];
    (*local)[2] = master[1]*rot[5];
    return;
  }
  if (code == 0x11B){
    (*local)[0] = master[0]*rot[0] + master[1]*rot[3];
    (*local)[1] = master[0]*rot[1] + master[1]*rot[4];
    (*local)[2] = master[2]*rot[8];
    return;
  }
  if (code == 0x0A1){
    (*local)[0] = master[0]*rot[0];
    (*local)[1] = master[2]*rot[7];
    (*local)[2] = master[1]*rot[5];
    return;
  }
  if (code == 0x10A){
    (*local)[0] = master[1]*rot[3];
    (*local)[1] = master[0]*rot[1];
    (*local)[2] = master[2]*rot[8];
    return;
  }
  if (code == 0x046){
    (*local)[0] = master[1]*rot[3];
    (*local)[1] = master[2]*rot[7];
    (*local)[2] = master[0]*rot[2];
    return;
  }
  if (code == 0x062) {
    (*local)[0] = master[2]*rot[6];
    (*local)[1] = master[0]*rot[1];
    (*local)[2] = master[1]*rot[5];
    return;
  }
  if (code == 0x054) {
    (*local)[0] = master[2]*rot[6];
    (*local)[1] = master[1]*rot[4];
    (*local)[2] = master[0]*rot[2];
    return;
  }

  // code = 0x111;
  if (code == rotation::kDiagonal) {
    (*local)[0] = master[0]*rot[0];
    (*local)[1] = master[1]*rot[4];
    (*local)[2] = master[2]*rot[8];
    return;
  }

  // code = 0x200;
  if (code == rotation::kIdentity){
    *local = master;
    return;
  }

  // General case
  (*local)[0] =  master[0]*rot[0];
  (*local)[1] =  master[0]*rot[1];
  (*local)[2] =  master[0]*rot[2];
  (*local)[0] += master[1]*rot[3];
  (*local)[1] += master[1]*rot[4];
  (*local)[2] += master[1]*rot[5];
  (*local)[0] += master[2]*rot[6];
  (*local)[1] += master[2]*rot[7];
  (*local)[2] += master[2]*rot[8];

}

template <typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void TransformationMatrix::DoTranslation(
    Vector3D<InputType> const &master,
    Vector3D<InputType> *const local) const {

  (*local)[0] = master[0] - trans[0];
  (*local)[1] = master[1] - trans[1];
  (*local)[2] = master[2] - trans[2];

}

/**
 * Transform a point to the local reference frame.
 * \param master Point to be transformed.
 * \param local Output destination. Should never be the same as the input
 *              vector!
 */
template <TranslationCode trans_code, RotationCode rot_code,
          typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void TransformationMatrix::Transform(Vector3D<InputType> const &master,
                                     Vector3D<InputType> *const local) const {

  // Identity
  if (trans_code == 0 && rot_code == rotation::kIdentity) {
    *local = master;
    return;
  }

  // Only translation
  if (trans_code == 1 && rot_code == rotation::kIdentity) {
    DoTranslation(master, local);
    return;
  }

  // Only rotation
  if (trans_code == 0 && rot_code != rotation::kIdentity) {
    DoRotation<rot_code>(master, local);
    return;
  }

  // General case
  DoTranslation(master, local);
  DoRotation<rot_code>(master, local);

}

/**
 * Since transformation cannot be done in place, allows the transformed vector
 * to be constructed by Transform directly.
 * \param master Point to be transformed.
 * \return Newly constructed Vector3D with the transformed coordinates.
 */
template <TranslationCode trans_code, RotationCode rot_code,
          typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Vector3D<InputType> TransformationMatrix::Transform(
    Vector3D<InputType> const &master) const {

  Vector3D<InputType> local;
  Transform<trans_code, rot_code>(master, &local);
  return local;

}

/**
 * Only transforms by rotation, ignoring the translation part. This is useful
 * when transforming directions.
 * \param master Point to be transformed.
 * \param local Output destination of transformation.
 */
template <RotationCode code, typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void TransformationMatrix::TransformRotation(
    Vector3D<InputType> const &master,
    Vector3D<InputType> *const local) const {

  // Rotational identity
  if (code == rotation::kIdentity) {
    *local = master;
    return;
  }

  // General case
  DoRotation<code>(master, local);

}

/**
 * Since transformation cannot be done in place, allows the transformed vector
 * to be constructed by TransformRotation directly.
 * \param master Point to be transformed.
 * \return Newly constructed Vector3D with the transformed coordinates.
 */
template <RotationCode code, typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Vector3D<InputType> TransformationMatrix::TransformRotation(
    Vector3D<InputType> const &master) const {

  Vector3D<InputType> local;
  TransformRotation<code>(master, &local);
  return local;

}

} // End namespace vecgeom

#endif // VECGEOM_BASE_TRANSMATRIX_H_