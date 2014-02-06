#ifndef VECGEOM_BASE_TRANSMATRIX_H_
#define VECGEOM_BASE_TRANSMATRIX_H_

#include <cmath>
#include "base/types.h"
#include "base/vector3d.h"

namespace vecgeom {

// /**
//  * Binary mask to access two bits for each matrix element determining the sign.
//  */
// enum MatrixEntry {
//   k00 = 0x001, k01 = 0x002, k02 = 0x004,
//   k10 = 0x008, k11 = 0x010, k12 = 0x020,
//   k20 = 0x040, k21 = 0x080, k22 = 0x100
// };

enum RotationCodes { kDiagonal = 0x111, kIdentity = 0x200 };
typedef int RotationCode;
typedef int TranslationCode;

template <typename Type>
class TransformationMatrix {

private:

  Type trans[3];
  Type rot[9];
  bool identity;
  bool has_rotation;
  bool has_translation;

public:

  VECGEOM_CUDA_HEADER_BOTH
  TransformationMatrix() {
    SetTranslation(0, 0, 0);
    SetRotation(0, 0, 0);
  }

  VECGEOM_CUDA_HEADER_BOTH
  TransformationMatrix(const Type tx, const Type ty, const Type tz,
              const Type phi, const Type theta,
              const Type psi) {
    SetTranslation(tx, ty, tz);
    SetRotation(phi, theta, psi);
  }

  VECGEOM_CUDA_HEADER_BOTH
  TransformationMatrix(TransformationMatrix const &other) {
    SetTranslation(other.Translation(0), other.Translation(1),
                   other.Translation(2));
    SetRotation(other.Rotation(0), other.Rotation(1), other.Rotation(2),
                other.Rotation(3), other.Rotation(4), other.Rotation(5),
                other.Rotation(6), other.Rotation(7), other.Rotation(8));
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Type> Translation() const {
    return Vector3D<Type>(trans[0], trans[1], trans[2]);
  }

  /**
   * No safety against faulty indexing.
   * \param index Index of translation entry in the range [0-2].
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Translation(const int index) const { return trans[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const* Rotation() const { return rot; }

  /**
   * No safety against faulty indexing.
   * \param index Index of rotation entry in the range [0-8].
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Rotation(const int index) const { return rot[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool IsIdentity() const { return identity; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool HasRotation() const { return has_rotation; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool HasTranslation() const { return has_translation; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SetTranslation(const Type tx, const Type ty,
                      const Type tz) {
    trans[0] = tx;
    trans[1] = ty;
    trans[2] = tz;
    SetProperties();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SetTranslation(Vector3D<Type> const &vec) {
    SetTranslation(vec[0], vec[1], vec[2]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SetProperties() {
    has_translation = (trans[0] || trans[1] || trans[2]) ? true : false;
    has_rotation = (GenerateRotationCode() == kIdentity) ? false : true;
    identity = !has_translation && !has_rotation;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SetRotation(const Type phi, const Type theta,
                   const Type psi) {

    const Type sinphi = sin(kDegToRad*phi);
    const Type cosphi = cos(kDegToRad*phi);
    const Type sinthe = sin(kDegToRad*theta);
    const Type costhe = cos(kDegToRad*theta);
    const Type sinpsi = sin(kDegToRad*psi);
    const Type cospsi = cos(kDegToRad*psi);

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
  VECGEOM_INLINE
  void SetRotation(Vector3D<Type> const &vec) {
    SetRotation(vec[0], vec[1], vec[2]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SetRotation(const Type rot0, const Type rot1, const Type rot2,
                   const Type rot3, const Type rot4, const Type rot5,
                   const Type rot6, const Type rot7, const Type rot8) {

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

  /**
   * Generates a bit-sequence of whether entries in the rotation matrix are
   * non-zero. Each entry uses two bits which is 00 for 0, 01 for positive
   * values and 11 for negative values. 10 has no meaning.
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  RotationCode GenerateRotationCode() const {
    int code = 0;
    for (int i = 0; i < 9; ++i) {
      // Assign each set of two bits
      code |= (1<<i) * (rot[i] != 0);
    }
    if (code == kDiagonal && (rot[0] == 1. && rot[4] == 1. && rot[8] == 1.)) {
      code = kIdentity;
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
  VECGEOM_INLINE
  TranslationCode GenerateTranslationCode() const {
    return static_cast<int>(has_translation);
  }

private:

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
  void DoRotation(Vector3D<InputType> const &master,
                  Vector3D<InputType> *const local) const {

    if (code == 0x1B1) {
      local[0] = master[0]*rot[0];
      local[1] = master[1]*rot[4] + master[2]*rot[7];
      local[2] = master[1]*rot[5] + master[2]*rot[8];
      return;
    }
    if (code == 0x18E) {
      local[0] = master[1]*rot[3];
      local[1] = master[0]*rot[1] + master[2]*rot[7];
      local[2] = master[0]*rot[2] + master[2]*rot[8];
      return;
    }
    if (code == 0x076){
      local[0] = master[2]*rot[6];
      local[1] = master[0]*rot[1] + master[1]*rot[4];
      local[2] = master[0]*rot[2] + master[1]*rot[5];
      return;
    }
    if (code == 0x16A) {
      local[0] = master[1]*rot[3] + master[2]*rot[6];
      local[1] = master[0]*rot[1];
      local[2] = master[2]*rot[5] + master[2]*rot[8];
      return;
    }
    if (code == 0x155) {
      local[0] = master[0]*rot[0] + master[2]*rot[6];
      local[1] = master[1]*rot[4];
      local[2] = master[0]*rot[2] + master[2]*rot[8];
      return;
    }
    if (code == 0x0AD){
      local[0] = master[0]*rot[0] + master[1]*rot[3];
      local[1] = master[2]*rot[7];
      local[2] = master[0]*rot[2] + master[1]*rot[5];
      return;
    }
    if (code == 0x0DC){
      local[0] = master[1]*rot[3] + master[2]*rot[6];
      local[1] = master[1]*rot[4] + master[2]*rot[7];
      local[2] = master[0]*rot[2];
      return;
    }
    if (code == 0x0E3) {
      local[0] = master[0]*rot[0] + master[2]*rot[6];
      local[1] = master[0]*rot[1] + master[2]*rot[7];
      local[2] = master[1]*rot[5];
      return;
    }
    if (code == 0x11B){
      local[0] = master[0]*rot[0] + master[1]*rot[3];
      local[1] = master[0]*rot[1] + master[1]*rot[4];
      local[2] = master[2]*rot[8];
      return;
    }
    if (code == 0x0A1){
      local[0] = master[0]*rot[0];
      local[1] = master[2]*rot[7];
      local[2] = master[1]*rot[5];
      return;
    }
    if (code == 0x10A){
      local[0] = master[1]*rot[3];
      local[1] = master[0]*rot[1];
      local[2] = master[2]*rot[8];
      return;
    }
    if (code == 0x046){
      local[0] = master[1]*rot[3];
      local[1] = master[2]*rot[7];
      local[2] = master[0]*rot[2];
      return;
    }
    if (code == 0x062) {
      local[0] = master[2]*rot[6];
      local[1] = master[0]*rot[1];
      local[2] = master[1]*rot[5];
      return;
    }
    if (code == 0x054) {
      local[0] = master[2]*rot[6];
      local[1] = master[1]*rot[4];
      local[2] = master[0]*rot[2];
      return;
    }

    // code = 0x111;
    if (code == kDiagonal) {
      local[0] = master[0]*rot[0];
      local[1] = master[1]*rot[4];
      local[2] = master[2]*rot[8];
      return;
    }

    // code = 0x200;
    if (code == kIdentity){
      local = master;
      return;
    }

    // General case
    local[0] =  master[0]*rot[0];
    local[1] =  master[0]*rot[1];
    local[2] =  master[0]*rot[2];
    local[0] += master[1]*rot[3];
    local[1] += master[1]*rot[4];
    local[2] += master[1]*rot[5];
    local[0] += master[2]*rot[6];
    local[1] += master[2]*rot[7];
    local[2] += master[2]*rot[8];

  }

  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void DoTranslation(Vector3D<InputType> const &master,
                     Vector3D<InputType> *const local) const {
    (*local)[0] = master[0] - trans[0];
    (*local)[1] = master[1] - trans[1];
    (*local)[2] = master[2] - trans[2];
  }

public:

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
  void Transform(Vector3D<InputType> const &master,
                 Vector3D<InputType> *const local) const {

    // Identity
    if (trans_code == 0 && rot_code == kIdentity) {
      *local = master;
      return;
    }

    // Only translation
    if (trans_code == 1 && rot_code == kIdentity) {
      DoTranslation(master, local);
      return;
    }

    // Only rotation
    if (trans_code == 0 && rot_code != kIdentity) {
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
  Vector3D<InputType> Transform(Vector3D<InputType> const &master) const {
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
  void TransformRotation(Vector3D<InputType> const &master,
                         Vector3D<InputType> *const local) const {

    // Rotational identity
    if (code == kIdentity) {
      *local = master;
      return;
    }

    // General case
    DoRotation(master, local);

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
  Vector3D<InputType> TransformRotation(
      Vector3D<InputType> const &master) const {
    Vector3D<InputType> local;
    TransformRotation<code>(master, &local);
    return local;
  }

}; // End class TransformationMatrix

template <TranslationCode trans_code, RotationCode rot_code, typename Type>
class SpecializedMatrix : public TransformationMatrix<Type> {

  typedef TransformationMatrix<Type> GeneralMatrix;

  /**
   * \sa TransformationMatrix::Transform(Vector3D<InputType> const &,
   *                                     Vector3D<InputType> *const)
   */
  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Transform(Vector3D<InputType> const &master,
                 Vector3D<InputType> *const local) const {
    GeneralMatrix::template Transform<trans_code, rot_code, InputType>(
      master, local
    );
  }

  /**
   * \sa TransformationMatrix::Transform(Vector3D<InputType> const &)
   */
  template <TranslationCode, RotationCode code, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<InputType> Transform(Vector3D<InputType> const &master) const {
    return GeneralMatrix::template Transform<trans_code, rot_code,
                                             InputType>(master);
  }

  /**
   * \sa TransformationMatrix::TransformRotation(Vector3D<InputType> const &,
   *                                             Vector3D<InputType> *const)
   */
  template <RotationCode code, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void TransformRotation(Vector3D<InputType> const &master,
                         Vector3D<InputType> *const local) const {
    GeneralMatrix::template TransformRotation<code, InputType>(
      master, local
    );
  }

  /**
   * \sa TransformationMatrix::TransformRotation(Vector3D<InputType> const &)
   */
  template <RotationCode code, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<InputType> TransformRotation(
      Vector3D<InputType> const &master) const {
    return GeneralMatrix::template TransformRotation<code, InputType>(
             master
           );
  }

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_TRANSMATRIX_H_