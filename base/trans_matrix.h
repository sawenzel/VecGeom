#ifndef VECGEOM_BASE_TRANSMATRIX_H_
#define VECGEOM_BASE_TRANSMATRIX_H_

#include <cmath>
#include "base/types.h"
#include "base/vector3d.h"

namespace vecgeom {

template <typename Type>
class TransMatrix {

private:

  Type trans[3];
  Type rot[9];
  bool identity;
  bool has_rotation;
  bool has_translation;

public:

  VECGEOM_CUDA_HEADER_BOTH
  TransMatrix() {
    SetTranslation(0, 0, 0);
    SetRotation(0, 0, 0);
  }

  VECGEOM_CUDA_HEADER_BOTH
  TransMatrix(const Type tx, const Type ty, const Type tz,
              const Type phi, const Type theta,
              const Type psi) {
    SetTranslation(tx, ty, tz);
    SetRotation(phi, theta, psi);
  }

  VECGEOM_CUDA_HEADER_BOTH
  TransMatrix(TransMatrix const &other) {
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
    has_rotation = (RotationFootprint(rot) == 1296) ? false : true;
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

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SetRotation(const Type *rot_) {
    SetRotation(rot_[0], rot_[1], rot_[2], rot_[3], rot_[4], rot_[5],
                rot_[6], rot_[7], rot_[8]);
  }

  /**
   * Computes the "rotation footprint" of a set of rotation entries, which is a
   * concept employed by Sandro to emit a unique id that identifies the
   * necessary steps for transformation.
   * \param rot C-style array with 9 entries, containing the rotational part of
   *            the transformation matrix.
   * \return Unique id to identify different types of rotation, determining the
   *         necessary steps to compute transformations.
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static int RotationFootprint(Type const *rot) {

    int footprint = 0;

    // Count zero-entries and give back a footprint that classifies them
    for (int i = 0; i < 9; ++i) {
      if (std::fabs(rot[i]) < 1e-12) {
        footprint += i*i*i; // Cubic power identifies cases uniquely
      }
    }

    // Diagonal matrix. Check if this is the trivial case.
    if (footprint == 720) {
      if (rot[0] == 1. && rot[4] == 1. && rot[8] == 1.) {
        // Trivial rotation (none)
        return 1296;
      }
    }

    return footprint;
  }

  /**
   * Computes the rotation footprint for this matrix.
   * \sa TransMatrix::RotationFootprint(Type const *rot)
   * \return Unique id to identify different types of rotation, determining the
   *         necessary steps to compute transformations.
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int RotationFootprint() const {
    return RotationFootprint(rot);
  }

private:

  /**
   * General case rotation transformation. Implemented separately to allow
   * inclusion in both regular and rotation-only transforms.
   */
  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void DoRotation(Vector3D<InputType> const &master,
                  Vector3D<InputType> *const local) const {
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

public:

  /**
   * Transform a point in to the local reference frame. Currently only few
   * checks are performed; should potentially have more cases to simplify
   * the transformation.
   * \param master Point to be transformed.
   * \param local Output destination. Should never be the same as the input
   *              vector!
   */
  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Transform(Vector3D<InputType> const &master,
                 Vector3D<InputType> *const local) const {

    // Vc can do early returns here as they are only dependent on the matrix
    if (IsIdentity()) {
      *local = master;
      return;
    }

    *local = master - Translation();

    if (!HasRotation()) return;

    // General case
    DoRotation(master, local);

  }

  /**
   * Since transformation cannot be done in place, allows the transformed vector
   * to be constructed by Transform directly.
   * \param master Point to be transformed.
   * \return Newly constructed Vector3D with the transformed coordinates.
   */
  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<InputType> Transform(Vector3D<InputType> const &master) const {
    Vector3D<InputType> local;
    Transform(master, &local);
    return local;
  }

  /**
   * Only transforms by rotation, ignoring the translation part. This is useful
   * when transforming directions.
   * \param master Point to be transformed.
   * \param local Output destination of transformation.
   */
  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void TransformRotation(Vector3D<InputType> const &master,
                         Vector3D<InputType> *const local) const {

    // Vector backends can do early returns here as this is only dependent on
    // the matrix
    if (!HasRotation()) {
      *local = master;
      return;
    }

    DoRotation(master, local);

  }

  /**
   * Since transformation cannot be done in place, allows the transformed vector
   * to be constructed by TransformRotation directly.
   * \param master Point to be transformed.
   * \return Newly constructed Vector3D with the transformed coordinates.
   */
  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<InputType> TransformRotation(
      Vector3D<InputType> const &master) const {
    Vector3D<InputType> local;
    TransformRotation(master, &local);
    return local;
  }

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_TRANSMATRIX_H_