/**
 * @file specialized_matrix.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_SPECIALIZEDMATRIX_H_
#define VECGEOM_BASE_SPECIALIZEDMATRIX_H_

#include "base/transformation_matrix.h"

namespace VECGEOM_NAMESPACE {

/**
 * @brief Specializes on the necessary translation on rotation, eliminating
 *        unecessary computations.
 */
template <TranslationCode trans_code, RotationCode rot_code>
class SpecializedMatrix : public TransformationMatrix {

public:

  virtual int memory_size() const { return sizeof(*this); }

  /**
   * \sa TransformationMatrix::Transform(Vector3D<InputType> const &,
   *                                     Vector3D<InputType> *const)
   */
  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Transform(Vector3D<InputType> const &master,
                 Vector3D<InputType> *const local) const {
    this->Transform<trans_code, rot_code, InputType>(master, local);
  }

  /**
   * \sa TransformationMatrix::Transform(Vector3D<InputType> const &)
   */
  template <TranslationCode, RotationCode code, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<InputType> Transform(Vector3D<InputType> const &master) const {
    return this->Transform<trans_code, rot_code, InputType>(master);
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
    this->TransformRotation<code, InputType>(master, local);
  }

  /**
   * \sa TransformationMatrix::TransformRotation(Vector3D<InputType> const &)
   */
  template <RotationCode code, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<InputType> TransformRotation(
      Vector3D<InputType> const &master) const {
    return this->TransformRotation<code, InputType>(master);
  }

};

} // End global namespace

#endif // VECGEOM_BASE_SPECIALIZEDMATRIX_H_