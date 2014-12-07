/**
 * @file specialized_transformation3d.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_SPECIALIZEDTRANSFORMATION3D_H_
#define VECGEOM_BASE_SPECIALIZEDTRANSFORMATION3D_H_

#include "base/Transformation3D.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Specializes on the necessary translation on rotation, eliminating
 *        unecessary computations.
 */
template <TranslationCode trans_code, RotationCode rot_code>
class SpecializedTransformation3D : public Transformation3D {

public:

  virtual int memory_size() const { return sizeof(*this); }

  /**
   * \sa Transformation3D::Transform(Vector3D<InputType> const &,
   *                                 Vector3D<InputType> *const)
   */
  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Transform(Vector3D<InputType> const &master,
                 Vector3D<InputType> *const local) const {
    this->Transform<trans_code, rot_code, InputType>(master, local);
  }

  /**
   * \sa Transformation3D::Transform(Vector3D<InputType> const &)
   */
  template <TranslationCode, RotationCode code, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<InputType> Transform(Vector3D<InputType> const &master) const {
    return this->Transform<trans_code, rot_code, InputType>(master);
  }

  /**
   * \sa Transformation3D::TransformDirection(Vector3D<InputType> const &,
   *                                             Vector3D<InputType> *const)
   */
  template <RotationCode code, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void TransformDirection(Vector3D<InputType> const &master,
                          Vector3D<InputType> *const local) const {
    this->TransformDirection<code, InputType>(master, local);
  }

  /**
   * \sa Transformation3D::TransformDirection(Vector3D<InputType> const &)
   */
  template <RotationCode code, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<InputType> TransformDirection(
      Vector3D<InputType> const &master) const {
    return this->TransformDirection<code, InputType>(master);
  }

};

} } // End global namespace

#endif // VECGEOM_BASE_SPECIALIZEDTRANSFORMATION3D_H_
