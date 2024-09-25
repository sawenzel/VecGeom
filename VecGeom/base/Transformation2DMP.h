/// \file Transformation2DMP.h This provides the precision templated Transformation2D

#ifndef VECGEOM_BASE_TRANSFORMATION2DMP_H_
#define VECGEOM_BASE_TRANSFORMATION2DMP_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"

#include <VecGeom/base/Transformation3D.h>

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/Vector2D.h"
#include "VecGeom/backend/scalar/Backend.h"
#ifdef VECGEOM_ENABLE_CUDA
#include "VecGeom/backend/cuda/Interface.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include <initializer_list>
// #ifdef VECGEOM_ROOT
// class TGeoMatrix;
// #endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class Transformation2DMP;);

inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA
}
namespace cuda {
class Transformation2DMP;
}
inline namespace VECGEOM_IMPL_NAMESPACE {
// class vecgeom::cuda::Transformation2D;
#endif

template <typename Real_s>
class Transformation2DMP {

  // On the precision templating: Real_s is the precision used for the stored variables,
  // Real_i the one for the input variables.

private:
  // TODO: it might be better to directly store this in terms of Vector3D<Real_t> !!
  // and would allow for higher level abstraction
  Real_s tx_{0.}, ty_{0.}, tz_{0.};
  Real_s rxx_{1.}, ryx_{0.};
  Real_s rxy_{0.}, ryy_{1.};
  bool fIdentity;
  bool fHasRotation;
  bool fHasTranslation;

public:
  VECCORE_ATT_HOST_DEVICE
  constexpr Transformation2DMP() : fIdentity(true), fHasRotation(false), fHasTranslation(false){};

  /**
   * Constructor from other transformation in other precision.
   * @tparam Real_i precision of other transformation
   * @param other other transformation
   */
  template <typename Real_i>
  Transformation2DMP(const Transformation2DMP<Real_i> &other)
  {
    // Assuming getTranslation returns a pointer or array of 3 elements
    auto translation = other.Translation();
    tx_              = static_cast<Real_s>(translation[0]);
    ty_              = static_cast<Real_s>(translation[1]);
    tz_              = static_cast<Real_s>(translation[2]);

    // Assuming getRotation returns a pointer or array of 9 elements
    auto rotation = other.Rotation();
    rxx_          = static_cast<Real_s>(rotation[0]);
    ryx_          = static_cast<Real_s>(rotation[1]);
    rxy_          = static_cast<Real_s>(rotation[2]);
    ryy_          = static_cast<Real_s>(rotation[3]);

    fIdentity       = other.IsIdentity();
    fHasRotation    = other.HasRotation();
    fHasTranslation = other.HasTranslation();
  }

  /**
   * Constructor from 3D transformation in other precision.
   * @tparam Real_i precision of other transformation
   * @param other other transformation
   */
  template <typename Real_i>
  Transformation2DMP(const Transformation3DMP<Real_i> &other)
  {
    // Assuming getTranslation returns a pointer or array of 3 elements
    auto translation = other.Translation();
    tx_              = static_cast<Real_s>(translation[0]);
    ty_              = static_cast<Real_s>(translation[1]);
    tz_              = static_cast<Real_s>(translation[2]);

    // Assuming getRotation returns a pointer or array of 9 elements
    auto rotation = other.Rotation();

    // 2D transformation should only be created for 2D rotations, so rzx, rxz, rzy, ryz should be 0 and ryy should be
    // +-1
    assert(Abs(rotation[2]) < vecgeom::kTolerance && Abs(rotation[5]) < vecgeom::kTolerance &&
           Abs(rotation[6]) < vecgeom::kTolerance && Abs(rotation[7]) < vecgeom::kTolerance &&
           Abs(rotation[8]) - 1. < vecgeom::kTolerance);

    rxx_ = static_cast<Real_s>(rotation[0]);
    ryx_ = static_cast<Real_s>(rotation[1]);
    rxy_ = static_cast<Real_s>(rotation[3]);
    ryy_ = static_cast<Real_s>(rotation[4]);

    fIdentity       = other.IsIdentity();
    fHasRotation    = other.HasRotation();
    fHasTranslation = other.HasTranslation();
  }

  /**
   * Constructor for translation only.
   * @param tx Translation in x-coordinate.
   * @param ty Translation in y-coordinate.
   * @param tz Translation in z-coordinate.
   */
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE Transformation2DMP(const Real_i tx, const Real_i ty, const Real_i tz)
      : tx_{static_cast<Real_s>(tx)}, ty_{static_cast<Real_s>(ty)}, tz_{static_cast<Real_s>(tz)},
        fIdentity(tx == 0 && ty == 0 && tz == 0), fHasRotation(false), fHasTranslation(tx != 0 || ty != 0 || tz != 0)
  {
  }

  /**
   * Constructor to manually set each entry. Used when converting from different
   * geometry.
   */
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE Transformation2DMP(const Real_i tx, const Real_i ty, const Real_i tz, const Real_i rxx,
                                             const Real_i ryx, const Real_i rxy, const Real_i ryy)
      : fIdentity(false), fHasRotation(true), fHasTranslation(true)
  {
    SetTranslation(tx, ty, tz);
    SetRotation(rxx, ryx, rxy, ryy);
    SetProperties();
  }
  /**
   * Constructor using the rotation from a different transformation
   */
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE Transformation2DMP(const Real_i tx, const Real_i ty, const Real_i tz,
                                             Transformation2DMP<Real_i> const &trot)
  {
    SetTranslation(tx, ty, tz);
    SetRotation(trot.Rotation()[0], trot.Rotation()[1], trot.Rotation()[2], trot.Rotation()[3]);
    SetProperties();
  }
  /**
   * Constructor to manually set each entry. Used when converting from different
   * geometry, supporting scaling.
   */
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE Transformation2DMP(const Real_i tx, const Real_i ty, const Real_i tz, const Real_i rxx,
                                             const Real_i ryx, const Real_i rxy, const Real_i ryy, Real_i sx, Real_i sy)
  {
    SetTranslation(tx, ty, tz);
    SetRotation(rxx, ryx, rxy, ryy);
    ApplyScale(sx, sy);
    SetProperties();
  }

  /**
   * Constructor copying the translation and rotation from memory
   * geometry.
   */
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE Transformation2DMP(const Real_i *trans, const Real_i *rot, bool has_trans, bool has_rot)
  {
    this->Set(trans, rot, has_trans, has_rot);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool operator==(Transformation2DMP const &rhs) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Transformation2DMP operator*(Transformation2DMP const &tf) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Transformation2DMP const &operator*=(Transformation2DMP const &rhs);

  VECCORE_ATT_HOST_DEVICE
  ~Transformation2DMP() {}

  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE void ApplyScale(const Real_i sx, const Real_i sy)
  {
    rxx_ *= static_cast<Real_s>(sx);
    ryx_ *= static_cast<Real_s>(sy);
    rxy_ *= static_cast<Real_s>(sx);
    ryy_ *= static_cast<Real_s>(sy);
  }

  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE bool ApproxEqual(Transformation2DMP<Real_i> const &rhs,
                                           Real_i tolerance = kToleranceDist<Real_i>) const
  {
    auto const data1 = &tx_;
    auto const data2 = &rhs.tx_;
    for (int i = 0; i < 7; ++i)
      if (vecCore::math::Abs(data1[i] - data2[i]) > tolerance) return false;
    return true;
  }

  VECCORE_ATT_HOST_DEVICE
  void Clear()
  {
    tx_             = 0.;
    ty_             = 0.;
    tz_             = 0.;
    rxx_            = 1.;
    ryx_            = 0.;
    rxy_            = 0.;
    ryy_            = 1.;
    fIdentity       = true;
    fHasRotation    = false;
    fHasTranslation = false;
  }

  int MemorySize() const { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  void FixZeroes()
  {
    if (std::abs(tx_) < vecgeom::kTolerance) tx_ = 0.;
    if (std::abs(ty_) < vecgeom::kTolerance) ty_ = 0.;
    if (std::abs(tz_) < vecgeom::kTolerance) tz_ = 0.;
    if (std::abs(rxx_) < vecgeom::kTolerance) rxx_ = 0.;
    if (std::abs(ryx_) < vecgeom::kTolerance) ryx_ = 0.;
    if (std::abs(rxy_) < vecgeom::kTolerance) rxy_ = 0.;
    if (std::abs(ryy_) < vecgeom::kTolerance) ryy_ = 0.;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Real_s> Translation() const { return Vector3D<Real_s>(tx_, ty_, tz_); }

  /**
   * No safety against faulty indexing.
   * @param index Index of translation entry in the range [0-2].
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Real_s Translation(const int index) const { return *(&tx_ + index); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Real_s const *Rotation() const { return &rxx_; }

  /**
   * No safety against faulty indexing.
   * \param index Index of rotation entry in the range [0-3].
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Real_s Rotation(const int index) const { return *(&rxx_ + index); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsIdentity() const { return fIdentity; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsXYRotation() const { return (fHasRotation); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool HasRotation() const { return fHasRotation; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool HasTranslation() const { return fHasTranslation; }

  void Print(std::ostream &s) const
  {
    s << "Transformation2DMP {{" << tx_ << "," << ty_ << "," << tz_ << "}";
    s << "{" << rxx_ << "," << ryx_ << "," << rxy_ << "," << ryy_ << "}}\n";
  }

  VECCORE_ATT_HOST_DEVICE
  void Print() const
  {
    printf("Transformation2D {{%.2f, %.2f, %.2f}, ", tx_, ty_, tz_);
    printf("{%.2f, %.2f, %.2f, %.2f}}", rxx_, ryx_, rxy_, ryy_);
  }

  // Mutators

  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE void SetTranslation(const Real_i tx, const Real_i ty, const Real_i tz)
  {
    tx_ = static_cast<Real_s>(tx);
    ty_ = static_cast<Real_s>(ty);
    tz_ = static_cast<Real_s>(tz);
  }

  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE void SetTranslation(Vector3D<Real_i> const &vec)
  {
    SetTranslation(vec[0], vec[1], vec[2]);
  }

  // FIXME tolerance still compile option
  VECCORE_ATT_HOST_DEVICE
  void SetProperties()
  {
    fHasTranslation = (fabs(tx_) > kTolerance || fabs(ty_) > kTolerance || fabs(tz_) > kTolerance) ? true : false;
    fHasRotation    = (fabs(rxx_ - 1.) > kTolerance) || (fabs(ryx_) > kTolerance) || (fabs(rxy_) > kTolerance) ||
                           (fabs(ryy_ - 1.) > kTolerance)
                          ? true
                          : false;
    fIdentity       = !fHasTranslation && !fHasRotation;
  }

  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE void SetRotation(const Real_i xx, const Real_i yx, const Real_i xy, const Real_i yy)
  {
    rxx_ = static_cast<Real_s>(xx);
    ryx_ = static_cast<Real_s>(yx);
    rxy_ = static_cast<Real_s>(xy);
    ryy_ = static_cast<Real_s>(yy);
  }

  // /**
  //  * Set rotation given an arbitrary axis and an angle.
  //  * \param aaxis Rotation axis. No need to be a unit vector.
  //  * \param ddelta Rotation angle in radians.
  //  */
  // VECCORE_ATT_HOST_DEVICE
  // void Set(Vector3D<double> const &aaxis, double ddelta);

  /**
   * Set transformation and rotation.
   * \param trans Pointer to at least 3 values.
   * \param rot Pointer to at least 4 values.
   */
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE void Set(const Real_i *trans, const Real_i *rot, bool has_trans,
                                                        bool has_rot)
  {
    if (has_trans) {
      tx_ = static_cast<Real_s>(trans[0]);
      ty_ = static_cast<Real_s>(trans[1]);
      tz_ = static_cast<Real_s>(trans[2]);
    }

    if (has_rot) {
      rxx_ = static_cast<Real_s>(rot[0]);
      ryx_ = static_cast<Real_s>(rot[1]);
      rxy_ = static_cast<Real_s>(rot[2]);
      ryy_ = static_cast<Real_s>(rot[3]);
    }

    fHasTranslation = has_trans;
    fHasRotation    = has_rot;
    fIdentity       = !fHasTranslation && !fHasRotation;
  }

private:
  // Templated rotation and translation methods which inline and compile to
  // optimized versions.

  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void DoRotation(Vector3D<Real_s> const &master,
                                                               Vector3D<Real_s> &local) const;

private:
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void DoTranslation(Vector3D<Real_s> const &master,
                                                                  Vector3D<Real_s> &local) const;

  template <bool vectortransform>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void InverseTransformKernel(Vector3D<Real_s> const &local,
                                                                           Vector3D<Real_s> &master) const;

public:
  // Transformation interface

  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transform(Vector3D<Real_s> const &master,
                                                              Vector3D<Real_s> &local) const;

  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_s> Transform(Vector3D<Real_s> const &master) const;

  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void TransformDirection(Vector3D<Real_s> const &master,
                                                                       Vector3D<Real_s> &local) const;

  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_s> TransformDirection(
      Vector3D<Real_s> const &master) const;

  /** The inverse transformation ( aka LocalToMaster ) of an object transform like a point
   *  this does not need to currently template on placement since such a transformation is much less used
   */
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void InverseTransform(Vector3D<Real_s> const &local,
                                                                     Vector3D<Real_s> &master) const;

  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_s> InverseTransform(Vector3D<Real_s> const &local) const;

  /** The inverse transformation of an object transforming like a vector */
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void InverseTransformDirection(Vector3D<Real_s> const &master,
                                                                              Vector3D<Real_s> &local) const;

  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_s> InverseTransformDirection(
      Vector3D<Real_s> const &master) const;

  /** compose transformations - multiply transformations */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void MultiplyFromRight(Transformation2DMP const &rhs);

  /**
   * @brief Returns the inverse of this matrix. Taken from G4AffineTransformation
   */
  VECCORE_ATT_HOST_DEVICE
  Transformation2DMP<Real_s> Inverse() const
  {
    Real_s ttx = -tx_, tty = -ty_, ttz = -tz_;
    return Transformation2DMP<Real_s>(ttx * rxx_ + tty * rxy_, ttx * ryx_ + tty * ryy_, ttz, rxx_, rxy_, ryx_, ryy_);
  }

  // Utility and CUDA

#ifdef VECGEOM_CUDA_INTERFACE
  size_t DeviceSizeOf() const { return DevicePtr<cuda::Transformation2DMP>::SizeOf(); }
  DevicePtr<cuda::Transformation2DMP> CopyToGpu() const;
  DevicePtr<cuda::Transformation2DMP> CopyToGpu(DevicePtr<cuda::Transformation2DMP> const gpu_ptr) const;
  static void CopyManyToGpu(const std::vector<Transformation2DMP const *> &trafos,
                            const std::vector<DevicePtr<cuda::Transformation2DMP>> &gpu_ptrs);
#endif

public:
  static const Transformation2DMP kIdentity;

}; // End class Transformation2D

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool Transformation2DMP<Real_t>::operator==(Transformation2DMP<Real_t> const &rhs) const
{
  return equal(&tx_, &tx_ + 7, &rhs.tx_);
}

/**
 * Rotates a vector to this transformation's frame of reference.
 * \param master Vector in original frame of reference.
 * \param local Output vector rotated to the new frame of reference.
 */
template <typename Real_s>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation2DMP<Real_s>::DoRotation(Vector3D<Real_s> const &master,
                                                                                         Vector3D<Real_s> &local) const
{
  local[0] = master[0] * rxx_;
  local[1] = master[0] * ryx_;
  local[2] = master[2]; // this must be set since the non-existing rzz_ = 1.
  local[0] += master[1] * rxy_;
  local[1] += master[1] * ryy_;
}

template <typename Real_s>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation2DMP<Real_s>::DoTranslation(
    Vector3D<Real_s> const &master, Vector3D<Real_s> &local) const
{

  local[0] = master[0] - tx_;
  local[1] = master[1] - ty_;
  local[2] = master[2] - tz_;
}

/**
 * Transform a point to the local reference frame.
 * \param master Point to be transformed.
 * \param local Output destination. Should never be the same as the input
 *              vector!
 */
template <typename Real_s>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation2DMP<Real_s>::Transform(Vector3D<Real_s> const &master,
                                                                                        Vector3D<Real_s> &local) const
{
  Vector3D<Real_s> tmp;
  DoTranslation(master, tmp);
  DoRotation(tmp, local);
}

/**
 * Since transformation cannot be done in place, allows the transformed vector
 * to be constructed by Transform directly.
 * \param master Point to be transformed.
 * \return Newly constructed Vector3D with the transformed coordinates.
 */
template <typename Real_s>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_s> Transformation2DMP<Real_s>::Transform(
    Vector3D<Real_s> const &master) const
{

  Vector3D<Real_s> local;
  Transform(master, local);
  return local;
}

template <typename Real_s>
template <bool transform_direction>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation2DMP<Real_s>::InverseTransformKernel(
    Vector3D<Real_s> const &local, Vector3D<Real_s> &master) const
{

  // we are just doing the full stuff here ( LocalToMaster is less critical
  // than other way round )

  if (transform_direction) {
    master[0] = local[0] * rxx_;
    master[0] += local[1] * ryx_;
    master[1] = local[0] * rxy_;
    master[1] += local[1] * ryy_;
    master[2] = local[2]; // assuming rzz_ = 1

  } else {
    master[0] = tx_;
    master[0] += local[0] * rxx_;
    master[0] += local[1] * ryx_;
    master[1] = ty_;
    master[1] += local[0] * rxy_;
    master[1] += local[1] * ryy_;
    master[2] = tz_;
    master[2] += local[2];
  }
}

template <typename Real_s>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation2DMP<Real_s>::InverseTransform(
    Vector3D<Real_s> const &local, Vector3D<Real_s> &master) const
{
  InverseTransformKernel<false>(local, master);
}

template <typename Real_s>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_s> Transformation2DMP<Real_s>::InverseTransform(
    Vector3D<Real_s> const &local) const
{
  Vector3D<Real_s> tmp;
  InverseTransform(local, tmp);
  return tmp;
}

template <typename Real_s>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation2DMP<Real_s>::InverseTransformDirection(
    Vector3D<Real_s> const &local, Vector3D<Real_s> &master) const
{
  InverseTransformKernel<true>(local, master);
}

template <typename Real_s>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_s> Transformation2DMP<Real_s>::InverseTransformDirection(
    Vector3D<Real_s> const &local) const
{
  Vector3D<Real_s> tmp;
  InverseTransformDirection(local, tmp);
  return tmp;
}

template <typename Real_s>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Transformation2DMP<Real_s> Transformation2DMP<Real_s>::operator*(
    Transformation2DMP<Real_s> const &rhs) const
{
  if (rhs.fIdentity) return Transformation2DMP<Real_s>(*this);
  return Transformation2DMP<Real_s>(tx_ * rhs.rxx_ + ty_ * rhs.ryx_ + tz_, // tx
                                    tx_ * rhs.rxy_ + ty_ * rhs.ryy_ + tz_, // ty
                                    tz_ + rhs.tz_,                         // tz
                                    rxx_ * rhs.rxx_ + rxy_ * rhs.ryx_,     // rxx
                                    ryx_ * rhs.rxx_ + ryy_ * rhs.ryx_,     // ryx
                                    rxx_ * rhs.rxy_ + rxy_ * rhs.ryy_,     // rxy
                                    ryx_ * rhs.rxy_ + ryy_ * rhs.ryy_);    // ryy
}

template <typename Real_s>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE void Transformation2DMP<Real_s>::MultiplyFromRight(
    Transformation2DMP<Real_s> const &rhs)
{
  // TODO: this code should directly operator on Vector3D and Matrix3D

  if (rhs.fIdentity) return;
  fIdentity = false;

  if (rhs.HasTranslation()) {
    fHasTranslation = true;
    // ideal for fused multiply add
    tx_ += rxx_ * rhs.tx_;
    tx_ += ryx_ * rhs.ty_;

    ty_ += rxy_ * rhs.tx_;
    ty_ += ryy_ * rhs.ty_;

    tz_ += rhs.tz_;
  }

  if (rhs.HasRotation()) {
    fHasRotation   = true;
    Precision tmpx = rxx_;
    Precision tmpy = ryx_;

    // first row of matrix
    rxx_ = tmpx * rhs.rxx_;
    ryx_ = tmpx * rhs.ryx_;
    rxx_ += tmpy * rhs.rxy_;
    ryx_ += tmpy * rhs.ryy_;

    tmpx = rxy_;
    tmpy = ryy_;

    // second row of matrix
    rxy_ = tmpx * rhs.rxx_;
    ryy_ = tmpx * rhs.ryx_;
    rxy_ += tmpy * rhs.rxy_;
    ryy_ += tmpy * rhs.ryy_;
  }
}

/**
 * Only transforms by rotation, ignoring the translation part. This is useful
 * when transforming directions.
 * \param master Point to be transformed.
 * \param local Output destination of transformation.
 */
template <typename Real_s>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation2DMP<Real_s>::TransformDirection(
    Vector3D<Real_s> const &master, Vector3D<Real_s> &local) const
{
  DoRotation(master, local);
}

/**
 * Since transformation cannot be done in place, allows the transformed vector
 * to be constructed by TransformDirection directly.
 * \param master Point to be transformed.
 * \return Newly constructed Vector3D with the transformed coordinates.
 */
template <typename Real_s>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_s> Transformation2DMP<Real_s>::TransformDirection(
    Vector3D<Real_s> const &master) const
{

  Vector3D<Real_s> local;
  TransformDirection(master, local);
  return local;
}

template <typename Real_s>
std::ostream &operator<<(std::ostream &os, Transformation2DMP<Real_s> const &trans)
{
  os << "TransformationMP {" << trans.Translation() << ", "
     << "(" << trans.Rotation(0) << ", " << trans.Rotation(1) << ", " << trans.Rotation(2) << ", " << trans.Rotation(3)
     << ")}"
     << "; identity(" << trans.IsIdentity() << "); rotation(" << trans.HasRotation() << ")";
  return os;
}
}
} // namespace vecgeom

#endif // VECGEOM_BASE_TRANSFORMATION2DMP_H_
