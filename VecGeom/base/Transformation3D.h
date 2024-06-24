/// \file Transformation3D.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_TRANSFORMATION3D_H_
#define VECGEOM_BASE_TRANSFORMATION3D_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/backend/scalar/Backend.h"
#ifdef VECGEOM_ENABLE_CUDA
#include "VecGeom/backend/cuda/Interface.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef VECGEOM_ROOT
class TGeoMatrix;
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class Transformation3D;);

inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA
}
namespace cuda {
class Transformation3D;
}
inline namespace VECGEOM_IMPL_NAMESPACE {
// class vecgeom::cuda::Transformation3D;
#endif

class Transformation3D {

private:
  // TODO: it might be better to directly store this in terms of Vector3D<Precision> !!
  // and would allow for higher level abstraction
  Precision tx_{0.}, ty_{0.}, tz_{0.};
  Precision rxx_{1.}, ryx_{0.}, rzx_{0.};
  Precision rxy_{0.}, ryy_{1.}, rzy_{0.};
  Precision rxz_{0.}, ryz_{0.}, rzz_{1.};
  bool fIdentity;
  bool fHasRotation;
  bool fHasTranslation;

public:
  VECCORE_ATT_HOST_DEVICE
  constexpr Transformation3D() : fIdentity(true), fHasRotation(false), fHasTranslation(false){};

  /**
   * Constructor for translation only.
   * @param tx Translation in x-coordinate.
   * @param ty Translation in y-coordinate.
   * @param tz Translation in z-coordinate.
   */
  VECCORE_ATT_HOST_DEVICE
  Transformation3D(const Precision tx, const Precision ty, const Precision tz)
      : tx_{tx}, ty_{ty}, tz_{tz}, fIdentity(tx == 0 && ty == 0 && tz == 0), fHasRotation(false),
        fHasTranslation(tx != 0 || ty != 0 || tz != 0)
  {
  }

  /**
   * @brief Rotation followed by translation.
   * @param tx Translation in x-coordinate.
   * @param ty Translation in y-coordinate.
   * @param tz Translation in z-coordinate.
   * @param phi Rotation angle about z-axis.
   * @param theta Rotation angle about new y-axis.
   * @param psi Rotation angle about new z-axis.
   */
  VECCORE_ATT_HOST_DEVICE
  Transformation3D(const Precision tx, const Precision ty, const Precision tz, const Precision phi,
                   const Precision theta, const Precision psi);

  /**
   * @brief RScale followed by rotation followed by translation.
   * @param tx Translation in x-coordinate.
   * @param ty Translation in y-coordinate.
   * @param tz Translation in z-coordinate.
   * @param phi Rotation angle about z-axis.
   * @param theta Rotation angle about new y-axis.
   * @param psi Rotation angle about new z-axis.
   * @param sx X scaling factor
   * @param sy Y scaling factor
   * @param sz Z scaling factor
   */
  VECCORE_ATT_HOST_DEVICE
  Transformation3D(const Precision tx, const Precision ty, const Precision tz, const Precision phi,
                   const Precision theta, const Precision psi, Precision sx, Precision sy, Precision sz);

  /**
   * Constructor to manually set each entry. Used when converting from different
   * geometry.
   */
  VECCORE_ATT_HOST_DEVICE
  Transformation3D(const Precision tx, const Precision ty, const Precision tz, const Precision r0, const Precision r1,
                   const Precision r2, const Precision r3, const Precision r4, const Precision r5, const Precision r6,
                   const Precision r7, const Precision r8);

  /**
   * Constructor using the rotation from a different transformation
   */
  VECCORE_ATT_HOST_DEVICE
  Transformation3D(const Precision tx, const Precision ty, const Precision tz, Transformation3D const &trot);

  /**
   * Constructor to manually set each entry. Used when converting from different
   * geometry, supporting scaling.
   */
  VECCORE_ATT_HOST_DEVICE
  Transformation3D(const Precision tx, const Precision ty, const Precision tz, const Precision r0, const Precision r1,
                   const Precision r2, const Precision r3, const Precision r4, const Precision r5, const Precision r6,
                   const Precision r7, const Precision r8, Precision sx, Precision sy, Precision sz);

  /**
   * Constructor copying the translation and rotation from memory
   * geometry.
   */
  VECCORE_ATT_HOST_DEVICE
  Transformation3D(const Precision *trans, const Precision *rot, bool has_trans, bool has_rot)
  {
    this->Set(trans, rot, has_trans, has_rot);
  }

  /**
   * Constructor for a rotation based on a given direction
   * @param axis direction of the new z axis
   * @param inverse if true the origial axis will be rotated into (0,0,u)
                    if false a vector (0,0,u) will be rotated into the original axis
   */
  VECCORE_ATT_HOST_DEVICE
  Transformation3D(const Vector3D<Precision> &axis, bool inverse = true);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool operator==(Transformation3D const &rhs) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Transformation3D operator*(Transformation3D const &tf) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Transformation3D const &operator*=(Transformation3D const &rhs);

  VECCORE_ATT_HOST_DEVICE
  ~Transformation3D() {}

  VECCORE_ATT_HOST_DEVICE
  void ApplyScale(const Precision sx, const Precision sy, const Precision sz)
  {
    rxx_ *= sx;
    ryx_ *= sy;
    rzx_ *= sz;
    rxy_ *= sx;
    ryy_ *= sy;
    rzy_ *= sz;
    rxz_ *= sx;
    ryz_ *= sy;
    rzz_ *= sz;
  }

  VECCORE_ATT_HOST_DEVICE
  bool ApproxEqual(Transformation3D const &rhs, Precision tolerance = kTolerance) const
  {
    auto const data1 = &tx_;
    auto const data2 = &rhs.tx_;
    for (int i = 0; i < 12; ++i)
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
    rzx_            = 0.;
    rxy_            = 0.;
    ryy_            = 1.;
    rzy_            = 0.;
    rxz_            = 0.;
    ryz_            = 0.;
    rzz_            = 1.;
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
    if (std::abs(rzx_) < vecgeom::kTolerance) rzx_ = 0.;
    if (std::abs(rxy_) < vecgeom::kTolerance) rxy_ = 0.;
    if (std::abs(ryy_) < vecgeom::kTolerance) ryy_ = 0.;
    if (std::abs(rzy_) < vecgeom::kTolerance) rzy_ = 0.;
    if (std::abs(rxz_) < vecgeom::kTolerance) rxz_ = 0.;
    if (std::abs(ryz_) < vecgeom::kTolerance) ryz_ = 0.;
    if (std::abs(rzz_) < vecgeom::kTolerance) rzz_ = 0.;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> Translation() const { return Vector3D<Precision>(tx_, ty_, tz_); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> Scale() const
  {
    Precision sx = std::sqrt(rxx_ * rxx_ + rxy_ * rxy_ + rxz_ * rxz_);
    Precision sy = std::sqrt(ryx_ * ryx_ + ryy_ * ryy_ + ryz_ * ryz_);
    Precision sz = std::sqrt(rzx_ * rzx_ + rzy_ * rzy_ + rzz_ * rzz_);

    if (Determinant() < 0) sz = -sz;

    return Vector3D<Precision>(sx, sy, sz);
  }

  /**
   * No safety against faulty indexing.
   * @param index Index of translation entry in the range [0-2].
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision Translation(const int index) const { return *(&tx_ + index); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision const *Rotation() const { return &rxx_; }

  /**
   * No safety against faulty indexing.
   * \param index Index of rotation entry in the range [0-8].
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision Rotation(const int index) const { return *(&rxx_ + index); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsIdentity() const { return fIdentity; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsXYRotation() const
  {
    return (fHasRotation && (std::abs(rzx_) < vecgeom::kTolerance) && (std::abs(rzy_) < vecgeom::kTolerance) &&
            (std::abs(rxz_) < vecgeom::kTolerance) && (std::abs(ryz_) < vecgeom::kTolerance) &&
            (std::abs(rzz_ - Precision(1.)) < vecgeom::kTolerance));
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsReflected() const { return Determinant() < 0; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool HasRotation() const { return fHasRotation; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool HasTranslation() const { return fHasTranslation; }

  VECCORE_ATT_HOST_DEVICE
  void Print() const;

  // print to a stream
  void Print(std::ostream &) const;

  VECCORE_ATT_HOST_DEVICE
  void PrintG4() const;

  // Mutators

  VECCORE_ATT_HOST_DEVICE
  void SetTranslation(const Precision tx, const Precision ty, const Precision tz);

  VECCORE_ATT_HOST_DEVICE
  void SetTranslation(Vector3D<Precision> const &vec);

  VECCORE_ATT_HOST_DEVICE
  void SetProperties();

  VECCORE_ATT_HOST_DEVICE
  void SetRotation(const Precision phi, const Precision theta, const Precision psi);

  VECCORE_ATT_HOST_DEVICE
  void SetRotation(Vector3D<Precision> const &vec);

  VECCORE_ATT_HOST_DEVICE
  void SetRotation(const Precision rot0, const Precision rot1, const Precision rot2, const Precision rot3,
                   const Precision rot4, const Precision rot5, const Precision rot6, const Precision rot7,
                   const Precision rot8);

  /**
   * Set rotation given an arbitrary axis and an angle.
   * \param aaxis Rotation axis. No need to be a unit vector.
   * \param ddelta Rotation angle in radians.
   */
  VECCORE_ATT_HOST_DEVICE
  void Set(Vector3D<double> const &aaxis, double ddelta);

  /**
   * Set transformation and rotation.
   * \param trans Pointer to at least 3 values.
   * \param rot Pointer to at least 9 values.
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Set(const Precision *trans, const Precision *rot, bool has_trans, bool has_rot)
  {
    if (has_trans) {
      tx_ = trans[0];
      ty_ = trans[1];
      tz_ = trans[2];
    }

    if (has_rot) {
      rxx_ = rot[0];
      ryx_ = rot[1];
      rzx_ = rot[2];
      rxy_ = rot[3];
      ryy_ = rot[4];
      rzy_ = rot[5];
      rxz_ = rot[6];
      ryz_ = rot[7];
      rzz_ = rot[8];
    }

    fHasTranslation = has_trans;
    fHasRotation    = has_rot;
    fIdentity       = !fHasTranslation && !fHasRotation;
  }

private:
  // Templated rotation and translation methods which inline and compile to
  // optimized versions.

  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void DoRotation(Vector3D<InputType> const &master,
                                                               Vector3D<InputType> &local) const;

private:
  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void DoTranslation(Vector3D<InputType> const &master,
                                                                  Vector3D<InputType> &local) const;

  template <bool vectortransform, typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void InverseTransformKernel(Vector3D<InputType> const &local,
                                                                           Vector3D<InputType> &master) const;

public:
  // Transformation interface

  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transform(Vector3D<InputType> const &master,
                                                              Vector3D<InputType> &local) const;

  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<InputType> Transform(Vector3D<InputType> const &master) const;

  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void TransformDirection(Vector3D<InputType> const &master,
                                                                       Vector3D<InputType> &local) const;

  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<InputType> TransformDirection(
      Vector3D<InputType> const &master) const;

  /** The inverse transformation ( aka LocalToMaster ) of an object transform like a point
   *  this does not need to currently template on placement since such a transformation is much less used
   */
  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void InverseTransform(Vector3D<InputType> const &local,
                                                                     Vector3D<InputType> &master) const;

  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<InputType> InverseTransform(
      Vector3D<InputType> const &local) const;

  /** The inverse transformation of an object transforming like a vector */
  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void InverseTransformDirection(Vector3D<InputType> const &master,
                                                                              Vector3D<InputType> &local) const;

  template <typename InputType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<InputType> InverseTransformDirection(
      Vector3D<InputType> const &master) const;

  /** compose transformations - multiply transformations */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void MultiplyFromRight(Transformation3D const &rhs);

  /** compose transformations - multiply transformations */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void CopyFrom(Transformation3D const &rhs)
  {
    // not sure this compiles under CUDA
    copy(&rhs, &rhs + 1, this);
  }

  VECCORE_ATT_HOST_DEVICE
  Transformation3D &RotateX(double a);

  VECCORE_ATT_HOST_DEVICE
  Transformation3D &RotateY(double a);

  VECCORE_ATT_HOST_DEVICE
  Transformation3D &RotateZ(double a);

  /**
   * @brief Returns determinant for the rotation.
   */
  VECCORE_ATT_HOST_DEVICE
  double Determinant() const
  {
    // Computes determinant in double precision
    double xx_ = rxx_;
    double xy_ = rxy_;
    double xz_ = rxz_;
    double yx_ = ryx_;
    double yy_ = ryy_;
    double yz_ = ryz_;
    double zx_ = rzx_;
    double zy_ = rzy_;
    double zz_ = rzz_;

    double detxx = yy_ * zz_ - yz_ * zy_;
    double detxy = yx_ * zz_ - yz_ * zx_;
    double detxz = yx_ * zy_ - yy_ * zx_;
    double det   = xx_ * detxx - xy_ * detxy + xz_ * detxz;
    return det;
  }

  /**
   * @brief Returns rotation axis as a unit vector
   */
  VECCORE_ATT_HOST_DEVICE
  Vector3D<double> Axis() const;

  /**
   * @brief Rectifies the round-off making the rotqation true.
   * @details  From clhep/src/RotationC.cc
     Assuming the representation of this is close to a true Rotation,
     but may have drifted due to round-off error from many operations,
     this forms an "exact" orthonormal matrix for the rotation again.

     The first step is to average with the transposed inverse.  This
     will correct for small errors such as those occuring when decomposing
     a LorentzTransformation.  Then we take the bull by the horns and
     formally extract the axis and delta (assuming the Rotation were true)
     and re-setting the rotation according to those.
   */
  VECCORE_ATT_HOST_DEVICE
  Transformation3D &Rectify();

  /**
   * @brief Stores the inverse of this matrix into inverse. Taken from G4AffineTransformation
   */
  VECCORE_ATT_HOST_DEVICE
  Transformation3D &Invert()
  {
    double ttx = -tx_, tty = -ty_, ttz = -tz_;
    tx_ = ttx * rxx_ + tty * rxy_ + ttz * rxz_;
    ty_ = ttx * ryx_ + tty * ryy_ + ttz * ryz_;
    tz_ = ttx * rzx_ + tty * rzy_ + ttz * rzz_;

    auto tmp1 = ryx_;
    ryx_      = rxy_;
    rxy_      = tmp1;
    auto tmp2 = rzx_;
    rzx_      = rxz_;
    rxz_      = tmp2;
    auto tmp3 = rzy_;
    rzy_      = ryz_;
    ryz_      = tmp3;
    return *this;
  }

  /**
   * @brief Returns the inverse of this matrix. Taken from G4AffineTransformation
   */
  VECCORE_ATT_HOST_DEVICE
  Transformation3D Inverse() const
  {
    double ttx = -tx_, tty = -ty_, ttz = -tz_;
    return Transformation3D(ttx * rxx_ + tty * rxy_ + ttz * rxz_, ttx * ryx_ + tty * ryy_ + ttz * ryz_,
                            ttx * rzx_ + tty * rzy_ + ttz * rzz_, rxx_, rxy_, rxz_, ryx_, ryy_, ryz_, rzx_, rzy_, rzz_);
  }

  // Utility and CUDA

#ifdef VECGEOM_CUDA_INTERFACE
  size_t DeviceSizeOf() const { return DevicePtr<cuda::Transformation3D>::SizeOf(); }
  DevicePtr<cuda::Transformation3D> CopyToGpu() const;
  DevicePtr<cuda::Transformation3D> CopyToGpu(DevicePtr<cuda::Transformation3D> const gpu_ptr) const;
  static void CopyManyToGpu(const std::vector<Transformation3D const *> &trafos,
                            const std::vector<DevicePtr<cuda::Transformation3D>> &gpu_ptrs);
#endif

#ifdef VECGEOM_ROOT
  // function to convert this transformation to a TGeo transformation
  // mainly used for the benchmark comparisons with ROOT
  static TGeoMatrix *ConvertToTGeoMatrix(Transformation3D const &);
#endif

public:
  static const Transformation3D kIdentity;

}; // End class Transformation3D

VECCORE_ATT_HOST_DEVICE
bool Transformation3D::operator==(Transformation3D const &rhs) const { return equal(&tx_, &tx_ + 12, &rhs.tx_); }

/**
 * Rotates a vector to this transformation's frame of reference.
 * \param master Vector in original frame of reference.
 * \param local Output vector rotated to the new frame of reference.
 */
template <typename InputType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation3D::DoRotation(Vector3D<InputType> const &master,
                                                                               Vector3D<InputType> &local) const
{
  local[0] = master[0] * rxx_;
  local[1] = master[0] * ryx_;
  local[2] = master[0] * rzx_;
  local[0] += master[1] * rxy_;
  local[1] += master[1] * ryy_;
  local[2] += master[1] * rzy_;
  local[0] += master[2] * rxz_;
  local[1] += master[2] * ryz_;
  local[2] += master[2] * rzz_;
}

template <typename InputType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation3D::DoTranslation(Vector3D<InputType> const &master,
                                                                                  Vector3D<InputType> &local) const
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
template <typename InputType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation3D::Transform(Vector3D<InputType> const &master,
                                                                              Vector3D<InputType> &local) const
{
  Vector3D<InputType> tmp;
  DoTranslation(master, tmp);
  DoRotation(tmp, local);
}

/**
 * Since transformation cannot be done in place, allows the transformed vector
 * to be constructed by Transform directly.
 * \param master Point to be transformed.
 * \return Newly constructed Vector3D with the transformed coordinates.
 */
template <typename InputType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<InputType> Transformation3D::Transform(
    Vector3D<InputType> const &master) const
{

  Vector3D<InputType> local;
  Transform(master, local);
  return local;
}

template <bool transform_direction, typename InputType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation3D::InverseTransformKernel(
    Vector3D<InputType> const &local, Vector3D<InputType> &master) const
{

  // we are just doing the full stuff here ( LocalToMaster is less critical
  // than other way round )

  if (transform_direction) {
    master[0] = local[0] * rxx_;
    master[0] += local[1] * ryx_;
    master[0] += local[2] * rzx_;
    master[1] = local[0] * rxy_;
    master[1] += local[1] * ryy_;
    master[1] += local[2] * rzy_;
    master[2] = local[0] * rxz_;
    master[2] += local[1] * ryz_;
    master[2] += local[2] * rzz_;
  } else {
    master[0] = tx_;
    master[0] += local[0] * rxx_;
    master[0] += local[1] * ryx_;
    master[0] += local[2] * rzx_;
    master[1] = ty_;
    master[1] += local[0] * rxy_;
    master[1] += local[1] * ryy_;
    master[1] += local[2] * rzy_;
    master[2] = tz_;
    master[2] += local[0] * rxz_;
    master[2] += local[1] * ryz_;
    master[2] += local[2] * rzz_;
  }
}

template <typename InputType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation3D::InverseTransform(Vector3D<InputType> const &local,
                                                                                     Vector3D<InputType> &master) const
{
  InverseTransformKernel<false, InputType>(local, master);
}

template <typename InputType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<InputType> Transformation3D::InverseTransform(
    Vector3D<InputType> const &local) const
{
  Vector3D<InputType> tmp;
  InverseTransform(local, tmp);
  return tmp;
}

template <typename InputType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation3D::InverseTransformDirection(
    Vector3D<InputType> const &local, Vector3D<InputType> &master) const
{
  InverseTransformKernel<true, InputType>(local, master);
}

template <typename InputType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<InputType> Transformation3D::InverseTransformDirection(
    Vector3D<InputType> const &local) const
{
  Vector3D<InputType> tmp;
  InverseTransformDirection(local, tmp);
  return tmp;
}

VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
Transformation3D Transformation3D::operator*(Transformation3D const &rhs) const
{
  if (rhs.fIdentity) return Transformation3D(*this);
  return Transformation3D(tx_ * rhs.rxx_ + ty_ * rhs.ryx_ + tz_ * rhs.rzx_ + rhs.tx_, // tx
                          tx_ * rhs.rxy_ + ty_ * rhs.ryy_ + tz_ * rhs.rzy_ + rhs.ty_, // ty
                          tx_ * rhs.rxz_ + ty_ * rhs.ryz_ + tz_ * rhs.rzz_ + rhs.tz_, // tz
                          rxx_ * rhs.rxx_ + rxy_ * rhs.ryx_ + rxz_ * rhs.rzx_,        // rxx
                          ryx_ * rhs.rxx_ + ryy_ * rhs.ryx_ + ryz_ * rhs.rzx_,        // ryx
                          rzx_ * rhs.rxx_ + rzy_ * rhs.ryx_ + rzz_ * rhs.rzx_,        // rzx
                          rxx_ * rhs.rxy_ + rxy_ * rhs.ryy_ + rxz_ * rhs.rzy_,        // rxy
                          ryx_ * rhs.rxy_ + ryy_ * rhs.ryy_ + ryz_ * rhs.rzy_,        // ryy
                          rzx_ * rhs.rxy_ + rzy_ * rhs.ryy_ + rzz_ * rhs.rzy_,        // rzy
                          rxx_ * rhs.rxz_ + rxy_ * rhs.ryz_ + rxz_ * rhs.rzz_,        // rxz
                          ryx_ * rhs.rxz_ + ryy_ * rhs.ryz_ + ryz_ * rhs.rzz_,        // ryz
                          rzx_ * rhs.rxz_ + rzy_ * rhs.ryz_ + rzz_ * rhs.rzz_);       // rzz
}

VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
Transformation3D const &Transformation3D::operator*=(Transformation3D const &rhs)
{
  if (rhs.fIdentity) return *this;
  fIdentity = false;

  fHasTranslation |= rhs.HasTranslation();
  if (fHasTranslation) {
    auto tx = tx_ * rhs.rxx_ + ty_ * rhs.ryx_ + tz_ * rhs.rzx_ + rhs.tx_;
    auto ty = tx_ * rhs.rxy_ + ty_ * rhs.ryy_ + tz_ * rhs.rzy_ + rhs.ty_;
    auto tz = tx_ * rhs.rxz_ + ty_ * rhs.ryz_ + tz_ * rhs.rzz_ + rhs.tz_;
    tx_     = tx;
    ty_     = ty;
    tz_     = tz;
  }

  fHasRotation |= rhs.HasRotation();
  if (rhs.HasRotation()) {
    auto rxx = rxx_ * rhs.rxx_ + rxy_ * rhs.ryx_ + rxz_ * rhs.rzx_;
    auto ryx = ryx_ * rhs.rxx_ + ryy_ * rhs.ryx_ + ryz_ * rhs.rzx_;
    auto rzx = rzx_ * rhs.rxx_ + rzy_ * rhs.ryx_ + rzz_ * rhs.rzx_;
    auto rxy = rxx_ * rhs.rxy_ + rxy_ * rhs.ryy_ + rxz_ * rhs.rzy_;
    auto ryy = ryx_ * rhs.rxy_ + ryy_ * rhs.ryy_ + ryz_ * rhs.rzy_;
    auto rzy = rzx_ * rhs.rxy_ + rzy_ * rhs.ryy_ + rzz_ * rhs.rzy_;
    auto rxz = rxx_ * rhs.rxz_ + rxy_ * rhs.ryz_ + rxz_ * rhs.rzz_;
    auto ryz = ryx_ * rhs.rxz_ + ryy_ * rhs.ryz_ + ryz_ * rhs.rzz_;
    auto rzz = rzx_ * rhs.rxz_ + rzy_ * rhs.ryz_ + rzz_ * rhs.rzz_;

    rxx_ = rxx;
    rxy_ = rxy;
    rxz_ = rxz;
    ryx_ = ryx;
    ryy_ = ryy;
    ryz_ = ryz;
    rzx_ = rzx;
    rzy_ = rzy;
    rzz_ = rzz;
  }

  return *this;
}

VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
void Transformation3D::MultiplyFromRight(Transformation3D const &rhs)
{
  // TODO: this code should directly operator on Vector3D and Matrix3D

  if (rhs.fIdentity) return;
  fIdentity = false;

  if (rhs.HasTranslation()) {
    fHasTranslation = true;
    // ideal for fused multiply add
    tx_ += rxx_ * rhs.tx_;
    tx_ += ryx_ * rhs.ty_;
    tx_ += rzx_ * rhs.tz_;

    ty_ += rxy_ * rhs.tx_;
    ty_ += ryy_ * rhs.ty_;
    ty_ += rzy_ * rhs.tz_;

    tz_ += rxz_ * rhs.tx_;
    tz_ += ryz_ * rhs.ty_;
    tz_ += rzz_ * rhs.tz_;
  }

  if (rhs.HasRotation()) {
    fHasRotation   = true;
    Precision tmpx = rxx_;
    Precision tmpy = ryx_;
    Precision tmpz = rzx_;

    // first row of matrix
    rxx_ = tmpx * rhs.rxx_;
    ryx_ = tmpx * rhs.ryx_;
    rzx_ = tmpx * rhs.rzx_;
    rxx_ += tmpy * rhs.rxy_;
    ryx_ += tmpy * rhs.ryy_;
    rzx_ += tmpy * rhs.rzy_;
    rxx_ += tmpz * rhs.rxz_;
    ryx_ += tmpz * rhs.ryz_;
    rzx_ += tmpz * rhs.rzz_;

    tmpx = rxy_;
    tmpy = ryy_;
    tmpz = rzy_;

    // second row of matrix
    rxy_ = tmpx * rhs.rxx_;
    ryy_ = tmpx * rhs.ryx_;
    rzy_ = tmpx * rhs.rzx_;
    rxy_ += tmpy * rhs.rxy_;
    ryy_ += tmpy * rhs.ryy_;
    rzy_ += tmpy * rhs.rzy_;
    rxy_ += tmpz * rhs.rxz_;
    ryy_ += tmpz * rhs.ryz_;
    rzy_ += tmpz * rhs.rzz_;

    tmpx = rxz_;
    tmpy = ryz_;
    tmpz = rzz_;

    // third row of matrix
    rxz_ = tmpx * rhs.rxx_;
    ryz_ = tmpx * rhs.ryx_;
    rzz_ = tmpx * rhs.rzx_;
    rxz_ += tmpy * rhs.rxy_;
    ryz_ += tmpy * rhs.ryy_;
    rzz_ += tmpy * rhs.rzy_;
    rxz_ += tmpz * rhs.rxz_;
    ryz_ += tmpz * rhs.ryz_;
    rzz_ += tmpz * rhs.rzz_;
  }
}

/**
 * Only transforms by rotation, ignoring the translation part. This is useful
 * when transforming directions.
 * \param master Point to be transformed.
 * \param local Output destination of transformation.
 */
template <typename InputType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation3D::TransformDirection(
    Vector3D<InputType> const &master, Vector3D<InputType> &local) const
{
  DoRotation(master, local);
}

/**
 * Since transformation cannot be done in place, allows the transformed vector
 * to be constructed by TransformDirection directly.
 * \param master Point to be transformed.
 * \return Newly constructed Vector3D with the transformed coordinates.
 */
template <typename InputType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<InputType> Transformation3D::TransformDirection(
    Vector3D<InputType> const &master) const
{

  Vector3D<InputType> local;
  TransformDirection(master, local);
  return local;
}

std::ostream &operator<<(std::ostream &os, Transformation3D const &trans);
}
} // namespace vecgeom

#endif // VECGEOM_BASE_TRANSFORMATION3D_H_
