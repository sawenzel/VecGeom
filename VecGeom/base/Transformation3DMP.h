/// \file Transformation3DMP.h This provides the precision templated Transformation3D, which is required for mixed precision

#ifndef VECGEOM_BASE_TRANSFORMATION3DMP_H_
#define VECGEOM_BASE_TRANSFORMATION3DMP_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"

#include <VecGeom/base/Transformation3D.h>

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

#include <initializer_list>
// #ifdef VECGEOM_ROOT
// class TGeoMatrix;
// #endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class Transformation3DMP;);

inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA
}
namespace cuda {
class Transformation3DMP;
}
inline namespace VECGEOM_IMPL_NAMESPACE {
// class vecgeom::cuda::Transformation3D;
#endif

template <typename Real_s>
class Transformation3DMP {

  // On the precision templating: Real_s is the precision used for the stored variables,
  // Real_i the one for the input variables.

private:
  // TODO: it might be better to directly store this in terms of Vector3D<Real_t> !!
  // and would allow for higher level abstraction
  Real_s tx_{0.}, ty_{0.}, tz_{0.};
  Real_s rxx_{1.}, ryx_{0.}, rzx_{0.};
  Real_s rxy_{0.}, ryy_{1.}, rzy_{0.};
  Real_s rxz_{0.}, ryz_{0.}, rzz_{1.};
  bool fIdentity;
  bool fHasRotation;
  bool fHasTranslation;

public:
  VECCORE_ATT_HOST_DEVICE
  constexpr Transformation3DMP() : fIdentity(true), fHasRotation(false), fHasTranslation(false){};

  /**
   * Constructor for translation only.
   * @param tx Translation in x-coordinate.
   * @param ty Translation in y-coordinate.
   * @param tz Translation in z-coordinate.
   */
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE Transformation3DMP(const Real_i tx, const Real_i ty, const Real_i tz)
      : tx_{static_cast<Real_s>(tx)}, ty_{static_cast<Real_s>(ty)}, tz_{static_cast<Real_s>(tz)},
        fIdentity(tx == 0 && ty == 0 && tz == 0), fHasRotation(false), fHasTranslation(tx != 0 || ty != 0 || tz != 0)
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
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE Transformation3DMP(const Real_i tx, const Real_i ty, const Real_i tz, const Real_i phi,
                                             const Real_i theta, const Real_i psi)
      : fIdentity(false), fHasRotation(true), fHasTranslation(true)
  {
    SetTranslation(tx, ty, tz);
    SetRotation(phi, theta, psi);
    SetProperties();
  }

  // enabling initilizer lists
  template <typename Real_i>
  Transformation3DMP(std::initializer_list<Real_i> values)
  {
    auto it      = values.begin();
    tx_          = *it++;
    ty_          = *it++;
    tz_          = *it++;
    Real_i phi   = *it++;
    Real_i theta = *it++;
    Real_i psi   = *it;
    SetTranslation(tx_, ty_, tz_);
    SetRotation(phi, theta, psi);
    SetProperties();
  }

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
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE Transformation3DMP(const Real_i tx, const Real_i ty, const Real_i tz, const Real_i phi,
                                             const Real_i theta, const Real_i psi, Real_i sx, Real_i sy, Real_i sz)
      : fIdentity(false), fHasRotation(true), fHasTranslation(true)
  {
    SetTranslation(tx, ty, tz);
    SetRotation(phi, theta, psi);
    ApplyScale(sx, sy, sz);
    SetProperties();
  }

  /**
   * Constructor to manually set each entry. Used when converting from different
   * geometry.
   */
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE Transformation3DMP(const Real_i tx, const Real_i ty, const Real_i tz, const Real_i rxx,
                                             const Real_i ryx, const Real_i rzx, const Real_i rxy, const Real_i ryy,
                                             const Real_i rzy, const Real_i rxz, const Real_i ryz, const Real_i rzz)
      : fIdentity(false), fHasRotation(true), fHasTranslation(true)
  {
    SetTranslation(tx, ty, tz);
    SetRotation(rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz);
    SetProperties();
  }
  /**
   * Constructor using the rotation from a different transformation
   */
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE Transformation3DMP(const Real_i tx, const Real_i ty, const Real_i tz,
                                             Transformation3DMP<Real_i> const &trot)
  {
    SetTranslation(tx, ty, tz);
    SetRotation(trot.Rotation()[0], trot.Rotation()[1], trot.Rotation()[2], trot.Rotation()[3], trot.Rotation()[4],
                trot.Rotation()[5], trot.Rotation()[6], trot.Rotation()[7], trot.Rotation()[8]);
    SetProperties();
  }
  /**
   * Constructor to manually set each entry. Used when converting from different
   * geometry, supporting scaling.
   */
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE Transformation3DMP(const Real_i tx, const Real_i ty, const Real_i tz, const Real_i rxx,
                                             const Real_i ryx, const Real_i rzx, const Real_i rxy, const Real_i ryy,
                                             const Real_i rzy, const Real_i rxz, const Real_i ryz, const Real_i rzz,
                                             Real_i sx, Real_i sy, Real_i sz)
  {
    SetTranslation(tx, ty, tz);
    SetRotation(rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz);
    ApplyScale(sx, sy, sz);
    SetProperties();
  }

  /**
   * Constructor copying the translation and rotation from memory
   * geometry.
   */
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE Transformation3DMP(const Real_i *trans, const Real_i *rot, bool has_trans, bool has_rot)
  {
    this->Set(trans, rot, has_trans, has_rot);
  }

  // /**
  //  * Constructor for a rotation based on a given direction
  //  * @param axis direction of the new z axis
  //  * @param inverse if true the origial axis will be rotated into (0,0,u)
  //                   if false a vector (0,0,u) will be rotated into the original axis
  //  */
  // VECCORE_ATT_HOST_DEVICE
  // Transformation3D(const Vector3D<Precision> &axis, bool inverse = true);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool operator==(Transformation3DMP const &rhs) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Transformation3DMP operator*(Transformation3DMP const &tf) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Transformation3DMP const &operator*=(Transformation3DMP const &rhs);

  // operator for multiplication with Transformation3D
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Transformation3DMP const &operator*=(Transformation3D const &rhs);

  VECCORE_ATT_HOST_DEVICE
  ~Transformation3DMP() {}

  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE void ApplyScale(const Real_i sx, const Real_i sy, const Real_i sz)
  {
    rxx_ *= static_cast<Real_s>(sx);
    ryx_ *= static_cast<Real_s>(sy);
    rzx_ *= static_cast<Real_s>(sz);
    rxy_ *= static_cast<Real_s>(sx);
    ryy_ *= static_cast<Real_s>(sy);
    rzy_ *= static_cast<Real_s>(sz);
    rxz_ *= static_cast<Real_s>(sx);
    ryz_ *= static_cast<Real_s>(sy);
    rzz_ *= static_cast<Real_s>(sz);
  }

  // CHECK THIS IF WE NEED BOTH WITH MP and WITHOUT MP INPUT OR ONLY THIS ONE FIXME
  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE bool ApproxEqual(Transformation3DMP<Real_i> const &rhs,
                                           Real_i tolerance = kToleranceDist<Real_i>) const
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
  Vector3D<Real_s> Translation() const { return Vector3D<Real_s>(tx_, ty_, tz_); }

  // VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  // Vector3D<Real_s> Scale() const
  // {
  //   Real_s sx = std::sqrt(rxx_ * rxx_ + rxy_ * rxy_ + rxz_ * rxz_);
  //   Real_s sy = std::sqrt(ryx_ * ryx_ + ryy_ * ryy_ + ryz_ * ryz_);
  //   Real_s sz = std::sqrt(rzx_ * rzx_ + rzy_ * rzy_ + rzz_ * rzz_);

  //   if (Determinant() < 0) sz = -sz;

  //   return Vector3D<Real_s>(sx, sy, sz);
  // }

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
   * \param index Index of rotation entry in the range [0-8].
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Real_s Rotation(const int index) const { return *(&rxx_ + index); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsIdentity() const { return fIdentity; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsXYRotation() const
  {
    return (fHasRotation && (std::abs(rzx_) < vecgeom::kTolerance) && (std::abs(rzy_) < vecgeom::kTolerance) &&
            (std::abs(rxz_) < vecgeom::kTolerance) && (std::abs(ryz_) < vecgeom::kTolerance) &&
            (std::abs(rzz_ - Real_s(1.)) < vecgeom::kTolerance));
  }

  // VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  // bool IsReflected() const { return Determinant() < 0; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool HasRotation() const { return fHasRotation; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool HasTranslation() const { return fHasTranslation; }

  // VECCORE_ATT_HOST_DEVICE
  // void Print() const;

  // // print to a stream
  // void Print(std::ostream &) const;

  // VECCORE_ATT_HOST_DEVICE
  // void PrintG4() const;

  void Print(std::ostream &s) const
  {
    s << "Transformation3DMP {{" << tx_ << "," << ty_ << "," << tz_ << "}";
    s << "{" << rxx_ << "," << ryx_ << "," << rzx_ << "," << rxy_ << "," << ryy_ << "," << rzy_ << "," << rxz_ << ","
      << ryz_ << "," << rzz_ << "}}\n";
  }

  VECCORE_ATT_HOST_DEVICE
  void Print() const
  {
    printf("Transformation3D {{%.2f, %.2f, %.2f}, ", tx_, ty_, tz_);
    printf("{%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f}}", rxx_, ryx_, rzx_, rxy_, ryy_, rzy_, rxz_, ryz_,
           rzz_);
  }

  VECCORE_ATT_HOST_DEVICE
  void PrintG4() const
  {
    using vecCore::math::Abs;
    using vecCore::math::Max;
    constexpr double deviationTolerance = 1.0e-05;
    printf("  TransformationMP: \n");

    bool UnitTr            = !fHasRotation;
    double diagDeviation   = Max(Abs(rxx_ - 1.0), Abs(ryy_ - 1.0), Abs(rzz_ - 1.0));
    double offdDeviationUL = Max(Abs(rxy_), Abs(rxz_), Abs(ryx_));
    double offdDeviationDR = Max(Abs(ryz_), Abs(rzx_), Abs(rzy_));
    double offdDeviation   = Max(offdDeviationUL, offdDeviationDR);

    if (UnitTr || Max(diagDeviation, offdDeviation) < deviationTolerance) {
      printf("    UNIT  Rotation \n");
    } else {
      printf("rx/x,y,z: %.6g %.6g %.6g\nry/x,y,z: %.6g %.6g %.6g\nrz/x,y,z: %.6g %.6g %.6g\n", rxx_, rxy_, rxz_, ryx_,
             ryy_, ryz_, rzx_, rzy_, rzz_);
    }

    printf("tr/x,y,z: %.6g %.6g %.6g\n", tx_, ty_, tz_);
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
    fHasRotation    = (fabs(rxx_ - 1.) > kTolerance) || (fabs(ryx_) > kTolerance) || (fabs(rzx_) > kTolerance) ||
                           (fabs(rxy_) > kTolerance) || (fabs(ryy_ - 1.) > kTolerance) || (fabs(rzy_) > kTolerance) ||
                           (fabs(rxz_) > kTolerance) || (fabs(ryz_) > kTolerance) || (fabs(rzz_ - 1.) > kTolerance)
                          ? true
                          : false;
    fIdentity       = !fHasTranslation && !fHasRotation;
  }

  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE void SetRotation(const Real_i phi, const Real_i theta, const Real_i psi)
  {

    const auto sinphi = sin(kDegToRad * phi);
    const auto cosphi = cos(kDegToRad * phi);
    const auto sinthe = sin(kDegToRad * theta);
    const auto costhe = cos(kDegToRad * theta);
    const auto sinpsi = sin(kDegToRad * psi);
    const auto cospsi = cos(kDegToRad * psi);

    rxx_ = static_cast<Real_s>(cospsi * cosphi - costhe * sinphi * sinpsi);
    ryx_ = static_cast<Real_s>(-sinpsi * cosphi - costhe * sinphi * cospsi);
    rzx_ = static_cast<Real_s>(sinthe * sinphi);
    rxy_ = static_cast<Real_s>(cospsi * sinphi + costhe * cosphi * sinpsi);
    ryy_ = static_cast<Real_s>(-sinpsi * sinphi + costhe * cosphi * cospsi);
    rzy_ = static_cast<Real_s>(-sinthe * cosphi);
    rxz_ = static_cast<Real_s>(sinpsi * sinthe);
    ryz_ = static_cast<Real_s>(cospsi * sinthe);
    rzz_ = static_cast<Real_s>(costhe);
  }

  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE void SetRotation(Vector3D<Real_i> const &vec)
  {
    SetRotation(vec[0], vec[1], vec[2]);
  }

  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE void SetRotation(const Real_i xx, const Real_i yx, const Real_i zx, const Real_i xy,
                                           const Real_i yy, const Real_i zy, const Real_i xz, const Real_i yz,
                                           const Real_i zz)
  {
    rxx_ = static_cast<Real_s>(xx);
    ryx_ = static_cast<Real_s>(yx);
    rzx_ = static_cast<Real_s>(zx);
    rxy_ = static_cast<Real_s>(xy);
    ryy_ = static_cast<Real_s>(yy);
    rzy_ = static_cast<Real_s>(zy);
    rxz_ = static_cast<Real_s>(xz);
    ryz_ = static_cast<Real_s>(yz);
    rzz_ = static_cast<Real_s>(zz);
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
   * \param rot Pointer to at least 9 values.
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
      rzx_ = static_cast<Real_s>(rot[2]);
      rxy_ = static_cast<Real_s>(rot[3]);
      ryy_ = static_cast<Real_s>(rot[4]);
      rzy_ = static_cast<Real_s>(rot[5]);
      rxz_ = static_cast<Real_s>(rot[6]);
      ryz_ = static_cast<Real_s>(rot[7]);
      rzz_ = static_cast<Real_s>(rot[8]);
    }

    fHasTranslation = has_trans;
    fHasRotation    = has_rot;
    fIdentity       = !fHasTranslation && !fHasRotation;
  }

private:
  // Templated rotation and translation methods which inline and compile to
  // optimized versions.

  template <typename Real_i>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void DoRotation(Vector3D<Real_i> const &master,
                                                               Vector3D<Real_i> &local) const;

private:
  template <typename Real_i>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void DoTranslation(Vector3D<Real_i> const &master,
                                                                  Vector3D<Real_i> &local) const;

  template <bool vectortransform, typename Real_i>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void InverseTransformKernel(Vector3D<Real_i> const &local,
                                                                           Vector3D<Real_i> &master) const;

public:
  // Transformation interface

  template <typename Real_i>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transform(Vector3D<Real_i> const &master,
                                                              Vector3D<Real_i> &local) const;

  template <typename Real_i>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_i> Transform(Vector3D<Real_i> const &master) const;

  template <typename Real_i>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void TransformDirection(Vector3D<Real_i> const &master,
                                                                       Vector3D<Real_i> &local) const;

  template <typename Real_i>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_i> TransformDirection(
      Vector3D<Real_i> const &master) const;

  /** The inverse transformation ( aka LocalToMaster ) of an object transform like a point
   *  this does not need to currently template on placement since such a transformation is much less used
   */
  template <typename Real_i>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void InverseTransform(Vector3D<Real_i> const &local,
                                                                     Vector3D<Real_i> &master) const;

  template <typename Real_i>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_i> InverseTransform(Vector3D<Real_i> const &local) const;

  /** The inverse transformation of an object transforming like a vector */
  template <typename Real_i>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void InverseTransformDirection(Vector3D<Real_i> const &master,
                                                                              Vector3D<Real_i> &local) const;

  template <typename Real_i>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_i> InverseTransformDirection(
      Vector3D<Real_i> const &master) const;

  /** compose transformations - multiply transformations */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void MultiplyFromRight(Transformation3DMP const &rhs);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void MultiplyFromRight(Transformation3D const &rhs);

  // FIXME
  // /** compose transformations - multiply transformations */
  // VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  // void CopyFrom(Transformation3D const &rhs)
  // {
  //   // not sure this compiles under CUDA
  //   copy(&rhs, &rhs + 1, this);
  // }

  // VECCORE_ATT_HOST_DEVICE
  // Transformation3D &RotateX(double a);

  // VECCORE_ATT_HOST_DEVICE
  // Transformation3D &RotateY(double a);

  // VECCORE_ATT_HOST_DEVICE
  // Transformation3D &RotateZ(double a);

  // /**
  //  * @brief Returns determinant for the rotation.
  //  */
  // VECCORE_ATT_HOST_DEVICE
  // double Determinant() const
  // {
  //   // Computes determinant in double precision
  //   double xx_ = rxx_;
  //   double xy_ = rxy_;
  //   double xz_ = rxz_;
  //   double yx_ = ryx_;
  //   double yy_ = ryy_;
  //   double yz_ = ryz_;
  //   double zx_ = rzx_;
  //   double zy_ = rzy_;
  //   double zz_ = rzz_;

  //   double detxx = yy_ * zz_ - yz_ * zy_;
  //   double detxy = yx_ * zz_ - yz_ * zx_;
  //   double detxz = yx_ * zy_ - yy_ * zx_;
  //   double det   = xx_ * detxx - xy_ * detxy + xz_ * detxz;
  //   return det;
  // }

  // /**
  //  * @brief Returns rotation axis as a unit vector
  //  */
  // VECCORE_ATT_HOST_DEVICE
  // Vector3D<double> Axis() const;

  /**
   * @brief Rectifies the round-off making the rotation true.
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
  // VECCORE_ATT_HOST_DEVICE
  // Transformation3D &Rectify();

  // /**
  //  * @brief Stores the inverse of this matrix into inverse. Taken from G4AffineTransformation
  //  */
  // VECCORE_ATT_HOST_DEVICE
  // Transformation3D &Invert()
  // {
  //   double ttx = -tx_, tty = -ty_, ttz = -tz_;
  //   tx_ = ttx * rxx_ + tty * rxy_ + ttz * rxz_;
  //   ty_ = ttx * ryx_ + tty * ryy_ + ttz * ryz_;
  //   tz_ = ttx * rzx_ + tty * rzy_ + ttz * rzz_;

  //   auto tmp1 = ryx_;
  //   ryx_      = rxy_;
  //   rxy_      = tmp1;
  //   auto tmp2 = rzx_;
  //   rzx_      = rxz_;
  //   rxz_      = tmp2;
  //   auto tmp3 = rzy_;
  //   rzy_      = ryz_;
  //   ryz_      = tmp3;
  //   return *this;
  // }

  /**
   * @brief Returns the inverse of this matrix. Taken from G4AffineTransformation
   */
  VECCORE_ATT_HOST_DEVICE
  Transformation3DMP<Real_s> Inverse() const
  {
    Real_s ttx = -tx_, tty = -ty_, ttz = -tz_;
    return Transformation3DMP<Real_s>(ttx * rxx_ + tty * rxy_ + ttz * rxz_, ttx * ryx_ + tty * ryy_ + ttz * ryz_,
                                      ttx * rzx_ + tty * rzy_ + ttz * rzz_, rxx_, rxy_, rxz_, ryx_, ryy_, ryz_, rzx_,
                                      rzy_, rzz_);
  }

  // Utility and CUDA

#ifdef VECGEOM_CUDA_INTERFACE
  size_t DeviceSizeOf() const { return DevicePtr<cuda::Transformation3DMP>::SizeOf(); }
  DevicePtr<cuda::Transformation3DMP> CopyToGpu() const;
  DevicePtr<cuda::Transformation3DMP> CopyToGpu(DevicePtr<cuda::Transformation3DMP> const gpu_ptr) const;
  static void CopyManyToGpu(const std::vector<Transformation3DMP const *> &trafos,
                            const std::vector<DevicePtr<cuda::Transformation3DMP>> &gpu_ptrs);
#endif

  // #ifdef VECGEOM_ROOT
  //   // function to convert this transformation to a TGeo transformation
  //   // mainly used for the benchmark comparisons with ROOT
  //   static TGeoMatrix *ConvertToTGeoMatrix(Transformation3D const &);
  // #endif

public:
  static const Transformation3DMP kIdentity;

}; // End class Transformation3D

template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool Transformation3DMP<Real_t>::operator==(Transformation3DMP<Real_t> const &rhs) const
{
  return equal(&tx_, &tx_ + 12, &rhs.tx_);
}

/**
 * Rotates a vector to this transformation's frame of reference.
 * \param master Vector in original frame of reference.
 * \param local Output vector rotated to the new frame of reference.
 */
template <typename Real_s>
template <typename Real_i>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation3DMP<Real_s>::DoRotation(Vector3D<Real_i> const &master,
                                                                                         Vector3D<Real_i> &local) const
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

template <typename Real_s>
template <typename Real_i>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation3DMP<Real_s>::DoTranslation(
    Vector3D<Real_i> const &master, Vector3D<Real_i> &local) const
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
template <typename Real_i>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation3DMP<Real_s>::Transform(Vector3D<Real_i> const &master,
                                                                                        Vector3D<Real_i> &local) const
{
  Vector3D<Real_i> tmp;
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
template <typename Real_i>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_i> Transformation3DMP<Real_s>::Transform(
    Vector3D<Real_i> const &master) const
{

  Vector3D<Real_i> local;
  Transform(master, local);
  return local;
}

template <typename Real_s>
template <bool transform_direction, typename Real_i>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation3DMP<Real_s>::InverseTransformKernel(
    Vector3D<Real_i> const &local, Vector3D<Real_i> &master) const
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

template <typename Real_s>
template <typename Real_i>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation3DMP<Real_s>::InverseTransform(
    Vector3D<Real_i> const &local, Vector3D<Real_i> &master) const
{
  InverseTransformKernel<false, Real_i>(local, master);
}

template <typename Real_s>
template <typename Real_i>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_i> Transformation3DMP<Real_s>::InverseTransform(
    Vector3D<Real_i> const &local) const
{
  Vector3D<Real_i> tmp;
  InverseTransform(local, tmp);
  return tmp;
}

template <typename Real_s>
template <typename Real_i>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation3DMP<Real_s>::InverseTransformDirection(
    Vector3D<Real_i> const &local, Vector3D<Real_i> &master) const
{
  InverseTransformKernel<true, Real_i>(local, master);
}

template <typename Real_s>
template <typename Real_i>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Real_i> Transformation3DMP<Real_s>::InverseTransformDirection(
    Vector3D<Real_i> const &local) const
{
  Vector3D<Real_i> tmp;
  InverseTransformDirection(local, tmp);
  return tmp;
}

template <typename Real_s>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Transformation3DMP<Real_s> Transformation3DMP<Real_s>::operator*(
    Transformation3DMP<Real_s> const &rhs) const
{
  if (rhs.fIdentity) return Transformation3DMP<Real_s>(*this);
  return Transformation3DMP<Real_s>(tx_ * rhs.rxx_ + ty_ * rhs.ryx_ + tz_ * rhs.rzx_ + rhs.tx_, // tx
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

template <typename Real_s>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Transformation3DMP<Real_s> const &Transformation3DMP<Real_s>::operator*=(
    Transformation3DMP<Real_s> const &rhs)
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

// This function is slow due to static cast and must only be used at construction but never at for propagation of rays!
// It is only used for interfacing Transformation3D's which might be obtained from solids directly
// This is needed e.g., in TopMatrixImpl in NavStateIndex.h
template <typename Real_s>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Transformation3DMP<Real_s> const &Transformation3DMP<Real_s>::operator*=(
    Transformation3D const &rhs)
{
  if (rhs.IsIdentity()) return *this;
  fIdentity = false;

  fHasTranslation |= rhs.HasTranslation();
  if (fHasTranslation) {
    auto tx = tx_ * rhs.Rotation()[0] + ty_ * rhs.Rotation()[1] + tz_ * rhs.Rotation()[2] + rhs.Translation(0);
    auto ty = tx_ * rhs.Rotation()[3] + ty_ * rhs.Rotation()[4] + tz_ * rhs.Rotation()[5] + rhs.Translation(1);
    auto tz = tx_ * rhs.Rotation()[6] + ty_ * rhs.Rotation()[7] + tz_ * rhs.Rotation()[8] + rhs.Translation(2);
    tx_     = static_cast<Real_s>(tx);
    ty_     = static_cast<Real_s>(ty);
    tz_     = static_cast<Real_s>(tz);
  }

  fHasRotation |= rhs.HasRotation();
  if (rhs.HasRotation()) {
    auto rxx = rxx_ * rhs.Rotation()[0] + rxy_ * rhs.Rotation()[1] + rxz_ * rhs.Rotation()[2];
    auto ryx = ryx_ * rhs.Rotation()[0] + ryy_ * rhs.Rotation()[1] + ryz_ * rhs.Rotation()[2];
    auto rzx = rzx_ * rhs.Rotation()[0] + rzy_ * rhs.Rotation()[1] + rzz_ * rhs.Rotation()[2];
    auto rxy = rxx_ * rhs.Rotation()[3] + rxy_ * rhs.Rotation()[4] + rxz_ * rhs.Rotation()[5];
    auto ryy = ryx_ * rhs.Rotation()[3] + ryy_ * rhs.Rotation()[4] + ryz_ * rhs.Rotation()[5];
    auto rzy = rzx_ * rhs.Rotation()[3] + rzy_ * rhs.Rotation()[4] + rzz_ * rhs.Rotation()[5];
    auto rxz = rxx_ * rhs.Rotation()[6] + rxy_ * rhs.Rotation()[7] + rxz_ * rhs.Rotation()[8];
    auto ryz = ryx_ * rhs.Rotation()[6] + ryy_ * rhs.Rotation()[7] + ryz_ * rhs.Rotation()[8];
    auto rzz = rzx_ * rhs.Rotation()[6] + rzy_ * rhs.Rotation()[7] + rzz_ * rhs.Rotation()[8];

    rxx_ = static_cast<Real_s>(rxx);
    rxy_ = static_cast<Real_s>(rxy);
    rxz_ = static_cast<Real_s>(rxz);
    ryx_ = static_cast<Real_s>(ryx);
    ryy_ = static_cast<Real_s>(ryy);
    ryz_ = static_cast<Real_s>(ryz);
    rzx_ = static_cast<Real_s>(rzx);
    rzy_ = static_cast<Real_s>(rzy);
    rzz_ = static_cast<Real_s>(rzz);
  }

  return *this;
}

template <typename Real_s>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE void Transformation3DMP<Real_s>::MultiplyFromRight(
    Transformation3DMP<Real_s> const &rhs)
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

// multiply from right with Transformation3D. This will be slow and should only be called at initialization, not during
// the computation!
template <typename Real_s>
VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE void Transformation3DMP<Real_s>::MultiplyFromRight(
    Transformation3D const &rhs)
{
  // TODO: this code should directly operator on Vector3D and Matrix3D

  if (rhs.IsIdentity()) return;
  fIdentity = false;

  if (rhs.HasTranslation()) {
    fHasTranslation = true;
    // ideal for fused multiply add
    tx_ += rxx_ * static_cast<Real_s>(rhs.Translation(0));
    tx_ += ryx_ * static_cast<Real_s>(rhs.Translation(1));
    tx_ += rzx_ * static_cast<Real_s>(rhs.Translation(2));

    ty_ += rxy_ * static_cast<Real_s>(rhs.Translation(0));
    ty_ += ryy_ * static_cast<Real_s>(rhs.Translation(1));
    ty_ += rzy_ * static_cast<Real_s>(rhs.Translation(2));

    tz_ += rxz_ * static_cast<Real_s>(rhs.Translation(0));
    tz_ += ryz_ * static_cast<Real_s>(rhs.Translation(1));
    tz_ += rzz_ * static_cast<Real_s>(rhs.Translation(2));
  }

  if (rhs.HasRotation()) {
    fHasRotation = true;
    Real_s tmpx  = rxx_;
    Real_s tmpy  = ryx_;
    Real_s tmpz  = rzx_;

    // first row of matrix
    rxx_ = tmpx * static_cast<Real_s>(rhs.Rotation()[0]);
    ryx_ = tmpx * static_cast<Real_s>(rhs.Rotation()[1]);
    rzx_ = tmpx * static_cast<Real_s>(rhs.Rotation()[2]);
    rxx_ += tmpy * static_cast<Real_s>(rhs.Rotation()[3]);
    ryx_ += tmpy * static_cast<Real_s>(rhs.Rotation()[4]);
    rzx_ += tmpy * static_cast<Real_s>(rhs.Rotation()[5]);
    rxx_ += tmpz * static_cast<Real_s>(rhs.Rotation()[6]);
    ryx_ += tmpz * static_cast<Real_s>(rhs.Rotation()[7]);
    rzx_ += tmpz * static_cast<Real_s>(rhs.Rotation()[8]);

    tmpx = rxy_;
    tmpy = ryy_;
    tmpz = rzy_;

    // second row of matrix
    rxy_ = tmpx * static_cast<Real_s>(rhs.Rotation()[0]);
    ryy_ = tmpx * static_cast<Real_s>(rhs.Rotation()[1]);
    rzy_ = tmpx * static_cast<Real_s>(rhs.Rotation()[2]);
    rxy_ += tmpy * static_cast<Real_s>(rhs.Rotation()[3]);
    ryy_ += tmpy * static_cast<Real_s>(rhs.Rotation()[4]);
    rzy_ += tmpy * static_cast<Real_s>(rhs.Rotation()[5]);
    rxy_ += tmpz * static_cast<Real_s>(rhs.Rotation()[6]);
    ryy_ += tmpz * static_cast<Real_s>(rhs.Rotation()[7]);
    rzy_ += tmpz * static_cast<Real_s>(rhs.Rotation()[8]);

    tmpx = rxz_;
    tmpy = ryz_;
    tmpz = rzz_;

    // third row of matrix
    rxz_ = tmpx * static_cast<Real_s>(rhs.Rotation()[0]);
    ryz_ = tmpx * static_cast<Real_s>(rhs.Rotation()[1]);
    rzz_ = tmpx * static_cast<Real_s>(rhs.Rotation()[2]);
    rxz_ += tmpy * static_cast<Real_s>(rhs.Rotation()[3]);
    ryz_ += tmpy * static_cast<Real_s>(rhs.Rotation()[4]);
    rzz_ += tmpy * static_cast<Real_s>(rhs.Rotation()[5]);
    rxz_ += tmpz * static_cast<Real_s>(rhs.Rotation()[6]);
    ryz_ += tmpz * static_cast<Real_s>(rhs.Rotation()[7]);
    rzz_ += tmpz * static_cast<Real_s>(rhs.Rotation()[8]);
  }
}

/**
 * Only transforms by rotation, ignoring the translation part. This is useful
 * when transforming directions.
 * \param master Point to be transformed.
 * \param local Output destination of transformation.
 */
template <typename Real_s>
template <typename InputType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void Transformation3DMP<Real_s>::TransformDirection(
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
template <typename Real_s>
template <typename InputType>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<InputType> Transformation3DMP<Real_s>::TransformDirection(
    Vector3D<InputType> const &master) const
{

  Vector3D<InputType> local;
  TransformDirection(master, local);
  return local;
}

template <typename Real_s>
std::ostream &operator<<(std::ostream &os, Transformation3DMP<Real_s> const &trans)
{
  os << "TransformationMP {" << trans.Translation() << ", "
     << "(" << trans.Rotation(0) << ", " << trans.Rotation(1) << ", " << trans.Rotation(2) << ", " << trans.Rotation(3)
     << ", " << trans.Rotation(4) << ", " << trans.Rotation(5) << ", " << trans.Rotation(6) << ", " << trans.Rotation(7)
     << ", " << trans.Rotation(8) << ")}"
     << "; identity(" << trans.IsIdentity() << "); rotation(" << trans.HasRotation() << ")";
  return os;
}
}
} // namespace vecgeom

#endif // VECGEOM_BASE_TRANSFORMATION3DMP_H_
