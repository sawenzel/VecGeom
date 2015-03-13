/// \file Transformation3D.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_TRANSFORMATION3D_H_
#define VECGEOM_BASE_TRANSFORMATION3D_H_

#include "base/Global.h"

#include "base/Vector3D.h"
#include "backend/Backend.h"

#include "backend/Backend.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "backend/cuda/Interface.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef VECGEOM_ROOT
class TGeoMatrix;
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class Transformation3D; )

inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECGEOM_NVCC
   } namespace cuda { class Transformation3D; }
   inline namespace VECGEOM_IMPL_NAMESPACE {
   //class vecgeom::cuda::Transformation3D;
#endif

class Transformation3D {


private:
  // TODO: it might be better to directly store this in terms of Vector3D<Precision> !!
  // and would allow for higher level abstraction
  Precision fTranslation[3];
  Precision fRotation[9];
  bool fIdentity;
  bool fHasRotation;
  bool fHasTranslation;

public:

  VECGEOM_CUDA_HEADER_BOTH
  constexpr Transformation3D() : fTranslation{0.,0.,0.},
  fRotation{1.,0.,0.,0.,1.,0.,0.,0.,1.}, fIdentity(true), fHasRotation(false), fHasTranslation(false) {};

  /**
   * Constructor for translation only.
   * @param tx Translation in x-coordinate.
   * @param ty Translation in y-coordinate.
   * @param tz Translation in z-coordinate.
   */
  VECGEOM_CUDA_HEADER_BOTH
  Transformation3D(const Precision tx, const Precision ty, const Precision tz) :
      fTranslation{tx,ty,tz},
      fRotation{1.,0.,0.,0.,1.,0.,0.,0.,1.},
      fIdentity( tx==0 && ty==0 && tz==0 ),
      fHasRotation(false),
      fHasTranslation( tx!=0 || ty!=0 || tz!=0 ) { }

  /**
   * @param tx Translation in x-coordinate.
   * @param ty Translation in y-coordinate.
   * @param tz Translation in z-coordinate.
   * @param phi Rotation angle about z-axis.
   * @param theta Rotation angle about new y-axis.
   * @param psi Rotation angle about new z-axis.
   */
  VECGEOM_CUDA_HEADER_BOTH
  Transformation3D(const Precision tx, const Precision ty,
                   const Precision tz, const Precision phi,
                   const Precision theta, const Precision psi);

  /**
   * Constructor to manually set each entry. Used when converting from different
   * geometry.
   */
  VECGEOM_CUDA_HEADER_BOTH
  Transformation3D(const Precision tx, const Precision ty,
                   const Precision tz, const Precision r0,
                   const Precision r1, const Precision r2,
                   const Precision r3, const Precision r4,
                   const Precision r5, const Precision r6,
                   const Precision r7, const Precision r8);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Transformation3D(Transformation3D const &other);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Transformation3D& operator=(Transformation3D const &rhs);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool operator==(Transformation3D const &rhs) const;

  VECGEOM_CUDA_HEADER_BOTH
  ~Transformation3D() {}

  int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> Translation() const {
    return Vector3D<Precision>(fTranslation[0], fTranslation[1],
                               fTranslation[2]);
  }

  /**
   * No safety against faulty indexing.
   * @param index Index of translation entry in the range [0-2].
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Translation(const int index) const { return fTranslation[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision const* Rotation() const { return fRotation; }

  /**
   * No safety against faulty indexing.
   * \param index Index of rotation entry in the range [0-8].
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Rotation(const int index) const { return fRotation[index]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool IsIdentity() const { return fIdentity; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool HasRotation() const { return fHasRotation; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool HasTranslation() const { return fHasTranslation; }

  VECGEOM_CUDA_HEADER_BOTH
  void Print() const;

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
                  Vector3D<InputType> &local) const;

  template <RotationCode code, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void DoRotation_new(Vector3D<InputType> const &master,
                      Vector3D<InputType> &local) const;
 private:

  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void DoTranslation(Vector3D<InputType> const &master,
                     Vector3D<InputType> &local) const;

  template <bool vectortransform, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void InverseTransformKernel(
      Vector3D<InputType> const &local,
      Vector3D<InputType> &master) const;

public:

  // Transformation interface

  template <TranslationCode trans_code, RotationCode rot_code,
            typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Transform(Vector3D<InputType> const &master,
                 Vector3D<InputType> &local) const;

  template <TranslationCode trans_code, RotationCode rot_code,
            typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<InputType> Transform(Vector3D<InputType> const &master) const;

  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Transform(Vector3D<InputType> const &master,
                 Vector3D<InputType> &local) const;

  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<InputType> Transform(Vector3D<InputType> const &master) const;


  template <RotationCode code, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void TransformDirection(Vector3D<InputType> const &master,
                         Vector3D<InputType> & local) const;

  template <RotationCode code, typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<InputType> TransformDirection(
      Vector3D<InputType> const &master) const;

  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void TransformDirection(Vector3D<InputType> const &master,
                         Vector3D<InputType> & local) const;

  template <typename InputType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<InputType> TransformDirection(
      Vector3D<InputType> const &master) const;

   /** The inverse transformation ( aka LocalToMaster ) of an object transform like a point
    *  this does not need to currently template on placement since such a transformation is much less used
    */
   template <typename InputType>
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   void InverseTransform(Vector3D<InputType> const &local,
                  Vector3D<InputType> &master) const;

   template <typename InputType>
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   Vector3D<InputType> InverseTransform(Vector3D<InputType> const &local) const;

   /** The inverse transformation of an object transforming like a vector */
   template <typename InputType>
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   void InverseTransformDirection(Vector3D<InputType> const &master,
                    Vector3D<InputType> & local) const;

   template <typename InputType>
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   Vector3D<InputType> InverseTransformDirection(Vector3D<InputType> const &master) const;

  /** compose transformations - multiply transformations */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void MultiplyFromRight( Transformation3D const & rhs );

  /** compose transformations - multiply transformations */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void CopyFrom( Transformation3D const & rhs )
  {
    // not sure this compiles under CUDA
    copy(&rhs, &rhs+1, this);
  }

  // Utility and CUDA

#ifdef VECGEOM_CUDA_INTERFACE
  size_t DeviceSizeOf() const { return DevicePtr<cuda::Transformation3D>::SizeOf(); }
  DevicePtr<cuda::Transformation3D> CopyToGpu() const;
  DevicePtr<cuda::Transformation3D> CopyToGpu(DevicePtr<cuda::Transformation3D> const gpu_ptr) const;
#endif


#ifdef VECGEOM_ROOT
// function to convert this transformation to a TGeo transformation
// mainly used for the benchmark comparisons with ROOT
  TGeoMatrix * ConvertToTGeoMatrix() const;
#endif

public:

  static const Transformation3D kIdentity;


}; // End class Transformation3D

VECGEOM_CUDA_HEADER_BOTH
Transformation3D::Transformation3D(Transformation3D const &other) : fIdentity(false), fHasRotation(false), fHasTranslation(false) {
  *this = other;
}

VECGEOM_CUDA_HEADER_BOTH
Transformation3D& Transformation3D::operator=(Transformation3D const &rhs) {
  copy(rhs.fTranslation, rhs.fTranslation+3, fTranslation);
  copy(rhs.fRotation, rhs.fRotation+9, fRotation);
  fIdentity = rhs.fIdentity;
  fHasTranslation = rhs.fHasTranslation;
  fHasRotation = rhs.fHasRotation;
  return *this;
}

VECGEOM_CUDA_HEADER_BOTH
bool Transformation3D::operator==(Transformation3D const &rhs) const {
  return equal(fTranslation, fTranslation+3, rhs.fTranslation) &&
         equal(fRotation, fRotation+9, rhs.fRotation);
}

/**
 * Rotates a vector to this transformation's frame of reference.
 * Templates on the RotationCode generated by GenerateTranslationCode() to
 * perform specialized rotation.
 * \sa GenerateTranslationCode()
 * \param master Vector in original frame of reference.
 * \param local Output vector rotated to the new frame of reference.
 */
template <RotationCode code, typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void Transformation3D::DoRotation(Vector3D<InputType> const &master,
                                  Vector3D<InputType> &local) const {

  if (code == 0x1B1) {
    local[0] = master[0]*fRotation[0];
    local[1] = master[1]*fRotation[4] + master[2]*fRotation[7];
    local[2] = master[1]*fRotation[5] + master[2]*fRotation[8];
    return;
  }
  if (code == 0x18E) {
    local[0] = master[1]*fRotation[3];
    local[1] = master[0]*fRotation[1] + master[2]*fRotation[7];
    local[2] = master[0]*fRotation[2] + master[2]*fRotation[8];
    return;
  }
  if (code == 0x076){
    local[0] = master[2]*fRotation[6];
    local[1] = master[0]*fRotation[1] + master[1]*fRotation[4];
    local[2] = master[0]*fRotation[2] + master[1]*fRotation[5];
    return;
  }
  if (code == 0x16A) {
    local[0] = master[1]*fRotation[3] + master[2]*fRotation[6];
    local[1] = master[0]*fRotation[1];
    local[2] = master[2]*fRotation[5] + master[2]*fRotation[8];
    return;
  }
  if (code == 0x155) {
    local[0] = master[0]*fRotation[0] + master[2]*fRotation[6];
    local[1] = master[1]*fRotation[4];
    local[2] = master[0]*fRotation[2] + master[2]*fRotation[8];
    return;
  }
  if (code == 0x0AD){
    local[0] = master[0]*fRotation[0] + master[1]*fRotation[3];
    local[1] = master[2]*fRotation[7];
    local[2] = master[0]*fRotation[2] + master[1]*fRotation[5];
    return;
  }
  if (code == 0x0DC){
    local[0] = master[1]*fRotation[3] + master[2]*fRotation[6];
    local[1] = master[1]*fRotation[4] + master[2]*fRotation[7];
    local[2] = master[0]*fRotation[2];
    return;
  }
  if (code == 0x0E3) {
    local[0] = master[0]*fRotation[0] + master[2]*fRotation[6];
    local[1] = master[0]*fRotation[1] + master[2]*fRotation[7];
    local[2] = master[1]*fRotation[5];
    return;
  }
  if (code == 0x11B){
    local[0] = master[0]*fRotation[0] + master[1]*fRotation[3];
    local[1] = master[0]*fRotation[1] + master[1]*fRotation[4];
    local[2] = master[2]*fRotation[8];
    return;
  }
  if (code == 0x0A1){
    local[0] = master[0]*fRotation[0];
    local[1] = master[2]*fRotation[7];
    local[2] = master[1]*fRotation[5];
    return;
  }
  if (code == 0x10A){
    local[0] = master[1]*fRotation[3];
    local[1] = master[0]*fRotation[1];
    local[2] = master[2]*fRotation[8];
    return;
  }
  if (code == 0x046){
    local[0] = master[1]*fRotation[3];
    local[1] = master[2]*fRotation[7];
    local[2] = master[0]*fRotation[2];
    return;
  }
  if (code == 0x062) {
    local[0] = master[2]*fRotation[6];
    local[1] = master[0]*fRotation[1];
    local[2] = master[1]*fRotation[5];
    return;
  }
  if (code == 0x054) {
    local[0] = master[2]*fRotation[6];
    local[1] = master[1]*fRotation[4];
    local[2] = master[0]*fRotation[2];
    return;
  }

  // code = 0x111;
  if (code == rotation::kDiagonal) {
    local[0] = master[0]*fRotation[0];
    local[1] = master[1]*fRotation[4];
    local[2] = master[2]*fRotation[8];
    return;
  }

  // code = 0x200;
  if (code == rotation::kIdentity){
    local = master;
    return;
  }

  // General case
  local[0] =  master[0]*fRotation[0];
  local[1] =  master[0]*fRotation[1];
  local[2] =  master[0]*fRotation[2];
  local[0] += master[1]*fRotation[3];
  local[1] += master[1]*fRotation[4];
  local[2] += master[1]*fRotation[5];
  local[0] += master[2]*fRotation[6];
  local[1] += master[2]*fRotation[7];
  local[2] += master[2]*fRotation[8];

}

/**
 * Rotates a vector to this transformation's frame of reference.
 * Templates on the RotationCode generated by GenerateTranslationCode() to
 * perform specialized rotation.
 * \sa GenerateTranslationCode()
 * \param master Vector in original frame of reference.
 * \param local Output vector rotated to the new frame of reference.
 */
template <RotationCode code, typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void Transformation3D::DoRotation_new(Vector3D<InputType> const &master,
                                      Vector3D<InputType> &local) const {

  // code = 0x200;
  if (code == rotation::kIdentity){
    local = master;
    return;
  }

  // General case
  local = Vector3D<double>();  // reset to zero -- any better way to do this???
  if(code & 0x001) { local[0] += master[0]*fRotation[0]; }
  if(code & 0x002) { local[1] += master[0]*fRotation[1]; }
  if(code & 0x004) { local[2] += master[0]*fRotation[2]; }
  if(code & 0x008) { local[0] += master[1]*fRotation[3]; }
  if(code & 0x010) { local[1] += master[1]*fRotation[4]; }
  if(code & 0x020) { local[2] += master[1]*fRotation[5]; }
  if(code & 0x040) { local[0] += master[2]*fRotation[6]; }
  if(code & 0x080) { local[1] += master[2]*fRotation[7]; }
  if(code & 0x100) { local[2] += master[2]*fRotation[8]; }
}

template <typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void Transformation3D::DoTranslation(
    Vector3D<InputType> const &master,
    Vector3D<InputType> &local) const {

  local[0] = master[0] - fTranslation[0];
  local[1] = master[1] - fTranslation[1];
  local[2] = master[2] - fTranslation[2];
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
void Transformation3D::Transform(Vector3D<InputType> const &master,
                                     Vector3D<InputType> &local) const {

  // Identity
  if (trans_code == translation::kIdentity && rot_code == rotation::kIdentity) {
    local = master;
    return;
  }

  // Only translation
  if (trans_code != translation::kIdentity && rot_code == rotation::kIdentity) {
    DoTranslation(master, local);
    return;
  }

  // Only rotation
  if (trans_code == translation::kIdentity && rot_code != rotation::kIdentity) {
    DoRotation<rot_code>(master, local);
    return;
  }

  // General case
  Vector3D<InputType> tmp;
  DoTranslation(master, tmp);
  DoRotation<rot_code>(tmp, local);
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
Vector3D<InputType> Transformation3D::Transform(
    Vector3D<InputType> const &master) const {

  Vector3D<InputType> local;
  Transform<trans_code, rot_code>(master, local);
  return local;

}

template <typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void Transformation3D::Transform(Vector3D<InputType> const &master,
                                 Vector3D<InputType> &local) const {
  Transform<translation::kGeneric, rotation::kGeneric>(master, local);
}
template <typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Vector3D<InputType> Transformation3D::Transform(
    Vector3D<InputType> const &master) const {
  return Transform<translation::kGeneric, rotation::kGeneric>(master);
}

template <bool transform_direction, typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void Transformation3D::InverseTransformKernel(
    Vector3D<InputType> const &local,
    Vector3D<InputType> &master) const {

  // we are just doing the full stuff here ( LocalToMaster is less critical
  // than other way round )

   if (transform_direction) {
      master[0] =  local[0]*fRotation[0];
      master[0] += local[1]*fRotation[1];
      master[0] += local[2]*fRotation[2];
      master[1] =  local[0]*fRotation[3];
      master[1] += local[1]*fRotation[4];
      master[1] += local[2]*fRotation[5];
      master[2] =  local[0]*fRotation[6];
      master[2] += local[1]*fRotation[7];
      master[2] += local[2]*fRotation[8];
   } else {
      master[0] = fTranslation[0];
      master[0] +=  local[0]*fRotation[0];
      master[0] += local[1]*fRotation[1];
      master[0] += local[2]*fRotation[2];
      master[1] = fTranslation[1];
      master[1] +=  local[0]*fRotation[3];
      master[1] += local[1]*fRotation[4];
      master[1] += local[2]*fRotation[5];
      master[2] = fTranslation[2];
      master[2] += local[0]*fRotation[6];
      master[2] += local[1]*fRotation[7];
      master[2] += local[2]*fRotation[8];
   }
}

template <typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void Transformation3D::InverseTransform(
    Vector3D<InputType> const &local, Vector3D<InputType> & master) const {
   InverseTransformKernel<false, InputType>(local, master);
}


template <typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Vector3D<InputType> Transformation3D::InverseTransform(
    Vector3D<InputType> const &local) const {
   Vector3D<InputType> tmp;
   InverseTransform(local, tmp);
   return tmp;
}

template <typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void Transformation3D::InverseTransformDirection(
    Vector3D<InputType> const &local, Vector3D<InputType> & master) const {
   InverseTransformKernel<true, InputType>(local, master);
}


template <typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Vector3D<InputType> Transformation3D::InverseTransformDirection(
    Vector3D<InputType> const &local) const {
   Vector3D<InputType> tmp;
   InverseTransformDirection(local, tmp);
   return tmp;
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void Transformation3D::MultiplyFromRight(Transformation3D const & rhs)
{
   // TODO: this code should directly operator on Vector3D and Matrix3D

   if(rhs.fIdentity) return;

   if(rhs.HasTranslation())
   {
   // ideal for fused multiply add
   fTranslation[0]+=fRotation[0]*rhs.fTranslation[0];
   fTranslation[0]+=fRotation[1]*rhs.fTranslation[1];
   fTranslation[0]+=fRotation[2]*rhs.fTranslation[2];

   fTranslation[1]+=fRotation[3]*rhs.fTranslation[0];
   fTranslation[1]+=fRotation[4]*rhs.fTranslation[1];
   fTranslation[1]+=fRotation[5]*rhs.fTranslation[2];

   fTranslation[2]+=fRotation[6]*rhs.fTranslation[0];
   fTranslation[2]+=fRotation[7]*rhs.fTranslation[1];
   fTranslation[2]+=fRotation[8]*rhs.fTranslation[2];
   }

   if(rhs.HasRotation())
   {
      Precision tmpx = fRotation[0];
      Precision tmpy = fRotation[1];
      Precision tmpz = fRotation[2];

      // first row of matrix
      fRotation[0] = tmpx*rhs.fRotation[0];
      fRotation[1] = tmpx*rhs.fRotation[1];
      fRotation[2] = tmpx*rhs.fRotation[2];
      fRotation[0]+= tmpy*rhs.fRotation[3];
      fRotation[1]+= tmpy*rhs.fRotation[4];
      fRotation[2]+= tmpy*rhs.fRotation[5];
      fRotation[0]+= tmpz*rhs.fRotation[6];
      fRotation[1]+= tmpz*rhs.fRotation[7];
      fRotation[2]+= tmpz*rhs.fRotation[8];

      tmpx = fRotation[3];
      tmpy = fRotation[4];
      tmpz = fRotation[5];

      // second row of matrix
      fRotation[3] = tmpx*rhs.fRotation[0];
      fRotation[4] = tmpx*rhs.fRotation[1];
      fRotation[5] = tmpx*rhs.fRotation[2];
      fRotation[3]+= tmpy*rhs.fRotation[3];
      fRotation[4]+= tmpy*rhs.fRotation[4];
      fRotation[5]+= tmpy*rhs.fRotation[5];
      fRotation[3]+= tmpz*rhs.fRotation[6];
      fRotation[4]+= tmpz*rhs.fRotation[7];
      fRotation[5]+= tmpz*rhs.fRotation[8];

      tmpx = fRotation[6];
      tmpy = fRotation[7];
      tmpz = fRotation[8];

      // third row of matrix
      fRotation[6] = tmpx*rhs.fRotation[0];
      fRotation[7] = tmpx*rhs.fRotation[1];
      fRotation[8] = tmpx*rhs.fRotation[2];
      fRotation[6]+= tmpy*rhs.fRotation[3];
      fRotation[7]+= tmpy*rhs.fRotation[4];
      fRotation[8]+= tmpy*rhs.fRotation[5];
      fRotation[6]+= tmpz*rhs.fRotation[6];
      fRotation[7]+= tmpz*rhs.fRotation[7];
      fRotation[8]+= tmpz*rhs.fRotation[8];
   }
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
void Transformation3D::TransformDirection(
    Vector3D<InputType> const &master,
    Vector3D<InputType> &local) const {

  // Rotational fIdentity
  if (code == rotation::kIdentity) {
    local = master;
    return;
  }

  // General case
  DoRotation<code>(master, local);

}

/**
 * Since transformation cannot be done in place, allows the transformed vector
 * to be constructed by TransformDirection directly.
 * \param master Point to be transformed.
 * \return Newly constructed Vector3D with the transformed coordinates.
 */
template <RotationCode code, typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Vector3D<InputType> Transformation3D::TransformDirection(
    Vector3D<InputType> const &master) const {

  Vector3D<InputType> local;
  TransformDirection<code>(master, local);
  return local;
}

template <typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void Transformation3D::TransformDirection(
    Vector3D<InputType> const &master,
    Vector3D<InputType> &local) const {
  TransformDirection<rotation::kGeneric>(master, local);
}

template <typename InputType>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Vector3D<InputType> Transformation3D::TransformDirection(
    Vector3D<InputType> const &master) const {
  return TransformDirection<rotation::kGeneric>(master);
}

std::ostream& operator<<(std::ostream& os, Transformation3D const &trans);

} } // End global namespace

#endif // VECGEOM_BASE_TRANSFORMATION3D_H_
