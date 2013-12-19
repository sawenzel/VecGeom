#ifndef LIBRARYGENERIC_H
#define LIBRARYGENERIC_H

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include "tbb/tick_count.h"

#ifndef __CUDACC__
#define STD_CXX11
#define CUDA_HEADER_DEVICE
#define CUDA_HEADER_HOST
#define CUDA_HEADER_BOTH
#include <Vc/Vc>
#else
#define NVCC
#define CUDA_HEADER_DEVICE __device__
#define CUDA_HEADER_HOST __host__
#define CUDA_HEADER_BOTH __host__ __device__
#endif /* __CUDACC__ */

const int kAlignmentBoundary = 32;
const double kDegToRad = M_PI/180.;
const double kRadToDeg = 180./M_PI;

enum Ct { kVc, kCuda };

template <Ct ct>
struct CtTraits {};

template <Ct ct, typename Type>
struct SOA3D {};

template <typename Type>
struct Vector3D {

private:

  Type vec[3];

public:

  #ifdef STD_CXX11
  Vector3D() : vec{0, 0, 0} {};
  #endif /* STD_CXX11 */

  CUDA_HEADER_BOTH
  Vector3D(const Type a,
           const Type b,
           const Type c) {
    vec[0] = a;
    vec[1] = b;
    vec[2] = c;
  }

  CUDA_HEADER_BOTH
  Vector3D(Vector3D const &other) {
    vec[0] = other[0];
    vec[1] = other[1];
    vec[2] = other[2];
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Type& operator[](const int index) {
    return vec[index];
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Type operator[](const int index) const {
    return vec[index];
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D& operator+=(Vector3D const &rhs) {
    this->vec[0] += rhs[0];
    this->vec[1] += rhs[1];
    this->vec[2] += rhs[2];
    return this;
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D& operator-=(Vector3D const &rhs) {
    this->vec[0] -= rhs[0];
    this->vec[1] -= rhs[1];
    this->vec[2] -= rhs[2];
    return this;
  }

  #ifdef STD_CXX11
  friend inline __attribute__((always_inline))
  std::ostream& operator<<(std::ostream& os, Vector3D<Type> const &v) {
    os << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
    return os;
  }
  #endif /* STD_CXX11 */

};

template <typename Type>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
Vector3D<Type> operator+(Vector3D<Type> const &lhs,
                                Vector3D<Type> const &rhs) {
  return Vector3D<Type>(lhs[0]+rhs[0], lhs[1]+rhs[1], lhs[2]+rhs[2]);
}

template <typename Type>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
Vector3D<Type> operator-(Vector3D<Type> const &lhs,
                                Vector3D<Type> const &rhs) {
  return Vector3D<Type>(lhs[0]-rhs[0], lhs[1]-rhs[1], lhs[2]-rhs[2]);
}

template<typename Type>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
Vector3D<Type> abs(Vector3D<Type> const &in) {
  return Vector3D<Type>(abs(in[0]), abs(in[1]), abs(in[2]));
}

struct Stopwatch {
  tbb::tick_count t1;
  tbb::tick_count t2;
  void Start() { t1 = tbb::tick_count::now(); }
  void Stop() { t2 = tbb::tick_count::now(); }
  double Elapsed() const { return (t2-t1).seconds(); }
};

template<Ct ct, typename Type>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
Type IfThenElse(const typename CtTraits<ct>::bool_v &cond,
                const Type &thenval, const Type &elseval);

enum RotationType { kDiagonal = 720, kNone = 1296 };

template<Ct ct>
struct TransMatrix {

private:

  typename CtTraits<ct>::float_t trans[3];
  typename CtTraits<ct>::float_t rot[9];
  bool identity;
  bool has_rotation;
  bool has_translation;

public:

  TransMatrix(const typename CtTraits<ct>::float_t tx,
              const typename CtTraits<ct>::float_t ty,
              const typename CtTraits<ct>::float_t tz,
              const typename CtTraits<ct>::float_t phi,
              const typename CtTraits<ct>::float_t theta,
              const typename CtTraits<ct>::float_t psi) {
    SetTranslation(tx, ty, tz);
    SetRotation(phi, theta, psi);
  }

  inline __attribute__((always_inline))
  bool IsIdentity() const { return identity; }

  inline __attribute__((always_inline))
  bool HasRotation() const { return has_rotation; }

  inline __attribute__((always_inline))
  bool HasTranslation() const { return has_translation; }

  inline __attribute__((always_inline))
  void SetTranslation(const typename CtTraits<ct>::float_t tx,
                      const typename CtTraits<ct>::float_t ty,
                      const typename CtTraits<ct>::float_t tz) {
    trans[0] = tx;
    trans[1] = ty;
    trans[2] = tz;
    SetProperties();
  }

  inline __attribute__((always_inline))
  void SetProperties() {
    has_translation = (trans[0] || trans[1] || trans[2]) ? true : false;
    has_rotation = (RotationFootprint(rot) == kNone) ? false : true;
    identity = !has_translation && !has_rotation;
  }

  inline __attribute__((always_inline))
  void SetRotation(const typename CtTraits<ct>::float_t phi,
                   const typename CtTraits<ct>::float_t theta,
                   const typename CtTraits<ct>::float_t psi) {

    typename CtTraits<ct>::float_t sinphi = sin(kDegToRad*phi);
    typename CtTraits<ct>::float_t cosphi = cos(kDegToRad*phi);
    typename CtTraits<ct>::float_t sinthe = sin(kDegToRad*theta);
    typename CtTraits<ct>::float_t costhe = cos(kDegToRad*theta);
    typename CtTraits<ct>::float_t sinpsi = sin(kDegToRad*psi);
    typename CtTraits<ct>::float_t cospsi = cos(kDegToRad*psi);

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

  static inline __attribute__((always_inline))
  typename CtTraits<ct>::int_t RotationFootprint(
      typename CtTraits<ct>::float_t const *rot) {

    int footprint = 0;

    // Count zero-entries and give back a footprint that classifies them
    for (int i = 0; i < 9; ++i) {
      if(abs(rot[i]) < 1e-12) {
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

};

#endif /* LIBRARY_H */