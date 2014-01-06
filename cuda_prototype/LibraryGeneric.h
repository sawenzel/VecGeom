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

enum ImplType { kVc, kCuda, kScalar };

template <ImplType it>
struct ImplTraits {};

template <typename Type>
struct Vector3D {

private:

  Type vec[3];

public:

  #ifdef STD_CXX11
  Vector3D() : vec{0, 0, 0} {};
  #else
  Vector3D() {
    vec[0] = 0;
    vec[1] = 0;
    vec[2] = 0;
  }
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

  CUDA_HEADER_HOST
  Vector3D<float> ToFloatHost() const;

  CUDA_HEADER_DEVICE
  Vector3D<float> ToFloatDevice() const;

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


template <typename Type>
struct SOA3D {

private:

  int size_;
  Type *a, *b, *c;

public:

  #ifdef STD_CXX11

  SOA3D() : a(nullptr), b(nullptr), c(nullptr), size_(0) {}

  SOA3D(const int size__) : size_(size__) {
    a = (Type*)
        _mm_malloc(sizeof(Type)*size_, kAlignmentBoundary);
    b = (Type*)
        _mm_malloc(sizeof(Type)*size_, kAlignmentBoundary);
    c = (Type*)
        _mm_malloc(sizeof(Type)*size_, kAlignmentBoundary);
  }

  SOA3D(SOA3D const &other) : SOA3D(other.size_) {
    const int count = other.size_;
    for (int i = 0; i < count; ++i) {
      a[i] = other.a[i];
      b[i] = other.b[i];
      c[i] = other.c[i];
    }
    size_ = count;
  }

  #else

  SOA3D() {
    a = NULL;
    b = NULL;
    c = NULL;
    size_ = 0;
  }

  SOA3D(const int size__) {
    size_ = size__;
    a = (Type*)
        _mm_malloc(sizeof(Type)*size_, kAlignmentBoundary);
    b = (Type*)
        _mm_malloc(sizeof(Type)*size_, kAlignmentBoundary);
    c = (Type*)
        _mm_malloc(sizeof(Type)*size_, kAlignmentBoundary);
  }

  #endif /* STD_CXX11 */

  #ifndef NVCC

  ~SOA3D() {
    Deallocate();
  }

  #endif /* NVCC */

  SOA3D(Type *const a_, Type *const b_, Type *const c_, const int size__) {
    a = a_;
    b = b_;
    c = c_;
    size_ = size__;
  }

  inline __attribute__((always_inline))
  void Deallocate() {
    if (a) _mm_free(a);
    if (b) _mm_free(b);
    if (c) _mm_free(c);
  }

  #ifdef NVCC

  inline __attribute__((always_inline))
  SOA3D<Type> CopyToGPU() const {
    const int count = size();
    const int memsize = count*sizeof(Type);
    Type *a_, *b_, *c_;
    cudaMalloc((void**)&a_, memsize);
    cudaMalloc((void**)&b_, memsize);
    cudaMalloc((void**)&c_, memsize);
    cudaMemcpy(a_, a, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(b_, b, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(c_, c, memsize, cudaMemcpyHostToDevice);
    return SOA3D<Type>(a_, b_, c_, count);
  }

  inline __attribute__((always_inline))
  void FreeFromGPU() {
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
  }

  #endif /* NVCC */

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  int size() const { return size_; }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D<Type> operator[](const int index) const {
    return Vector3D<Type>(a[index], b[index], c[index]);
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Type* Memory(const int index) {
    if (index == 0) return a;
    if (index == 1) return b;
    if (index == 2) return c;
    #ifndef NVCC
    throw new std::out_of_range("");
    #else
    return NULL;
    #endif /* NVCC */
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Type const* Memory(const int index) const {
    if (index == 0) return a;
    if (index == 1) return b;
    if (index == 2) return c;
    #ifndef NVCC
    throw new std::out_of_range("");
    #else
    return NULL;
    #endif /* NVCC */
  }

};

struct Stopwatch {
  tbb::tick_count t1;
  tbb::tick_count t2;
  void Start() { t1 = tbb::tick_count::now(); }
  void Stop() { t2 = tbb::tick_count::now(); }
  double Elapsed() const { return (t2-t1).seconds(); }
};

// Solve with overloading instead
// template<ImplType it, typename Type>
// CUDA_HEADER_BOTH
// inline __attribute__((always_inline))
// Type CondAssign(const typename ImplTraits<it>::bool_v &cond,
//                 const Type &thenval, const Type &elseval);

enum RotationType { kDiagonal = 720, kNone = 1296 };

struct TransMatrix {

private:

  double trans[3];
  double rot[9];
  bool identity;
  bool has_rotation;
  bool has_translation;

public:

  TransMatrix() {
    SetTranslation(0, 0, 0);
    SetRotation(0, 0, 0);
  }

  TransMatrix(const double tx, const double ty, const double tz,
              const double phi, const double theta,
              const double psi) {
    SetTranslation(tx, ty, tz);
    SetRotation(phi, theta, psi);
  }

  inline __attribute__((always_inline))
  Vector3D<double> Translation() const {
    return Vector3D<double>(trans[0], trans[1], trans[2]);
  }

  inline __attribute__((always_inline))
  double Translation(const int index) const { return trans[index]; }

  inline __attribute__((always_inline))
  const double* Rotation() const { return rot; }

  inline __attribute__((always_inline))
  double Rotation(const int index) const { return rot[index]; }

  inline __attribute__((always_inline))
  bool IsIdentity() const { return identity; }

  inline __attribute__((always_inline))
  bool HasRotation() const { return has_rotation; }

  inline __attribute__((always_inline))
  bool HasTranslation() const { return has_translation; }

  inline __attribute__((always_inline))
  void SetTranslation(const double tx, const double ty,
                      const double tz) {
    trans[0] = tx;
    trans[1] = ty;
    trans[2] = tz;
    SetProperties();
  }

  inline __attribute__((always_inline))
  void SetProperties() {
    has_translation = (trans[0] || trans[1] || trans[2]) ? true : false;
    has_rotation = (RotationFootprint(rot) == 1296) ? false : true;
    identity = !has_translation && !has_rotation;
  }

  inline __attribute__((always_inline))
  void SetRotation(const double phi, const double theta,
                   const double psi) {

    const double sinphi = sin(kDegToRad*phi);
    const double cosphi = cos(kDegToRad*phi);
    const double sinthe = sin(kDegToRad*theta);
    const double costhe = cos(kDegToRad*theta);
    const double sinpsi = sin(kDegToRad*psi);
    const double cospsi = cos(kDegToRad*psi);

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
  int RotationFootprint(double const *rot) {

    int footprint = 0;

    // Count zero-entries and give back a footprint that classifies them
    for (int i = 0; i < 9; ++i) {
      if (abs(rot[i]) < 1e-12) {
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