#ifndef LIBRARYGENERIC_H
#define LIBRARYGENERIC_H

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <cfloat>
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

const double kInfinity = DBL_MAX;
const int kAlignmentBoundary = 32;
const double kDegToRad = M_PI/180.;
const double kRadToDeg = 180./M_PI;

enum ImplType { kVc, kCuda, kScalar };

template <ImplType it>
struct ImplTraits {};

template <>
struct ImplTraits<kScalar> {
  typedef double float_t;
  typedef int    int_v;
  typedef double float_v;
  typedef bool   bool_v;
  #ifdef STD_CXX11
  constexpr static bool early_return = true;
  constexpr static float_v kZero = 0;
  constexpr static bool_v kFalse = false;
  #else
  const static bool early_return = true;
  const static float_v kZero = 0;
  const static bool_v kFalse = false;
  #endif /* STD_CXX11 */
};

template <typename Type>
struct Vector3D {

private:

  Type vec[3];

public:

  #ifdef STD_CXX11
  Vector3D() : vec{0, 0, 0} {};
  #else
  CUDA_HEADER_BOTH
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

template <typename Type1, typename Type2, typename Type3>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
Vector3D<Type3> operator+(Vector3D<Type1> const &lhs,
                          Vector3D<Type2> const &rhs) {
  return Vector3D<Type3>(lhs[0]+rhs[0], lhs[1]+rhs[1], lhs[2]+rhs[2]);
}

template <typename Type1, typename Type2, typename Type3>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
Vector3D<Type3> operator-(Vector3D<Type1> const &lhs,
                          Vector3D<Type2> const &rhs) {
  return Vector3D<Type3>(lhs[0]-rhs[0], lhs[1]-rhs[1], lhs[2]-rhs[2]);
}

template <typename Type1, typename Type2>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
Vector3D<Type1> operator-(Vector3D<Type1> const &lhs,
                          Vector3D<Type2> const &rhs) {
  return Vector3D<Type1>(lhs[0]-rhs[0], lhs[1]-rhs[1], lhs[2]-rhs[2]);
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
  Type& x(const int index) {
    return a[index];
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Type const& x(const int index) const {
    return a[index];
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Type& y(const int index) {
    return b[index];
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Type const& y(const int index) const {
    return b[index];
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Type& z(const int index) {
    return c[index];
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Type const& z(const int index) const {
    return c[index];
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  void Set(const int index, const Type a_, const Type b_, const Type c_) {
    a[index] = a_;
    b[index] = b_;
    c[index] = c_;
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  void Set(const int index, Vector3D<Type> const &vec) {
    a[index] = vec[0];
    b[index] = vec[1];
    c[index] = vec[2];
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

template <typename Float>
struct TransMatrix {

private:

  Float trans[3];
  Float rot[9];
  bool identity;
  bool has_rotation;
  bool has_translation;

public:

  CUDA_HEADER_BOTH
  TransMatrix() {
    SetTranslation(0, 0, 0);
    SetRotation(0, 0, 0);
  }

  CUDA_HEADER_BOTH
  TransMatrix(const Float tx, const Float ty, const Float tz,
              const Float phi, const Float theta,
              const Float psi) {
    SetTranslation(tx, ty, tz);
    SetRotation(phi, theta, psi);
  }

  template <typename other_t>
  CUDA_HEADER_BOTH
  TransMatrix(TransMatrix<other_t> const &other) {
    SetTranslation(Float(other.trans[0]), Float(other.trans[1]),
                   Float(other.trans[2]));
    SetRotation(Float(other.rot[0]), Float(other.rot[1]), Float(other.rot[2]), 
                Float(other.rot[3]), Float(other.rot[4]), Float(other.rot[5]),
                Float(other.rot[6]), Float(other.rot[7]), Float(other.rot[8]));

  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D<Float> Translation() const {
    return Vector3D<Float>(trans[0], trans[1], trans[2]);
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Float Translation(const int index) const { return trans[index]; }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  const Float* Rotation() const { return rot; }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Float Rotation(const int index) const { return rot[index]; }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  bool IsIdentity() const { return identity; }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  bool HasRotation() const { return has_rotation; }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  bool HasTranslation() const { return has_translation; }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  void SetTranslation(const Float tx, const Float ty,
                      const Float tz) {
    trans[0] = tx;
    trans[1] = ty;
    trans[2] = tz;
    SetProperties();
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  void SetProperties() {
    has_translation = (trans[0] || trans[1] || trans[2]) ? true : false;
    has_rotation = (RotationFootprint(rot) == 1296) ? false : true;
    identity = !has_translation && !has_rotation;
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  void SetRotation(const Float phi, const Float theta,
                   const Float psi) {

    const Float sinphi = sin(kDegToRad*phi);
    const Float cosphi = cos(kDegToRad*phi);
    const Float sinthe = sin(kDegToRad*theta);
    const Float costhe = cos(kDegToRad*theta);
    const Float sinpsi = sin(kDegToRad*psi);
    const Float cospsi = cos(kDegToRad*psi);

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

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  void SetRotation(const Float *rot_) {

    rot[0] = rot_[0];
    rot[1] = rot_[1];
    rot[2] = rot_[2];
    rot[3] = rot_[3];
    rot[4] = rot_[4];
    rot[5] = rot_[5];
    rot[6] = rot_[6];
    rot[7] = rot_[7];
    rot[8] = rot_[8];

    SetProperties();
  }

  CUDA_HEADER_BOTH
  static inline __attribute__((always_inline))
  int RotationFootprint(Float const *rot) {

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

  // Currently only very simple checks are performed.
  template <typename Type>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  void MasterToLocal(Vector3D<Type> const &master,
                     Vector3D<Type> &local) const {

    // Vc can do early returns here as they are only dependent on the matrix
    if (IsIdentity()) {
      local = master;
      return;
    }
    if (!HasRotation()) {
      local = master - Translation();
      return;
    }

    // General case
    const Vector3D<Type> t =
        master - Translation();
    local[0] =  t[0]*rot[0];
    local[1] =  t[0]*rot[1];
    local[2] =  t[0]*rot[2];
    local[0] += t[1]*rot[3];
    local[1] += t[1]*rot[4];
    local[2] += t[1]*rot[5];
    local[0] += t[2]*rot[6];
    local[1] += t[2]*rot[7];
    local[2] += t[2]*rot[8];

  }

  template <typename Type>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D<Type> MasterToLocal(Vector3D<Type> const &master) const {
    Vector3D<Type> local;
    MasterToLocal(master, local);
    return local;
  }

};

#endif /* LIBRARY_H */