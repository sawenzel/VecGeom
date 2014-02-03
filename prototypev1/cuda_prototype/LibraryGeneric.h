#ifndef LIBRARYGENERIC_H
#define LIBRARYGENERIC_H

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <cfloat>
#include "tbb/tick_count.h"
#include "mm_malloc.h"
#include <cmath>
#include <cassert>
#include <cstdlib>

#ifndef __CUDACC__
#define STD_CXX11
#define CUDA_HEADER_DEVICE
#define CUDA_HEADER_HOST
#define CUDA_HEADER_BOTH
#else
#define NVCC
#define CUDA_HEADER_DEVICE __device__
#define CUDA_HEADER_HOST __host__
#define CUDA_HEADER_BOTH __host__ __device__
#endif /* __CUDACC__ */

template <typename Type>
struct Vector3D;

template <typename Type>
struct SOA3D;

template <typename Float>
struct TransMatrix;

enum RotationType { kDiagonal = 720, kNone = 1296 };

const int kAlignmentBoundary = 32;
const double kDegToRad = M_PI/180.;
const double kRadToDeg = 180./M_PI;
const double kInfinity = INFINITY;
const double kTiny = 1e-20;
const double kGTolerance = 1e-9;

enum ImplType { kVc, kCuda, kScalar, kCilk };

template <ImplType it>
struct Impl {};

// Possibility to switch to doubles
typedef double CudaFloat;

template <>
struct Impl<kScalar> {
  typedef double float_t;
  typedef int    int_v;
  typedef double float_v;
  typedef bool   bool_v;
  #ifdef STD_CXX11
  constexpr static bool early_return = true;
  constexpr static float_v kZero = 0;
  constexpr static bool_v kTrue = true;
  constexpr static bool_v kFalse = false;
  #else
  const static bool early_return = true;
  const static float_v kZero = 0;
  const static bool_v kTrue = true;
  const static bool_v kFalse = false;
  #endif /* STD_CXX11 */
};

template <typename Type>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
Type CondAssign(const bool &cond,
                const Type &thenval, const Type &elseval) {
  return (cond) ? thenval : elseval;
}

template <typename Type1, typename Type2>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
void MaskedAssign(const bool &cond,
                  const Type1 &thenval, Type2 &output) {
  output = (cond) ? thenval : output;
}

template <ImplType it, typename Type>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
Type Abs(const Type&);

template <>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
double Abs<kScalar, double>(double const &val) {
  return std::fabs(val);
}

template <>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
float Abs<kScalar, float>(float const &val) {
  return std::fabs(val);
}

template <ImplType it, typename Type>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
Type Sqrt(const Type&);

template <>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
float Sqrt<kScalar, float>(float const &val) {
  return std::sqrt(val);
}

struct Stopwatch {
  tbb::tick_count t1;
  tbb::tick_count t2;
  void Start() { t1 = tbb::tick_count::now(); }
  double Stop() {
    t2 = tbb::tick_count::now();
    return Elapsed();
  }
  double Elapsed() const { return (t2-t1).seconds(); }
};

template <typename Type>
void PrintArray(Type const &arr, const int size) {
  #ifdef NVCC
  if (size <= 0) {
    printf("[]\n");
    return;
  }
  printf("[%f", arr[0]);
  for (int i = 1; i < size; ++i) printf(", %f", arr[i]);
  printf("]\n");
  #else
  if (size <= 0) {
    std::cout << "[]\n";
    return;
  }
  std::cout << "[" << arr[0];
  for (int i = 1; i < size; ++i) std::cout << ", " << arr[i];
  std::cout << "]";
  #endif
}

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

  CUDA_HEADER_HOST
  Vector3D(std::string const &str) {
    int begin = 1, end = str.find(",");
    vec[0] = std::atof(str.substr(begin, end-begin).c_str());
    begin = end + 2;
    end = str.find(",", begin);
    vec[1] = std::atof(str.substr(begin, end-begin).c_str());
    begin = end + 2;
    end = str.find(")", begin);
    vec[2] = std::atof(str.substr(begin, end-begin).c_str());
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Type& operator[](const int index) {
    return vec[index];
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Type const& operator[](const int index) const {
    return vec[index];
  }

  template <typename TypeOther>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D& operator+=(Vector3D<TypeOther> const &rhs) {
    this->vec[0] += rhs[0];
    this->vec[1] += rhs[1];
    this->vec[2] += rhs[2];
    return this;
  }

  template <typename TypeOther>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D& operator+=(TypeOther const &rhs) {
    this->vec[0] += rhs;
    this->vec[1] += rhs;
    this->vec[2] += rhs;
    return this;
  }

  template <typename TypeOther>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D& operator-=(Vector3D<TypeOther> const &rhs) {
    this->vec[0] -= rhs[0];
    this->vec[1] -= rhs[1];
    this->vec[2] -= rhs[2];
    return this;
  }

  template <typename TypeOther>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D& operator-=(TypeOther const &rhs) {
    this->vec[0] -= rhs;
    this->vec[1] -= rhs;
    this->vec[2] -= rhs;
    return this;
  }

  template <typename TypeOther>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D& operator*=(Vector3D<TypeOther> const &rhs) {
    this->vec[0] *= rhs[0];
    this->vec[1] *= rhs[1];
    this->vec[2] *= rhs[2];
    return this;
  }

  template <typename TypeOther>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D& operator*=(TypeOther const &rhs) {
    this->vec[0] *= rhs;
    this->vec[1] *= rhs;
    this->vec[2] *= rhs;
    return this;
  }

  template <typename TypeOther>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D& operator/=(Vector3D<TypeOther> const &rhs) {
    this->vec[0] /= rhs[0];
    this->vec[1] /= rhs[1];
    this->vec[2] /= rhs[2];
    return this;
  }

  template <typename TypeOther>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D& operator/=(TypeOther const &rhs) {
    const TypeOther inverse = TypeOther(1) / rhs;
    this->vec[0] *= inverse;
    this->vec[1] *= inverse;
    this->vec[2] *= inverse;
    return this;
  }

  #ifdef STD_CXX11

  friend inline __attribute__((always_inline))
  std::ostream& operator<<(std::ostream& os, Vector3D<Type> const &v) {
    os << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
    return os;
  }

  #endif /* STD_CXX11 */

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Type Length() const {
    return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  void Normalize() {
    const Type inverse = Type(1) / Length();
    vec[0] *= inverse;
    vec[1] *= inverse;
    vec[2] *= inverse;
  }

  template <ImplType it>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D<Type> Abs() const {
    return Vector3D<Type>(Abs<it>(vec[0]), Abs<it>(vec[1]), Abs<it>(vec[2]));
  }

};

template <typename Type1, typename Type2, typename Type3>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
Vector3D<Type3> operator+(Vector3D<Type1> const &lhs,
                          Vector3D<Type2> const &rhs) {
  return Vector3D<Type3>(lhs[0]+rhs[0], lhs[1]+rhs[1], lhs[2]+rhs[2]);
}

template <typename Type1, typename Type2>
CUDA_HEADER_BOTH
inline __attribute__((always_inline))
Vector3D<Type1> operator+(Vector3D<Type1> const &lhs,
                          Vector3D<Type2> const &rhs) {
  return Vector3D<Type1>(lhs[0]+rhs[0], lhs[1]+rhs[1], lhs[2]+rhs[2]);
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

  template <typename OtherType>
  CUDA_HEADER_BOTH
  TransMatrix(TransMatrix<OtherType> const &other) {
    SetTranslation(Float(other.Translation(0)), Float(other.Translation(1)),
                   Float(other.Translation(2)));
    SetRotation(Float(other.Rotation(0)), Float(other.Rotation(1)),
                Float(other.Rotation(2)), Float(other.Rotation(3)),
                Float(other.Rotation(4)), Float(other.Rotation(5)),
                Float(other.Rotation(6)), Float(other.Rotation(7)),
                Float(other.Rotation(8)));

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
  Float const* Rotation() const { return rot; }

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
  void SetTranslation(Vector3D<Float> const &vec) {
    SetTranslation(vec[0], vec[1], vec[2]);
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
  void SetRotation(Vector3D<Float> const &vec) {
    SetRotation(vec[0], vec[1], vec[2]);
  }

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  void SetRotation(const Float rot0, const Float rot1, const Float rot2,
                   const Float rot3, const Float rot4, const Float rot5,
                   const Float rot6, const Float rot7, const Float rot8) {

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

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  void SetRotation(const Float *rot_) {
    SetRotation(rot_[0], rot_[1], rot_[2], rot_[3], rot_[4], rot_[5],
                rot_[6], rot_[7], rot_[8]);
  }

  CUDA_HEADER_BOTH
  static inline __attribute__((always_inline))
  int RotationFootprint(Float const *rot) {

    int footprint = 0;

    // Count zero-entries and give back a footprint that classifies them
    for (int i = 0; i < 9; ++i) {
      if (Abs<kScalar>(rot[i]) < 1e-12) {
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

private:

  template <typename Type>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  void DoRotation(Vector3D<Type> const &master, Vector3D<Type> &local) const {
    local[0] =  master[0]*rot[0];
    local[1] =  master[0]*rot[1];
    local[2] =  master[0]*rot[2];
    local[0] += master[1]*rot[3];
    local[1] += master[1]*rot[4];
    local[2] += master[1]*rot[5];
    local[0] += master[2]*rot[6];
    local[1] += master[2]*rot[7];
    local[2] += master[2]*rot[8];
  }

public:

  // Currently only very simple checks are performed.
  template <typename Type>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  void Transform(Vector3D<Type> const &master,
                 Vector3D<Type> &local) const {

    // Vc can do early returns here as they are only dependent on the matrix
    if (IsIdentity()) {
      local = master;
      return;
    }

    local = master - Translation();

    if (!HasRotation()) return;

    // General case
    DoRotation(master, local);

  }

  template <typename Type>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D<Type> Transform(Vector3D<Type> const &master) const {
    Vector3D<Type> local;
    Transform(master, local);
    return local;
  }

  template <typename Type>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  void TransformRotation(Vector3D<Type> const &master,
                         Vector3D<Type> &local) const {

    // Vc can do early returns here as they are only dependent on the matrix
    if (!HasRotation()) {
      local = master;
      return;
    }

    DoRotation(master, local);

  }

  template <typename Type>
  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  Vector3D<Type> TransformRotation(Vector3D<Type> const &master) const {
    Vector3D<Type> local;
    TransformRotation(master, local);
    return local;
  }

};


#endif /* LIBRARY_H */