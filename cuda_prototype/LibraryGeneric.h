#ifndef LIBRARYGENERIC_H
#define LIBRARYGENERIC_H

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
  double Elapsed() { return (t2-t1).seconds(); }
};

#endif /* LIBRARY_H */