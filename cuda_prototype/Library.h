#ifndef LIBRARY_H
#define LIBRARY_H

#include <stdexcept>
#include "mm_malloc.h"

#ifndef __CUDACC__
#include <Vc/Vc>
#endif /* __CUDACC__ */

#define ALIGNMENT_BOUNDARY 32

enum Ct { kVc, kCuda };

template <Ct ct>
struct CtTraits {};

template <Ct ct>
struct Vector3D {
private:
  typename CtTraits<ct>::float_t vec[3];
public:
  #ifndef __CUDACC__
  Vector3D() : vec{0, 0, 0} {};
  #endif /* __CUDACC__ */
  Vector3D(const typename CtTraits<ct>::float_t a,
           const typename CtTraits<ct>::float_t b,
           const typename CtTraits<ct>::float_t c) {
    vec[0] = a;
    vec[1] = b;
    vec[2] = c;
  }
  Vector3D(Vector3D const &other) {
    vec[0] = other[0];
    vec[1] = other[1];
    vec[2] = other[2];
  }
  typename CtTraits<ct>::float_t& operator[](const int index) {
    return vec[index];
  }
  typename CtTraits<ct>::float_t operator[](const int index) const {
    return vec[index];
  }
};

template <Ct ct>
struct SOA3D {};

#ifndef __CUDACC__

template <>
struct CtTraits<kVc> {
  typedef Vc::int_v    int_t;
  typedef Vc::double_v float_t;
  typedef Vc::ushort_v bool_t;
};

template <>
struct SOA3D<kVc> {
private:
  int size_;
  Vc::Memory<CtTraits<kVc>::float_t> a, b, c;
public:
  SOA3D(const int size__) : size_(size__), a(size__), b(size__), c(size__) {}
  SOA3D(SOA3D const &other) : a(other.a), b(other.b), c(other.c) {}
  inline int size() const { return size_; }
  inline int vectors() const { return a.vectorsCount(); }
  Vector3D<kVc> operator[](const int index) const {
    return Vector3D<kVc>(a[index], b[index], c[index]);
  }
  Vc::Memory<CtTraits<kVc>::float_t>& operator[](const int index) {
    if (index == 0) return a;
    if (index == 1) return b;
    if (index == 2) return c;
    throw new std::out_of_range("");
  }
};

#endif /* __CUDACC__ */

template <>
struct CtTraits<kCuda> {
  typedef int   int_t;
  typedef float float_t;
  typedef bool  bool_t;
};

template <>
struct SOA3D<kCuda> {
private:
  int size_;
  CtTraits<kCuda>::float_t *a, *b, *c;
public:
  #ifndef __CUDACC__
  SOA3D(const int size__) : size_(size__) {
    a = (CtTraits<kCuda>::float_t*)
        _mm_malloc(sizeof(CtTraits<kCuda>::float_t)*size_, ALIGNMENT_BOUNDARY);
    b = (CtTraits<kCuda>::float_t*)
        _mm_malloc(sizeof(CtTraits<kCuda>::float_t)*size_, ALIGNMENT_BOUNDARY);
    c = (CtTraits<kCuda>::float_t*)
        _mm_malloc(sizeof(CtTraits<kCuda>::float_t)*size_, ALIGNMENT_BOUNDARY);
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
  ~SOA3D() {
    _mm_free(a);
    _mm_free(b);
    _mm_free(c);
  }
  #endif /* __CUDACC__ */
  inline int size() const { return size_; }
  Vector3D<kCuda> operator[](const int index) const {
    return Vector3D<kCuda>(a[index], b[index], c[index]);
  }
  CtTraits<kCuda>::float_t* operator[](const int index) {
    if (index == 0) return a;
    if (index == 1) return b;
    if (index == 2) return c;
    throw new std::out_of_range("");
  }
};

#endif