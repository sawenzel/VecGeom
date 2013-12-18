#ifndef LIBRARYVC_H
#define LIBRARYVC_H

#include "LibraryGeneric.h"

template <>
struct CtTraits<kVc> {
  typedef double       float_t;
  typedef Vc::int_v    int_v;
  typedef Vc::double_v float_v;
  typedef Vc::double_m bool_v;
};

typedef CtTraits<kVc>::int_v   VcInt;
typedef CtTraits<kVc>::float_v VcFloat;
typedef CtTraits<kVc>::bool_v  VcBool;
typedef CtTraits<kVc>::float_t VcScalar;

template <typename Type>
struct SOA3D<kVc, Type> {

private:

  int size_;
  Vc::Memory<Type> a, b, c;

public:

  SOA3D(const int size__) : size_(size__), a(size__), b(size__), c(size__) {}

  SOA3D(SOA3D const &other) : a(other.a), b(other.b), c(other.c) {}

  inline __attribute__((always_inline))
  int size() const { return size_; }

  inline __attribute__((always_inline))
  int vectors() const { return a.vectorsCount(); }

  inline __attribute__((always_inline))
  Vector3D<Type> operator[](const int index) const {
    return Vector3D<Type>(a.vector(index), b.vector(index), c.vector(index));
  }

  inline __attribute__((always_inline))
  Vc::Memory<Type>& Memory(const int index) {
    if (index == 0) return a;
    if (index == 1) return b;
    if (index == 2) return c;
    throw new std::out_of_range("");
  }

};

typedef SOA3D<kVc, VcFloat> SOA3D_Vc_Float;

#endif /* LIBRARYVC_H */