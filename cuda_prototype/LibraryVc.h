#ifndef LIBRARYVC_H
#define LIBRARYVC_H

#include <Vc/Vc>
#include "LibraryGeneric.h"

template <>
struct ImplTraits<kVc> {
  typedef double       float_t;
  typedef Vc::int_v    int_v;
  typedef Vc::double_v float_v;
  typedef Vc::double_m bool_v;
  const static bool early_return = false;
  const static float_v kZero;
  const static bool_v kTrue;
  const static bool_v kFalse;
};

const int kVectorSize = ImplTraits<kVc>::float_v::Size;

typedef ImplTraits<kVc>::int_v   VcInt;
typedef ImplTraits<kVc>::float_v VcFloat;
typedef ImplTraits<kVc>::bool_v  VcBool;
typedef ImplTraits<kVc>::float_t VcScalarFloat;

template <typename Type>
struct VcSOA3D {

private:

  int size_;
  Vc::Memory<Type> a, b, c;

public:

  VcSOA3D(const int size__) : size_(size__), a(size__), b(size__), c(size__) {}

  VcSOA3D(VcSOA3D const &other) : a(other.a), b(other.b), c(other.c) {}

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

typedef VcSOA3D<VcFloat> VcSOA3D_Float;

template <typename Type>
inline __attribute__((always_inline))
Type CondAssign(const VcBool &cond,
                const Type &thenval, const Type &elseval) {
  Type ret;
  ret(cond) = thenval;
  ret(!cond) = elseval;
  return ret;
}

template <typename Type1, typename Type2>
inline __attribute__((always_inline))
void MaskedAssign(const VcBool &cond,
                  const Type1 &thenval, Type2 &output) {
  output(cond) = thenval;
}

#endif /* LIBRARYVC_H */