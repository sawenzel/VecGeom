#ifndef VECGEOM_VECTOR3D_H_
#define VECGEOM_VECTOR3D_H_

#include "Global.h"

template <typename Type>
class Vector3D {

  typedef Vector3D<Type> VecType;

private:

  Type vec[3];

public:

  Vector3D(const Type a, const Type b, const Type c) {
    vec[0] = a;
    vec[1] = b;
    vec[2] = c;
  }

  Vector3D() {
    Vector3D<Type>(Type(0.), Type(0.), Type(0.));
  }

  Vector3D(const Type a) {
    Vector3D<Type>(a, a, a);
  }

  Vector3D(Vector3D const &other) {
    vec[0] = other[0];
    vec[1] = other[1];
    vec[2] = other[2];
  }

  Type& operator[](const int index) {
    return vec[index];
  }

  Type const& operator[](const int index) const {
    return vec[index];
  }

  Type& x() { return vec[0]; }

  Type const& x() const { return vec[0]; }

  Type& y() { return vec[1]; }

  Type const& y() const { return vec[1]; }

  Type& z() { return vec[2]; }
  Type const& z() const { return vec[2]; }

  // Inplace binary operators

  #define INPLACE_BINARY_OP(OPERATOR) \
  VecType& operator OPERATOR(const VecType &other) { \
    this->vec[0] OPERATOR other.vec[0]; \
    this->vec[1] OPERATOR other.vec[1]; \
    this->vec[2] OPERATOR other.vec[2]; \
    return *this; \
  } \
  VecType& operator OPERATOR(const Type &scalar) { \
    this->vec[0] OPERATOR scalar; \
    this->vec[1] OPERATOR scalar; \
    this->vec[2] OPERATOR scalar; \
    return *this; \
  }
  INPLACE_BINARY_OP(+=)
  INPLACE_BINARY_OP(-=)
  INPLACE_BINARY_OP(*=)
  INPLACE_BINARY_OP(/=)
  #undef INPLACE_BINARY_OP

  // Binary operators

  #define BINARY_OP(OPERATOR, INPLACE) \
  VecType operator OPERATOR(const VecType &other) const { \
    VecType result(*this); \
    result INPLACE other; \
    return result; \
  } \
  VecType operator OPERATOR(const Type &scalar) const { \
    VecType result(*this); \
    result INPLACE scalar; \
    return result; \
  }
  BINARY_OP(+, +=)
  BINARY_OP(-, -=)
  BINARY_OP(*, *=)
  BINARY_OP(/, /=)
  #undef BINARY_OP

};

#endif // VECGEOM_VECTOR3D_H_