#ifndef VECGEOM_VECTOR3D_H_

#include "Global.h"

template <typename Type>
class Vector3D {

  typedef Vector3D<Type> VecType;

private:

  Type vec[3];

public:

  inline
  Vector3D(const Type a, const Type b, const Type c) {
    vec[0] = a;
    vec[1] = b;
    vec[2] = c;
  }

  inline
  Vector3D() {
    Vector3D<Type>(Type(0.), Type(0.), Type(0.));
  }

  inline
  Vector3D(const Type a) {
    Vector3D<Type>(a, a, a);
  }

  inline
  Vector3D(Vector3D const &other) {
    vec[0] = other[0];
    vec[1] = other[1];
    vec[2] = other[2];
  }

  /**
   * Contains no check for correct indexing to avoid impairing performance.
   * @param index Index of content in the range [0-2].
   */
  inline
  Type& operator[](const int index) {
    return vec[index];
  }

  /**
   * Contains no check for correct indexing to avoid impairing performance.
   * @param index Index of content in the range [0-2].
   */
  inline
  Type const& operator[](const int index) const {
    return vec[index];
  }

  inline
  Type& x() { return vec[0]; }

  inline
  Type const& x() const { return vec[0]; }

  inline
  Type& y() { return vec[1]; }

  inline
  Type const& y() const { return vec[1]; }

  inline
  Type& z() { return vec[2]; }

  inline
  Type const& z() const { return vec[2]; }

  // Inplace binary operators

  #define INPLACE_BINARY_OP(OPERATOR) \
  inline \
  VecType& operator OPERATOR(const VecType &other) { \
    this->vec[0] OPERATOR other.vec[0]; \
    this->vec[1] OPERATOR other.vec[1]; \
    this->vec[2] OPERATOR other.vec[2]; \
    return *this; \
  } \
  inline \
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
  inline \
  VecType operator OPERATOR(const VecType &other) const { \
    VecType result(*this); \
    result INPLACE other; \
    return result; \
  } \
  inline \
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