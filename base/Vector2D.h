/// \file Vector2D.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_VECTOR2D_H_
#define VECGEOM_BASE_VECTOR2D_H_

#include "base/Global.h"

#include "backend/scalar/Backend.h"
#include "base/AlignedBase.h"

#ifdef VECGEOM_VC_ACCELERATION
#include <Vc/Vc>
#endif
#include <algorithm>
#include <ostream>

namespace VECGEOM_NAMESPACE {

template <typename Type>
class Vector2D : public AlignedBase {

private:

#ifndef VECGEOM_VC_ACCELERATION
  Type vec[2];
#else
  Vc::Memory<Vc::Vector<Type>, 2> vec;
#endif

  typedef Vector2D<Type> VecType;

public:

  VECGEOM_CUDA_HEADER_BOTH
  Vector2D();

  VECGEOM_CUDA_HEADER_BOTH
  Vector2D(const Type x, const Type y);

  VECGEOM_CUDA_HEADER_BOTH
  Vector2D(Vector2D const &other);

  VECGEOM_CUDA_HEADER_BOTH
  VecType operator=(VecType const &other);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& operator[](const int index);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type operator[](const int index) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& x();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type x() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& y();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type y() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Set(const Type x, const Type y);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Cross(VecType const &other) const {
    return vec[0]*other.vec[1] - vec[1]*other.vec[0];
  }

  template <typename StreamType>
  friend inline std::ostream& operator<<(std::ostream &os,
                                         Vector2D<StreamType> const &v);

  #define VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(OPERATOR) \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType& operator OPERATOR(const VecType &other) { \
    vec[0] OPERATOR other.vec[0]; \
    vec[1] OPERATOR other.vec[1]; \
    return *this; \
  } \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType& operator OPERATOR(const Type &scalar) { \
    vec[0] OPERATOR scalar; \
    vec[1] OPERATOR scalar; \
    return *this; \
  }
  VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(+=)
  VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(-=)
  VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(*=)
  VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(/=)
  #undef VECTOR2D_TEMPLATE_INPLACE_BINARY_OP

};

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
Vector2D<Type>::Vector2D() {
  vec[0] = 0;
  vec[1] = 0;
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
Vector2D<Type>::Vector2D(const Type x, const Type y) {
  vec[0] = x;
  vec[1] = y;
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
Vector2D<Type>::Vector2D(Vector2D const &other) {
  vec[0] = other.vec[0];
  vec[1] = other.vec[1];
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
Vector2D<Type> Vector2D<Type>::operator=(Vector2D<Type> const &other) {
  vec[0] = other.vec[0];
  vec[1] = other.vec[1];
  return *this;
}

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
Type& Vector2D<Type>::operator[](const int index) { return vec[index]; }

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
Type Vector2D<Type>::operator[](const int index) const { return vec[index]; }

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
Type& Vector2D<Type>::x() { return vec[0]; }

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
Type Vector2D<Type>::x() const { return vec[0]; }

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
Type& Vector2D<Type>::y() { return vec[1]; }

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
Type Vector2D<Type>::y() const { return vec[1]; }

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
void Vector2D<Type>::Set(const Type x, const Type y) {
  vec[0] = x;
  vec[1] = y;
}

template <typename Type>
std::ostream& operator<<(std::ostream &os, Vector2D<Type> const &v) {
  os << "(" << v.vec[0] << ", " << v.vec[1] << ")";
  return os;
}

#define VECTOR2D_BINARY_OP(OPERATOR, INPLACE) \
template <typename Type> \
VECGEOM_INLINE \
VECGEOM_CUDA_HEADER_BOTH \
Vector2D<Type> operator OPERATOR(const Vector2D<Type> &lhs, \
                                 const Vector2D<Type> &rhs) { \
  Vector2D<Type> result(lhs); \
  result INPLACE rhs; \
  return result; \
} \
template <typename Type, typename ScalarType> \
VECGEOM_INLINE \
VECGEOM_CUDA_HEADER_BOTH \
Vector2D<Type> operator OPERATOR(Vector2D<Type> const &lhs, \
                                 const ScalarType rhs) { \
  Vector2D<Type> result(lhs); \
  result INPLACE rhs; \
  return result; \
} \
template <typename Type, typename ScalarType> \
VECGEOM_INLINE \
VECGEOM_CUDA_HEADER_BOTH \
Vector2D<Type> operator OPERATOR(const ScalarType rhs, \
                                 Vector2D<Type> const &lhs) { \
  Vector2D<Type> result(rhs); \
  result INPLACE lhs; \
  return result; \
}
VECTOR2D_BINARY_OP(+, +=)
VECTOR2D_BINARY_OP(-, -=)
VECTOR2D_BINARY_OP(*, *=)
VECTOR2D_BINARY_OP(/, /=)
#undef VECTOR2D_BINARY_OP

} // End global namespace

#endif // VECGEOM_BASE_VECTOR2D_H_