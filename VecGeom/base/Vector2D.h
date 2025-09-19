/// \file Vector2D.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_VECTOR2D_H_
#define VECGEOM_BASE_VECTOR2D_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/backend/scalar/Backend.h"
#include "VecGeom/base/AlignedBase.h"

#include <algorithm>
#include <ostream>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(template <typename Type> class Vector2D;);
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, Vector2D, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename Type>
class Vector2D : public AlignedBase {

private:
  Type vec[2];

  typedef Vector2D<Type> VecType;

public:
  using value_type = Type;

  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Vector2D()
  {
    vec[0] = 0;
    vec[1] = 0;
  }

  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Vector2D(const Type x, const Type y)
  {
    vec[0] = x;
    vec[1] = y;
  }

  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Vector2D(const Type a)
  {
    vec[0] = a;
    vec[1] = a;
  }

  VECCORE_ATT_HOST_DEVICE
  Vector2D(Vector2D const &other)
  {
    vec[0] = other[0];
    vec[1] = other[1];
  }

  template <typename Real_i>
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Vector2D(const Vector2D<Real_i> &other)
  {
    vec[0] = static_cast<Type>(other.x());
    vec[1] = static_cast<Type>(other.y());
  }

  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE VecType operator=(VecType const &other)
  {
    vec[0] = other.vec[0];
    vec[1] = other.vec[1];
    return *this;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type &operator[](const int index) { return vec[index]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type operator[](const int index) const { return vec[index]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type &x() { return vec[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type x() const { return vec[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type &y() { return vec[1]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type y() const { return vec[1]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Set(const Type x, const Type y)
  {
    vec[0] = x;
    vec[1] = y;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Set(const Type a)
  {
    vec[0] = a;
    vec[1] = a;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector2D<Type> Abs() const { return Vector2D<Type>(vecCore::math::Abs(vec[0]), vecCore::math::Abs(vec[1])); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Min() const { return vecCore::math::Min(vec[0], vec[1]); };

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Max() const { return vecCore::math::Max(vec[0], vec[1]); };

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type CrossZ(VecType const &other) const { return vec[0] * other.vec[1] - vec[1] * other.vec[0]; }

  /// The dot product of two Vector2D<T> objects
  /// \return T (where T is float, double, or various SIMD vector types)
  template <typename Type2>
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE static Type Dot(Vector2D<Type> const &left, Vector2D<Type2> const &right)
  {
    return left[0] * right[0] + left[1] * right[1];
  }

  /// The dot product of two Vector2D<T> objects
  /// \return T (where T is float, double, or various SIMD vector types)
  template <typename Type2>
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Type Dot(Vector2D<Type2> const &right) const
  {
    return Dot(*this, right);
  }

  /// \return Squared magnitude of the vector.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Mag2() const { return Dot(*this, *this); }

  /// \return Magnitude of the vector.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Mag() const { return Sqrt(Mag2()); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Length2() const { return Mag2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Length() const { return Mag(); }

  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Vector2D<Type> Unit() const
  {
    return Type(1.) / NonZero(Mag()) * VecType(*this);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Normalize() { *this *= (Type(1.) / NonZero(Mag())); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector2D<Type> Normalized() const { return Unit(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsNormalized() const
  {
    Precision norm = Mag2();
    return Type(1.) - vecgeom::kTolerance < norm && norm < Type(1.) + vecgeom::kTolerance;
  }

  /// \return Azimuthal angle between -pi and pi.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Phi() const { return ATan2(vec[1], vec[0]); }

#define VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(OPERATOR) \
  VECCORE_ATT_HOST_DEVICE                             \
  VECGEOM_FORCE_INLINE                                \
  VecType &operator OPERATOR(const VecType & other)   \
  {                                                   \
    vec[0] OPERATOR other.vec[0];                     \
    vec[1] OPERATOR other.vec[1];                     \
    return *this;                                     \
  }                                                   \
  VECCORE_ATT_HOST_DEVICE                             \
  VECGEOM_FORCE_INLINE                                \
  VecType &operator OPERATOR(const Type & scalar)     \
  {                                                   \
    vec[0] OPERATOR scalar;                           \
    vec[1] OPERATOR scalar;                           \
    return *this;                                     \
  }
  VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(+=)
  VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(-=)
  VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(*=)
  VECTOR2D_TEMPLATE_INPLACE_BINARY_OP(/=)
#undef VECTOR2D_TEMPLATE_INPLACE_BINARY_OP
};

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool operator==(Vector2D<Type> const &lhs, Vector2D<Type> const &rhs)
{
  return lhs[0] == rhs[0] && lhs[1] == rhs[1];
}

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool operator!=(Vector2D<Type> const &lhs, Vector2D<Type> const &rhs)
{
  return !(lhs == rhs);
}

template <typename Type>
std::ostream &operator<<(std::ostream &os, Vector2D<Type> const &vec)
{
  os << "(" << vec[0] << ", " << vec[1] << ")";
  return os;
}

#define VECTOR2D_BINARY_OP(OPERATOR, INPLACE)                                                              \
  template <typename Type>                                                                                 \
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector2D<Type> operator OPERATOR(const Vector2D<Type> &lhs, \
                                                                                const Vector2D<Type> &rhs) \
  {                                                                                                        \
    Vector2D<Type> result(lhs);                                                                            \
    result INPLACE rhs;                                                                                    \
    return result;                                                                                         \
  }                                                                                                        \
  template <typename Type, typename ScalarType>                                                            \
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector2D<Type> operator OPERATOR(Vector2D<Type> const &lhs, \
                                                                                const ScalarType rhs)      \
  {                                                                                                        \
    Vector2D<Type> result(lhs);                                                                            \
    result INPLACE rhs;                                                                                    \
    return result;                                                                                         \
  }                                                                                                        \
  template <typename Type, typename ScalarType>                                                            \
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector2D<Type> operator OPERATOR(const ScalarType rhs,      \
                                                                                Vector2D<Type> const &lhs) \
  {                                                                                                        \
    Vector2D<Type> result(rhs);                                                                            \
    result INPLACE lhs;                                                                                    \
    return result;                                                                                         \
  }
VECTOR2D_BINARY_OP(+, +=)
VECTOR2D_BINARY_OP(-, -=)
VECTOR2D_BINARY_OP(*, *=)
VECTOR2D_BINARY_OP(/, /=)
#undef VECTOR2D_BINARY_OP

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_BASE_VECTOR2D_H_
