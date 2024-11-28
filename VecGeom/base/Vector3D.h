/// \file vector3d.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_VECTOR3D_H_
#define VECGEOM_BASE_VECTOR3D_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"

#include <cstdlib>
#include <ostream>
#include <string>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(template <typename Type> class Vector3D;);
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, Vector3D, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Three dimensional vector class supporting most arithmetic operations.
 * @details If vector acceleration is enabled, the scalar template instantiation
 *          will use vector instructions for operations when possible.
 */
template <typename Type>
class Vector3D : public AlignedBase {

  typedef Vector3D<Type> VecType;

private:
  Type vec[3];

public:
  using value_type = Type;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D(const Type a, const Type b, const Type c)
  {
    vec[0] = a;
    vec[1] = b;
    vec[2] = c;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D()
  {
    vec[0] = 0;
    vec[1] = 0;
    vec[2] = 0;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D(const Type a)
  {
    vec[0] = a;
    vec[1] = a;
    vec[2] = a;
  }

  template <typename TypeOther>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D(Vector3D<TypeOther> const &other)
  {
    vec[0] = other[0];
    vec[1] = other[1];
    vec[2] = other[2];
  }

  /**
   * Constructs a vector from an std::string of the same format as output by the
   * "<<"-operator for outstreams.
   * @param str String formatted as "(%d, %d, %d)".
   */
  VECCORE_ATT_HOST
  Vector3D(std::string const &str)
  {
    int begin = 1, end = str.find(",");
    vec[0] = std::atof(str.substr(begin, end - begin).c_str());
    begin  = end + 2;
    end    = str.find(",", begin);
    vec[1] = std::atof(str.substr(begin, end - begin).c_str());
    begin  = end + 2;
    end    = str.find(")", begin);
    vec[2] = std::atof(str.substr(begin, end - begin).c_str());
  }

  /**
   * Contains no check for correct indexing to avoid impairing performance.
   * @param index Index of content in the range [0-2].
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type &operator[](const int index) { return vec[index]; }

  /**
   * Contains no check for correct indexing to avoid impairing performance.
   * @param index Index of content in the range [0-2].
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type const &operator[](const int index) const { return vec[index]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type &x() { return vec[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type const &x() const { return vec[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type &y() { return vec[1]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type const &y() const { return vec[1]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type &z() { return vec[2]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type const &z() const { return vec[2]; }

  VECCORE_ATT_HOST_DEVICE
  void Set(Type const &a, Type const &b, Type const &c)
  {
    vec[0] = a;
    vec[1] = b;
    vec[2] = c;
  }

  VECCORE_ATT_HOST_DEVICE
  void Set(const Type a) { Set(a, a, a); }

  /// \return the length squared perpendicular to z direction
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Perp2() const { return vec[0] * vec[0] + vec[1] * vec[1]; }

  /// \return the length perpendicular to z direction
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Perp() const { return Sqrt(Perp2()); }

  /// The dot product of two Vector3D<T> objects
  /// \return T (where T is float, double, or various SIMD vector types)
  template <typename Type2>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Type Dot(Vector3D<Type> const &left, Vector3D<Type2> const &right)
  {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
  }

  /// The dot product of two Vector3D<T> objects
  /// \return T (where T is float, double, or various SIMD vector types)
  template <typename Type2>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Type Dot(Vector3D<Type2> const &right) const
  {
    return Dot(*this, right);
  }

  // For UVector3 compatibility. It is equal to normal multiplication.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VecType MultiplyByComponents(VecType const &other) const { return *this * other; }

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
  Type Length() const { return Mag(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Length2() const { return Mag2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VecType Unit() const { return Type(1.) / Mag() * VecType(*this); }

  /// Normalizes the vector by dividing each entry by the length.
  /// \sa Vector3D::Length()
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Normalize() { *this *= (Type(1.) / Length()); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VecType Normalized() const { return Unit(); }

  // checks if vector is normalized
  // only reasonable to call with standard scalare usage
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsNormalized() const
  {
    // static_assert here that Type should be primitive type
    Precision norm = Mag2();
    return Type(1.) - vecgeom::kTolerance < norm && norm < Type(1.) + vecgeom::kTolerance;
  }

  /// \return Azimuthal angle between -pi and pi.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Phi() const { return ATan2(vec[1], vec[0]); }

  /// \return Polar angle between 0 and pi.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Theta() const { return ACos(vec[2] / Mag()); }

  /// The cross (vector) product of two Vector3D<T> objects
  /// \return Type (where Type is float, double, or various SIMD vector types)
  template <class FirstType, class SecondType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Type> Cross(Vector3D<FirstType> const &left,
                                                                           Vector3D<SecondType> const &right)
  {
    return Vector3D<Type>(left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2],
                          left[0] * right[1] - left[1] * right[0]);
  }

  /// The cross (vector) product of two Vector3D<T> objects
  /// \return Type (where Type is float, double, or various SIMD vector types)
  template <class OtherType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Type> Cross(Vector3D<OtherType> const &right) const
  {
    return Cross<Type, OtherType>(*this, right);
  }

  /// Maps each vector entry to a function that manipulates the entry type.
  /// \param f A function of type "Type f(const Type&)" to map over entries.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Map(Type (*f)(const Type &))
  {
    vec[0] = f(vec[0]);
    vec[1] = f(vec[1]);
    vec[2] = f(vec[2]);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VecType Abs() const
  {
    return VecType(vecCore::math::Abs(vec[0]), vecCore::math::Abs(vec[1]), vecCore::math::Abs(vec[2]));
  }

  template <typename BoolType>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void MaskedAssign(Vector3D<BoolType> const &condition,
                                                                 Vector3D<Type> const &value)
  {
    vec[0] = (condition[0]) ? value[0] : vec[0];
    vec[1] = (condition[1]) ? value[1] : vec[1];
    vec[2] = (condition[2]) ? value[2] : vec[2];
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Min() const { return vecCore::math::Min(vec[0], vec[1], vec[2]); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Type Max() const { return vecCore::math::Max(vec[0], vec[1], vec[2]); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static VecType FromCylindrical(Type r, Type phi, Type z) { return VecType(r * cos(phi), r * sin(phi), z); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VecType &FixZeroes()
  {
    using vecCore::math::Abs;
    for (int i = 0; i < 3; ++i) {
      vecCore::MaskedAssign(vec[i], Abs(vec[i]) < kTolerance, Type(0.0));
    }
    return *this;
  }

  // Inplace binary operators

#define VECTOR3D_TEMPLATE_INPLACE_BINARY_OP(OPERATOR)                                                       \
  VECCORE_ATT_HOST_DEVICE                                                                                   \
  VECGEOM_FORCE_INLINE                                                                                      \
  VecType &operator OPERATOR(const VecType &other)                                                          \
  {                                                                                                         \
    vec[0] OPERATOR other.vec[0];                                                                           \
    vec[1] OPERATOR other.vec[1];                                                                           \
    vec[2] OPERATOR other.vec[2];                                                                           \
    return *this;                                                                                           \
  }                                                                                                         \
  template <typename OtherType>                                                                             \
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE VecType &operator OPERATOR(const Vector3D<OtherType> &other) \
  {                                                                                                         \
    vec[0] OPERATOR other[0];                                                                               \
    vec[1] OPERATOR other[1];                                                                               \
    vec[2] OPERATOR other[2];                                                                               \
    return *this;                                                                                           \
  }                                                                                                         \
  VECCORE_ATT_HOST_DEVICE                                                                                   \
  VECGEOM_FORCE_INLINE                                                                                      \
  VecType &operator OPERATOR(const Type &scalar)                                                            \
  {                                                                                                         \
    vec[0] OPERATOR scalar;                                                                                 \
    vec[1] OPERATOR scalar;                                                                                 \
    vec[2] OPERATOR scalar;                                                                                 \
    return *this;                                                                                           \
  }
  VECTOR3D_TEMPLATE_INPLACE_BINARY_OP(+=)
  VECTOR3D_TEMPLATE_INPLACE_BINARY_OP(-=)
  VECTOR3D_TEMPLATE_INPLACE_BINARY_OP(*=)
  VECTOR3D_TEMPLATE_INPLACE_BINARY_OP(/=)
#undef VECTOR3D_TEMPLATE_INPLACE_BINARY_OP

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  operator bool() const { return vec[0] && vec[1] && vec[2]; }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, Vector3D<T> const &vec)
{
  os << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")";
  return os;
}

#define VECTOR3D_BINARY_OP(OPERATOR, INPLACE)                                                                   \
  template <typename Type, typename OtherType>                                                                  \
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Type> operator OPERATOR(const Vector3D<Type> &lhs,      \
                                                                                const Vector3D<OtherType> &rhs) \
  {                                                                                                             \
    Vector3D<Type> result(lhs);                                                                                 \
    result INPLACE rhs;                                                                                         \
    return result;                                                                                              \
  }                                                                                                             \
  template <typename Type, typename ScalarType>                                                                 \
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Type> operator OPERATOR(Vector3D<Type> const &lhs,      \
                                                                                const ScalarType rhs)           \
  {                                                                                                             \
    Vector3D<Type> result(lhs);                                                                                 \
    result INPLACE rhs;                                                                                         \
    return result;                                                                                              \
  }                                                                                                             \
  template <typename Type, typename ScalarType>                                                                 \
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Type> operator OPERATOR(const ScalarType lhs,           \
                                                                                Vector3D<Type> const &rhs)      \
  {                                                                                                             \
    Vector3D<Type> result(lhs);                                                                                 \
    result INPLACE rhs;                                                                                         \
    return result;                                                                                              \
  }
VECTOR3D_BINARY_OP(+, +=)
VECTOR3D_BINARY_OP(-, -=)
VECTOR3D_BINARY_OP(*, *=)
VECTOR3D_BINARY_OP(/, /=)
#undef VECTOR3D_BINARY_OP

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool operator==(Vector3D<Type> const &lhs, Vector3D<Type> const &rhs)
{
  return Abs(lhs[0] - rhs[0]) < kToleranceDist<Type> && Abs(lhs[1] - rhs[1]) < kToleranceDist<Type> &&
         Abs(lhs[2] - rhs[2]) < kToleranceDist<Type>;
}

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool operator!=(Vector3D<Type> const &lhs, Vector3D<Type> const &rhs)
{
  return !(lhs == rhs);
}

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool operator<(Vector3D<Type> const &lhs, Vector3D<Type> const &rhs)
{
  return lhs[0] < rhs[0] && lhs[1] < rhs[1] && lhs[2] < rhs[2];
}

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool operator<=(Vector3D<Type> const &lhs, Vector3D<Type> const &rhs)
{
  return lhs[0] <= rhs[0] && lhs[1] <= rhs[1] && lhs[2] <= rhs[2];
}

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool operator>(Vector3D<Type> const &lhs, Vector3D<Type> const &rhs)
{
  return lhs[0] > rhs[0] && lhs[1] > rhs[1] && lhs[2] > rhs[2];
}

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool operator>=(Vector3D<Type> const &lhs, Vector3D<Type> const &rhs)
{
  return lhs[0] >= rhs[0] && lhs[1] >= rhs[1] && lhs[2] >= rhs[2];
}

template <typename Type>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Type> operator-(Vector3D<Type> const &vec)
{
  return Vector3D<Type>(-vec[0], -vec[1], -vec[2]);
}

VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
Vector3D<bool> operator!(Vector3D<bool> const &vec) { return Vector3D<bool>(!vec[0], !vec[1], !vec[2]); }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#define VECTOR3D_SCALAR_BOOLEAN_LOGICAL_OP(OPERATOR)                                               \
  VECCORE_ATT_HOST_DEVICE                                                                          \
  VECGEOM_FORCE_INLINE                                                                             \
  Vector3D<bool> operator OPERATOR(Vector3D<bool> const &lhs, Vector3D<bool> const &rhs)           \
  {                                                                                                \
    return Vector3D<bool>(lhs[0] OPERATOR rhs[0], lhs[1] OPERATOR rhs[1], lhs[2] OPERATOR rhs[2]); \
  }
VECTOR3D_SCALAR_BOOLEAN_LOGICAL_OP(&&)
VECTOR3D_SCALAR_BOOLEAN_LOGICAL_OP(||)
#undef VECTOR3D_SCALAR_BOOLEAN_LOGICAL_OP
#pragma GCC diagnostic pop
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

namespace vecCore {

template <typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void MaskedAssign(vecgeom::Vector3D<T> &v, const vecCore::Mask<T> &mask,
                                                               const vecgeom::Vector3D<T> &val)
{
  vecCore::MaskedAssign(v[0], mask, val[0]);
  vecCore::MaskedAssign(v[1], mask, val[1]);
  vecCore::MaskedAssign(v[2], mask, val[2]);
}

/// @brief Minimum between two vectors
/// @tparam T Vector type
/// @param v1 first vector
/// @param v2 second vector
/// @return Vector having the minimum of the two vector components
namespace math {
template <typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE vecgeom::Vector3D<T> Min(vecgeom::Vector3D<T> const &v1,
                                                                      vecgeom::Vector3D<T> const &v2)
{
  vecgeom::Vector3D<T> result(vecCore::math::Min(v1.x(), v2.x()), vecCore::math::Min(v1.y(), v2.y()),
                              vecCore::math::Min(v1.z(), v2.z()));
  return result;
}

/// @brief Maximum between two vectors
/// @tparam T Vector type
/// @param v1 first vector
/// @param v2 second vector
/// @return Vector having the maximum of the two vector components
template <typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE vecgeom::Vector3D<T> Max(vecgeom::Vector3D<T> const &v1,
                                                                      vecgeom::Vector3D<T> const &v2)
{
  vecgeom::Vector3D<T> result(vecCore::math::Max(v1.x(), v2.x()), vecCore::math::Max(v1.y(), v2.y()),
                              vecCore::math::Max(v1.z(), v2.z()));
  return result;
}
} // namespace math
} // namespace vecCore

// for use in GEANT4
using UVector3 = VECGEOM_NAMESPACE::Vector3D<double>;

#endif // VECGEOM_BASE_VECTOR3D_H_
