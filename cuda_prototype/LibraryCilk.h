#ifndef LIBRARYCILK_H
#define LIBRARYCILK_H

#include "LibraryGeneric.h"

template <int vec_size = 8, typename Type = double>
struct CilkVector {

  typedef CilkVector<vec_size, Type> VecType;
  typedef CilkVector<vec_size, bool> VecBool;

public:

  Type vec[vec_size] __attribute__((aligned(32)));

  CilkVector() {
    // user should not assume any default value
  }

  CilkVector(Type const scalar) {
    vec[:] = scalar;
  }

  explicit CilkVector(Type const *from) {
    vec[:] = from[0:vec_size];
  }

  CilkVector(CilkVector const &other) {
    vec[:] = other.vec[:];
  }

  inline Type* Vec() { return vec; }

  inline Type const* Vec() const { return vec; }

  static constexpr int VecSize() { return vec_size; }

  void Store(Type *destination) const {
    destination[0:vec_size] = vec[:];
  }

  inline VecType& operator=(VecType const &other) {
    vec[:] = other.vec[:];
    return *this;
  }

  inline VecType& operator=(Type const &scalar) {
    vec[:] = scalar;
    return *this;
  }

  inline VecType& operator+=(VecType const &other) {
    this->vec[:] += other.vec[:];
    return *this;
  }

  inline VecType& operator+=(Type const &scalar) {
    this->vec[:] += scalar;
    return *this;
  }

  inline VecType& operator-=(VecType const &other) {
    this->vec[:] -= other.vec[:];
    return *this;
  }

  inline VecType& operator-=(Type const &scalar) {
    this->vec[:] -= scalar;
    return *this;
  }

  inline VecType& operator*=(VecType const &other) {
    this->vec[:] *= other.vec[:];
    return *this;
  }

  inline VecType& operator*=(Type const &scalar) {
    this->vec[:] *= scalar;
    return *this;
  }

  inline VecType& operator/=(VecType const &other) {
    this->vec[:] /= other.vec[:];
    return *this;
  }

  inline VecType& operator/=(Type const &scalar) {
    this->vec[:] /= scalar;
    return *this;
  }

  inline VecType& operator|=(VecType const &other) {
    this->vec[:] |= other.vec[:];
    return *this;
  }

  inline VecType& operator|=(Type const &scalar) {
    this->vec[:] |= scalar;
    return *this;
  }

  inline VecType operator+(const VecType& other) const {
    VecType result;
    result.vec[:] = this->vec[:] + other.vec[:];
    return result;
  }


  inline VecType operator+(const Type& scalar) const {
    VecType result;
    result.vec[:] = this->vec[:] + scalar;
    return result;
  }

  inline VecType operator-(const VecType& other) const {
    VecType result;
    result.vec[:] = this->vec[:] - other.vec[:];
    return result;
  }


  inline VecType operator-(const Type& scalar) const {
    VecType result;
    result.vec[:] = this->vec[:] - scalar;
    return result;
  }

  inline VecType operator*(const VecType& other) const {
    VecType result;
    result.vec[:] = this->vec[:] * other.vec[:];
    return result;
  }


  inline VecType operator*(const Type& scalar) const {
    VecType result;
    result.vec[:] = this->vec[:] * scalar;
    return result;
  }

  inline VecType operator/(const VecType& other) const {
    VecType result;
    result.vec[:] = this->vec[:] / other.vec[:];
    return result;
  }


  inline VecType operator/(const Type& scalar) const {
    VecType result;
    result.vec[:] = this->vec[:] / scalar;
    return result;
  }

  inline VecBool operator>(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] > other.vec[:];
    return result;
  }

  inline VecBool operator>(const Type& scalar) const {
    VecBool result;
    result.vec[:] = this->vec[:] > scalar;
    return result;
  }

  inline VecBool operator>=(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] >= other.vec[:];
    return result;
  }

  inline VecBool operator>=(const Type& scalar) const {
    VecBool result;
    result.vec[:] = this->vec[:] >= scalar;
    return result;
  }

  inline VecBool operator<(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] < other.vec[:];
    return result;
  }

  inline VecBool operator<(const Type& scalar) const {
    VecBool result;
    result.vec[:] = this->vec[:] < scalar;
    return result;
  }

  inline VecBool operator<=(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] <= other.vec[:];
    return result;
  }

  inline VecBool operator<=(const Type& scalar) const {
    VecBool result;
    result.vec[:] = this->vec[:] <= scalar;
    return result;
  }

  inline VecBool operator||(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] || other.vec[:];
    return result;
  }

  inline VecBool operator&&(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] && other.vec[:];
    return result;
  }

  inline VecBool operator==(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] == other.vec[:];
    return result;
  }

  inline VecBool operator==(const Type& scalar) const {
    VecBool result;
    result.vec[:] = this->vec[:] == scalar;
    return result;
  }

  inline VecBool operator!=(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] != other.vec[:];
    return result;
  }

  inline VecBool operator!=(const Type& scalar) const {
    VecBool result;
    result.vec[:] = this->vec[:] != scalar;
    return result;
  }

  inline VecType operator!() const {
    VecType result;
    result.vec[:] = !vec[:];
    return result;
  }

  explicit operator bool() {
    return __sec_reduce_all_nonzero(vec[:]);
  }

  friend inline
  std::ostream& operator<<(std::ostream& os, VecType const &v) {
    if (vec_size <= 0) {
      os << "[]\n";
      return os;
    }
    os << "[" << v.vec[0];
    for (int i = 1; i < vec_size; ++i) os << ", " << v.vec[i];
    os << "]";
    return os;
  }

};

template <>
struct Impl<kCilk> {
  typedef double                float_t;
  typedef CilkVector<8, int>    int_v;
  typedef CilkVector<8, double> float_v;
  typedef CilkVector<8, bool>   bool_v;
  constexpr static bool early_return = false;
  constexpr static double kZero = 0;
  constexpr static bool kTrue = true;
  constexpr static bool kFalse = false;
};

typedef Impl<kCilk>::float_v CilkFloat;
typedef Impl<kCilk>::bool_v  CilkBool;
typedef Impl<kCilk>::float_t CilkScalarFloat;

template <typename Type1, typename Type2>
inline __attribute__((always_inline))
void MaskedAssign(const CilkBool &cond,
                  const Type1 &thenval, Type2 &output) {
  if (cond.vec[:]) {
    output.vec[:] = thenval.vec[:];
  }
}

template <>
inline __attribute__((always_inline))
CilkFloat Abs<kCilk, CilkFloat>(CilkFloat const &val) {
  CilkFloat result;
  result.vec[:] = std::fabs(val.vec[:]);
  return result;
}

template <>
inline __attribute__((always_inline))
CilkFloat Sqrt<kVc, CilkFloat>(CilkFloat const &val) {
  CilkFloat result;
  result.vec[:] = std::sqrt(val.vec[:]);
  return result;
}

#endif /* LIBRARYCILK_H */