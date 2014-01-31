#ifndef VECGEOM_BACKEND_CILKBACKEND_H_
#define VECGEOM_BACKEND_CILKBACKEND_H_

#include <iostream>
#include "Base/Utilities.h"
#include "Base/Types.h"

// Need a way to detect this... Don't want to include Vc just for this!
constexpr int kVectorSize = 4;

template <typename Type = double, int vec_size = kVectorSize>
struct CilkVector;

template <>
struct Impl<kCilk> {
  typedef CilkVector<int>    int_v;
  typedef CilkVector<double> double_v;
  typedef CilkVector<bool>   bool_v;
  const static bool early_returns = false;
};

typedef Impl<kCilk>::int_v    CilkInt;
typedef Impl<kCilk>::double_v CilkDouble;
typedef Impl<kCilk>::bool_v   CilkBool;

/**
 * Wrapper struct to allow arithmetic operations to be performed using the Cilk
 * backend without explicitly using array notation in the kernel.
 */
template <typename Type, int vec_size>
struct CilkVector {

  typedef CilkVector<Type, vec_size> VecType;
  typedef CilkVector<bool, vec_size> VecBool;

public:

  // Left public to allow array notation
  Type vec[vec_size] __attribute__((aligned(32)));

  /**
   * User should not assume any default value when constructing without
   * arguments.
   */
  CilkVector() {}

  CilkVector(Type const scalar) {
    vec[:] = scalar;
  }

  explicit CilkVector(Type const *from) {
    vec[:] = from[0:vec_size];
  }

  CilkVector(CilkVector const &other) {
    vec[:] = other.vec[:];
  }

  VECGEOM_INLINE
  static constexpr int Size() { return vec_size; }

  VECGEOM_INLINE
  void Store(Type *destination) const {
    destination[0:vec_size] = vec[:];
  }

  VECGEOM_INLINE
  VecType& operator=(VecType const &other) {
    vec[:] = other.vec[:];
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator=(Type const &scalar) {
    vec[:] = scalar;
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator+=(VecType const &other) {
    this->vec[:] += other.vec[:];
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator+=(Type const &scalar) {
    this->vec[:] += scalar;
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator-=(VecType const &other) {
    this->vec[:] -= other.vec[:];
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator-=(Type const &scalar) {
    this->vec[:] -= scalar;
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator*=(VecType const &other) {
    this->vec[:] *= other.vec[:];
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator*=(Type const &scalar) {
    this->vec[:] *= scalar;
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator/=(VecType const &other) {
    this->vec[:] /= other.vec[:];
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator/=(Type const &scalar) {
    this->vec[:] /= scalar;
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator|=(VecType const &other) {
    this->vec[:] |= other.vec[:];
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator|=(Type const &scalar) {
    this->vec[:] |= scalar;
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator&=(VecType const &other) {
    this->vec[:] &= other.vec[:];
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator&=(Type const &scalar) {
    this->vec[:] &= scalar;
    return *this;
  }

  VECGEOM_INLINE
  VecType operator+(const VecType& other) const {
    VecType result(*this);
    result += other;
    return result;
  }


  VECGEOM_INLINE
  VecType operator+(const Type& scalar) const {
    VecType result(*this);
    result += scalar;
    return result;
  }

  VECGEOM_INLINE
  VecType operator-(const VecType& other) const {
    VecType result(*this);
    result -= other;
    return result;
  }


  VECGEOM_INLINE
  VecType operator-(const Type& scalar) const {
    VecType result(*this);
    result -= scalar;
    return result;
  }

  VECGEOM_INLINE
  VecType operator*(const VecType& other) const {
    VecType result(*this);
    result *= other;
    return result;
  }


  VECGEOM_INLINE
  VecType operator*(const Type& scalar) const {
    VecType result(*this);
    result *= scalar;
    return result;
  }

  VECGEOM_INLINE
  VecType operator/(const VecType& other) const {
    VecType result(*this);
    result /= other;
    return result;
  }


  VECGEOM_INLINE
  VecType operator/(const Type& scalar) const {
    VecType result(*this);
    result /= scalar;
    return result;
  }

  VECGEOM_INLINE
  VecBool operator>(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] > other.vec[:];
    return result;
  }

  VECGEOM_INLINE
  VecBool operator>(const Type& scalar) const {
    VecBool result;
    result.vec[:] = this->vec[:] > scalar;
    return result;
  }

  VECGEOM_INLINE
  VecBool operator>=(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] >= other.vec[:];
    return result;
  }

  VECGEOM_INLINE
  VecBool operator>=(const Type& scalar) const {
    VecBool result;
    result.vec[:] = this->vec[:] >= scalar;
    return result;
  }

  VECGEOM_INLINE
  VecBool operator<(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] < other.vec[:];
    return result;
  }

  VECGEOM_INLINE
  VecBool operator<(const Type& scalar) const {
    VecBool result;
    result.vec[:] = this->vec[:] < scalar;
    return result;
  }

  VECGEOM_INLINE
  VecBool operator<=(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] <= other.vec[:];
    return result;
  }

  VECGEOM_INLINE
  VecBool operator<=(const Type& scalar) const {
    VecBool result;
    result.vec[:] = this->vec[:] <= scalar;
    return result;
  }

  VECGEOM_INLINE
  VecBool operator||(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] || other.vec[:];
    return result;
  }

  VECGEOM_INLINE
  VecBool operator&&(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] && other.vec[:];
    return result;
  }

  VECGEOM_INLINE
  VecBool operator==(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] == other.vec[:];
    return result;
  }

  VECGEOM_INLINE
  VecBool operator==(const Type& scalar) const {
    VecBool result;
    result.vec[:] = this->vec[:] == scalar;
    return result;
  }

  VECGEOM_INLINE
  VecBool operator!=(const VecType& other) const {
    VecBool result;
    result.vec[:] = this->vec[:] != other.vec[:];
    return result;
  }

  VECGEOM_INLINE
  VecBool operator!=(const Type& scalar) const {
    VecBool result;
    result.vec[:] = this->vec[:] != scalar;
    return result;
  }

  VECGEOM_INLINE
  VecType operator!() const {
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

#endif // VECGEOM_BACKEND_CILKBACKEND_H_