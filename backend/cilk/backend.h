/**
 * @file cilk/backend.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BACKEND_CILKBACKEND_H_
#define VECGEOM_BACKEND_CILKBACKEND_H_

#include <algorithm>
#include <iostream>

#include "base/global.h"

#include "backend/scalar/backend.h"

namespace vecgeom {

// Need a way to detect this... Don't want to include Vc just for this!
constexpr int kVectorSize = 4;

template <typename Type = Precision, int vec_size = kVectorSize>
struct CilkVector;

struct kCilk {
  typedef CilkVector<int>       int_v;
  typedef CilkVector<Precision> precision_v;
  typedef CilkVector<bool>      bool_v;
  constexpr static bool early_returns = false;
  const static precision_v kOne;
  const static precision_v kZero;
  const static bool_v kTrue;
  const static bool_v kFalse;
};

typedef kCilk::int_v       CilkInt;
typedef kCilk::precision_v CilkPrecision;
typedef kCilk::bool_v      CilkBool;

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
  void store(Type *destination) const {
    destination[0:vec_size] = vec[:];
  }

  VECGEOM_INLINE
  void Map(Type (*f)(const Type&)) {
    vec[:] = f(vec[:]);
  }

  VECGEOM_INLINE
  void Map(Type (*f)(const Type)) {
    vec[:] = f(vec[:]);
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

};

/**
 * Currently only implemented for then/else-values which are Cilk vectors.
 * Should probably support scalar assignments as well.
 */
template <typename Type>
VECGEOM_INLINE
void CondAssign(CilkBool const &cond,
                Type const &thenval, Type const &elseval, Type *const output) {
  if (cond.vec[:]) {
    output->vec[:] = thenval.vec[:];
  } else {
    output->vec[:] = elseval.vec[:];
  }
}

/**
 * Currently only implemented for then/else-values which are Cilk vectors.
 * Should probably support scalar assignments as well.
 */
template <typename Type1, typename Type2>
VECGEOM_INLINE
void MaskedAssign(CilkBool const &cond,
                  Type1 const &thenval, Type2 *const output) {
  if (cond.vec[:]) {
    output->vec[:] = thenval.vec[:];
  }
}

VECGEOM_INLINE
CilkPrecision Abs(CilkPrecision const &val) {
  CilkPrecision result;
  result.Map(std::fabs);
  return result;
}

VECGEOM_INLINE
CilkPrecision Sqrt(CilkPrecision const &val) {
  CilkPrecision result;
  result.Map(std::sqrt);
  return result;
}

VECGEOM_INLINE
CilkPrecision Min(CilkPrecision const &val1, CilkPrecision const &val2) {
  CilkPrecision result;
  result.vec[:] = std::min(val1.vec[:], val2.vec[:]);
  return result;
}

VECGEOM_INLINE
CilkPrecision Max(CilkPrecision const &val1, CilkPrecision const &val2) {
  CilkPrecision result;
  result.vec[:] = std::max(val1.vec[:], val2.vec[:]);
  return result;
}

VECGEOM_INLINE
CilkInt Min(CilkInt const &val1, CilkInt const &val2) {
  CilkInt result;
  result.vec[:] = std::min(val1.vec[:], val2.vec[:]);
  return result;
}

VECGEOM_INLINE
CilkInt Max(CilkInt const &val1, CilkInt const &val2) {
  CilkInt result;
  result.vec[:] = std::max(val1.vec[:], val2.vec[:]);
  return result;
}

} // End namespace vecgeom

#endif // VECGEOM_BACKEND_CILKBACKEND_H_