#ifndef LIBRARYCILK_H
#define LIBRARYCILK_H

#include "LibraryGeneric.h"

template <typename Type>
struct CilkVector {

private:

  Type &vec;
  int size;
  bool free;

public:

  CilkVector(Type &arr, const int size_, const bool free_ = true)
      : vec(arr), size(size_), free(free_) {}
  CilkVector(Type *arr, const int size_, const bool free_ = true)
      : vec(*arr), size(size_), free(free_) {}
  CilkVector(Type scalar) : vec(scalar), size(1), free(false) {}

  ~CilkVector() {
    if (free) {
      _mm_free(&vec);
    }
  }

  CilkVector<Type>& operator=(CilkVector<Type> const &other) {
    vec[0:size] = other.vec[0:size];
    return *this;
  }

  CilkVector<Type>& CilkVector::operator=(Type const &scalar) {
    vec[0:size] = scalar;
    return *this;
  }

  CilkVector<Type> CilkVector::operator+(CilkVector<Type> const &lhs,
                                         CilkVector<Type> const &rhs) {
    const int s = lhs.size;
    CilkVector<Type> output(
      (Type*)_mm_malloc(sizeof(Type)*s, kAlignmentBoundary), s, true
    );
    output[0:size] = lhs[0:size] + rhs[0:size];
    return output;
  }

  CilkVector<Type> CilkVector::operator-(CilkVector<Type> const &lhs,
                                         CilkVector<Type> const &rhs) {
    const int s = lhs.size;
    CilkVector<Type> output(
      (Type*)_mm_malloc(sizeof(Type)*s, kAlignmentBoundary), s, true
    );
    output[0:size] = lhs[0:size] - rhs[0:size];
    return output;
  }

  CilkVector<Type> CilkVector::operator*(CilkVector<Type> const &lhs,
                                         Type const &rhs) {
    const int s = lhs.size;
    CilkVector<Type> output(
      (Type*)_mm_malloc(sizeof(Type)*s, kAlignmentBoundary), s, true
    );
    output[0:size] = lhs[0:size] * rhs;
    return output;
  }

  CilkVector<Type> CilkVector::operator/(CilkVector<Type> const &lhs,
                                         Type const &rhs) {
    const int s = lhs.size;
    CilkVector<Type> output(
      (Type*)_mm_malloc(sizeof(Type)*s, kAlignmentBoundary), s, true
    );
    output[0:size] = lhs[0:size] / rhs;
    return output;
  }

  CilkVector<Type> CilkVector::operator<(CilkVector<Type> const &lhs,
                                         Type const &rhs) {
    const int s = lhs.size;
    CilkVector<Type> output(
      (Type*)_mm_malloc(sizeof(Type)*s, kAlignmentBoundary), s, true
    );
    output[0:size] = lhs[0:size] < rhs;
    return output;
  }

  CilkVector<Type> CilkVector::operator>(CilkVector<Type> const &lhs,
                                         Type const &rhs) {
    const int s = lhs.size;
    CilkVector<Type> output(
      (Type*)_mm_malloc(sizeof(Type)*s, kAlignmentBoundary), s, true
    );
    output[0:size] = lhs[0:size] > rhs;
    return output;
  }

  CilkVector<Type> CilkVector::operator<=(CilkVector<Type> const &lhs,
                                          Type const &rhs) {
    const int s = lhs.size;
    CilkVector<Type> output(
      (Type*)_mm_malloc(sizeof(Type)*s, kAlignmentBoundary), s, true
    );
    output[0:size] = lhs[0:size] <= rhs;
    return output;
  }

  CilkVector<Type> CilkVector::operator>=(CilkVector<Type> const &lhs,
                                          Type const &rhs) {
    const int s = lhs.size;
    CilkVector<Type> output(
      (Type*)_mm_malloc(sizeof(Type)*s, kAlignmentBoundary), s, true
    );
    output[0:size] = lhs[0:size] >= rhs;
    return output;
  }

};

template <>
struct ImplTraits<kCilk> {
  typedef double             float_t;
  typedef CilkVector<int>    int_v;
  typedef CilkVector<double> float_v;
  typedef CilkVector<bool>   bool_v;
  constexpr static bool early_return = false;
  const static float_v kZero;
  const static bool_v kTrue;
  const static bool_v kFalse;
};

typedef ImplTraits<kCilk>::float_v CilkFloat;
typedef ImplTraits<kCilk>::bool_v  CilkBool;
typedef ImplTraits<kCilk>::float_t CilkScalarFloat;

#endif /* LIBRARYCILK_H */