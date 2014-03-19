/**
 * @file vector3d.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_VECTOR3D_H_
#define VECGEOM_BASE_VECTOR3D_H_

#include <iostream>
#include <string>
#include <cstdlib>
#ifdef VECGEOM_VC_ACCELERATION
#include <Vc/Vc>
#endif
#include "base/global.h"

#include "backend.h"

namespace vecgeom {

/**
 * @brief Three dimensional vector class supporting most arithmetic operations.
 * @details If vector acceleration is enabled, the scalar template instantiation
 *          will use vector instructions for operations when possible.
 */
template <typename Type>
class Vector3D {

  typedef Vector3D<Type> VecType;

private:

  Type vec[3];

public:

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D() {
    vec[0] = 0;
    vec[1] = 0;
    vec[2] = 0;
  }

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D(const Type a, const Type b, const Type c) {
    vec[0] = a;
    vec[1] = b;
    vec[2] = c;
  }

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D(const Type a) {
    Vector3D(a, a, a);
  }

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D(Vector3D const &other) {
    vec[0] = other[0];
    vec[1] = other[1];
    vec[2] = other[2];
  }

  /**
   * Constructs a vector from an std::string of the same format as output by the
   * "<<"-operator for outstreams.
   * @param str String formatted as "(%d, %d, %d)".
   */
  VECGEOM_CUDA_HEADER_HOST
  Vector3D(std::string const &str) {
    int begin = 1, end = str.find(",");
    vec[0] = std::atof(str.substr(begin, end-begin).c_str());
    begin = end + 2;
    end = str.find(",", begin);
    vec[1] = std::atof(str.substr(begin, end-begin).c_str());
    begin = end + 2;
    end = str.find(")", begin);
    vec[2] = std::atof(str.substr(begin, end-begin).c_str());
  }

  /**
   * Contains no check for correct indexing to avoid impairing performance.
   * @param index Index of content in the range [0-2].
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& operator[](const int index) {
    return vec[index];
  }

  /**
   * Contains no check for correct indexing to avoid impairing performance.
   * @param index Index of content in the range [0-2].
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& operator[](const int index) const {
    return vec[index];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& x() { return vec[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& x() const { return vec[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& y() { return vec[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& y() const { return vec[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type& z() { return vec[2]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type const& z() const { return vec[2]; }

  /**
   * @return Length of the vector as sqrt(x^2 + y^2 + z^2).
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Length() const {
    return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
  }

  /**
   * Normalizes the vector by dividing each entry by the length.
   * @sa Vector3D::Length()
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Normalize() {
    *this /= Length();
  }

  /**
   * Maps each vector entry to a function that manipulates the entry type.
   * @param f A function of type "Type f(const Type&)" to map over entries.
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Map(Type (*f)(const Type&)) {
    vec[0] = f(vec[0]);
    vec[1] = f(vec[1]);
    vec[2] = f(vec[2]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Type> Abs() const {
    return Vector3D<Type>(vecgeom::Abs(vec[0]), vecgeom::Abs(vec[1]),
                          vecgeom::Abs(vec[2]));
  }

  template <typename BoolType>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void MaskedAssign(Vector3D<BoolType> const &condition,
                    Vector3D<Type> const &value) {
    vec[0] = (condition[0]) ? value[0] : vec[0];
    vec[1] = (condition[1]) ? value[1] : vec[1];
    vec[2] = (condition[2]) ? value[2] : vec[2];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Min() const {
    Type min = (vec[1] < vec[0]) ? vec[1] : vec[0];
    min = (vec[2] < min) ? vec[2] : min;
    return min;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Max() const {
    Type max = (vec[1] > vec[0]) ? vec[1] : vec[0];
    max = (vec[2] > max) ? vec[2] : max;
    return max;
  }

  #define INPLACE_BINARY_OP(OPERATOR) \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType& operator OPERATOR(const VecType &other) { \
    this->vec[0] OPERATOR other.vec[0]; \
    this->vec[1] OPERATOR other.vec[1]; \
    this->vec[2] OPERATOR other.vec[2]; \
  } \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType& operator OPERATOR(const Type &other) { \
    this->vec[0] OPERATOR other; \
    this->vec[1] OPERATOR other; \
    this->vec[2] OPERATOR other; \
  }
  INPLACE_BINARY_OP(+=)
  INPLACE_BINARY_OP(-=)
  INPLACE_BINARY_OP(*=)
  INPLACE_BINARY_OP(/=)
  #undef INPLACE_BINARY_OP

  #define BINARY_OP(OPERATOR, INPLACE) \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType& operator OPERATOR(const VecType &other) { \
    VecType result(*this); \
    result INPLACE other; \
    return result; \
  } \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType& operator OPERATOR(const Type &other) { \
    VecType result(*this); \
    result INPLACE other; \
    return result; \
  }
  BINARY_OP(+, +=)
  BINARY_OP(-, -=)
  BINARY_OP(*, *=)
  BINARY_OP(/, /=)
  #undef BINARY_OP

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<bool> operator<(VecType const &other) const {
    Vector3D<bool> result;
    result[0] = this->vec[0] < other.vec[0];
    result[1] = this->vec[1] < other.vec[1];
    result[2] = this->vec[2] < other.vec[2];
    return result;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<bool> operator<(const Type other) const {
    Vector3D<bool> result;
    result[0] = this->vec[0] < other;
    result[1] = this->vec[1] < other;
    result[2] = this->vec[2] < other;
    return result;
  }

  VECGEOM_CUDA_HEADER_HOST
  VECGEOM_INLINE
  friend std::ostream& operator<<(std::ostream& os, VecType const &v) {
    os << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
    return os;
  }

};

#ifdef VECGEOM_VC_ACCELERATION // Activated as a compiler flag

template <>
class Vector3D<Precision> {

  typedef Vector3D<Precision> VecType;
  typedef Vector3D<bool> BoolType;

private:

  Vc::Memory<Vc::Vector<Precision>, 3> mem;

public:

  Vector3D(const Precision a, const Precision b, const Precision c) {
    mem[0] = a;
    mem[1] = b;
    mem[2] = c;
  }

  Vector3D(const Precision a) {
    mem = a;
  }

  Vector3D() : Vector3D(0) {}

  Vector3D(Vector3D const &other) {
    this->mem = other.mem;
  }

  Vector3D(std::string const &str) {
    int begin = 1, end = str.find(",");
    mem[0] = atof(str.substr(begin, end-begin).c_str());
    begin = end + 2;
    end = str.find(",", begin);
    mem[1] = atof(str.substr(begin, end-begin).c_str());
    begin = end + 2;
    end = str.find(")", begin);
    mem[2] = atof(str.substr(begin, end-begin).c_str());
  }

  VECGEOM_INLINE
  Precision& operator[](const int index) {
    return mem[index];
  }

  VECGEOM_INLINE
  const Precision& operator[](const int index) const {
    return mem[index];
  }

  VECGEOM_INLINE
  Precision& x() { return mem[0]; }

  VECGEOM_INLINE
  const Precision& x() const { return mem[0]; }

  VECGEOM_INLINE
  Precision& y() { return mem[1]; }

  VECGEOM_INLINE
  const Precision& y() const { return mem[1]; }

  VECGEOM_INLINE
  Precision& z() { return mem[2]; }

  VECGEOM_INLINE
  const Precision& z() const { return mem[2]; }

  VECGEOM_INLINE
  Precision Length() const {
    return sqrt(mem[0]*mem[0] + mem[1]*mem[1] + mem[2]*mem[2]);
  }

  VECGEOM_INLINE
  void Normalize() {
    *this /= Length();
  }

  VECGEOM_INLINE
  void Map(Precision (*f)(const Precision&)) {
    mem[0] = f(mem[0]);
    mem[1] = f(mem[1]);
    mem[2] = f(mem[2]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void MaskedAssign(Vector3D<bool> const &condition,
                    Vector3D<Precision> const &value) {
    mem[0] = (condition[0]) ? value[0] : mem[0];
    mem[1] = (condition[1]) ? value[1] : mem[1];
    mem[2] = (condition[2]) ? value[2] : mem[2];
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Min() const {
    return std::min(std::min(mem[0], mem[1]), mem[2]);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Max() const {
    return std::max(std::max(mem[0], mem[1]), mem[2]);
  }

  template <typename TypeOther>
  VECGEOM_INLINE
  VecType& operator+=(Vector3D<TypeOther> const &rhs) {
    this->mem += rhs.mem;
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator+=(const Precision scalar) {
    this->mem += scalar;
    return *this;
  }

  template <typename TypeOther>
  VECGEOM_INLINE
  VecType& operator-=(Vector3D<TypeOther> const &rhs) {
    this->mem -= rhs.mem;
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator-=(const Precision scalar) {
    this->mem -= scalar;
    return *this;
  }

  template <typename TypeOther>
  VECGEOM_INLINE
  VecType& operator*=(Vector3D<TypeOther> const &rhs) {
    this->mem *= rhs.mem;
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator*=(const Precision scalar) {
    this->mem *= scalar;
    return *this;
  }

  template <typename TypeOther>
  VECGEOM_INLINE
  VecType& operator/=(Vector3D<TypeOther> const &rhs) {
    this->mem /= rhs.mem;
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator/=(const Precision scalar) {
    const Precision inverse = 1.0 / scalar;
    *this *= inverse;
    return *this;
  }

  template <typename TypeOther>
  VECGEOM_INLINE
  VecType operator+(Vector3D<TypeOther> const &other) const {
    VecType result(*this);
    result += other;
    return result;
  }

  VECGEOM_INLINE
  VecType operator+(const Precision scalar) const {
    VecType result(*this);
    result += scalar;
    return result;
  }

  template <typename TypeOther>
  VECGEOM_INLINE
  VecType operator-(Vector3D<TypeOther> const &other) const {
    VecType result(*this);
    result -= other;
    return result;
  }

  VECGEOM_INLINE
  VecType operator-(const Precision scalar) const {
    VecType result(*this);
    result -= scalar;
    return result;
  }

  VECGEOM_INLINE
  VecType operator*(Vector3D<Precision> const &other) const {
    VecType result(*this);
    result *= other;
    return result;
  }

  VECGEOM_INLINE
  VecType operator*(const Precision scalar) const {
    VecType result(*this);
    result *= scalar;
    return result;
  }

  template <typename TypeOther>
  VECGEOM_INLINE
  VecType operator/(Vector3D<TypeOther> const &other) const {
    VecType result(*this);
    result /= other;
    return result;
  }

  VECGEOM_INLINE
  VecType operator/(const Precision scalar) const {
    VecType result(*this);
    result /= scalar;
    return result;
  }

  VECGEOM_INLINE
  BoolType operator<(VecType const &rhs) const {
    return this->mem < rhs.mem;
  }

  VECGEOM_INLINE
  BoolType operator<(const Precision scalar) const {
    Vector3D<Precision> rhs(scalar);
    return this->mem < rhs.mem;
  }


  VECGEOM_INLINE
  BoolType operator>(VecType const &rhs) const {
    return this->mem > rhs.mem;
  }

  VECGEOM_INLINE
  BoolType operator>(const Precision scalar) const {
    Vector3D<Precision> rhs(scalar);
    return this->mem > rhs.mem;
  }


  VECGEOM_INLINE
  BoolType operator<=(VecType const &rhs) const {
    return this->mem <= rhs.mem;
  }

  VECGEOM_INLINE
  BoolType operator<=(const Precision scalar) const {
    Vector3D<Precision> rhs(scalar);
    return this->mem <= rhs.mem;
  }


  VECGEOM_INLINE
  BoolType operator>=(VecType const &rhs) const {
    return this->mem >= rhs.mem;
  }

  VECGEOM_INLINE
  BoolType operator>=(const Precision scalar) const {
    Vector3D<Precision> rhs(scalar);
    return this->mem >= rhs.mem;
  }


  VECGEOM_INLINE
  BoolType operator==(VecType const &rhs) const {
    return this->mem == rhs.mem;
  }

  VECGEOM_INLINE
  BoolType operator==(const Precision scalar) const {
    Vector3D<Precision> rhs(scalar);
    return this->mem == rhs.mem;
  }


  VECGEOM_INLINE
  BoolType operator!=(VecType const &rhs) const {
    return this->mem != rhs.mem;
  }

  VECGEOM_INLINE
  BoolType operator!=(const Precision scalar) const {
    Vector3D<Precision> rhs(scalar);
    return this->mem != rhs.mem;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> Abs() const {
    return Vector3D<Precision>(vecgeom::Abs((*this)[0]),
                               vecgeom::Abs((*this)[1]),
                               vecgeom::Abs((*this)[2]));
  }

  #ifdef VECGEOM_STD_CXX11
  friend
  VECGEOM_INLINE
  std::ostream& operator<<(std::ostream& os, VecType const &v) {
    os << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
    return os;
  }
  #endif /* VECGEOM_STD_CXX11 */

};

#endif // Vc acceleration enabled

} // End namespace vecgeom

#endif // VECGEOM_BASE_VECTOR3D_H_