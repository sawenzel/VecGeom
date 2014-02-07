#ifndef VECGEOM_BASE_VECTOR3D_H_
#define VECGEOM_BASE_VECTOR3D_H_

#include <iostream>
#include <string>
#include "base/types.h"
#include "base/utilities.h"
#ifdef VECGEOM_VC_ACCELERATION
  #include <Vc/Vc>
#endif

namespace vecgeom {

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
    vec[0] = atof(str.substr(begin, end-begin).c_str());
    begin = end + 2;
    end = str.find(",", begin);
    vec[1] = atof(str.substr(begin, end-begin).c_str());
    begin = end + 2;
    end = str.find(")", begin);
    vec[2] = atof(str.substr(begin, end-begin).c_str());
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

  template <typename TypeOther>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType& operator+=(Vector3D<TypeOther> const &rhs) {
    this->vec[0] += rhs[0];
    this->vec[1] += rhs[1];
    this->vec[2] += rhs[2];
    return *this;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType& operator+=(Type const &scalar) {
    this->vec[0] += scalar;
    this->vec[1] += scalar;
    this->vec[2] += scalar;
    return *this;
  }

  template <typename TypeOther>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType& operator-=(Vector3D<TypeOther> const &rhs) {
    this->vec[0] -= rhs[0];
    this->vec[1] -= rhs[1];
    this->vec[2] -= rhs[2];
    return *this;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType& operator-=(Type const &scalar) {
    this->vec[0] -= scalar;
    this->vec[1] -= scalar;
    this->vec[2] -= scalar;
    return *this;
  }

  template <typename TypeOther>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType& operator*=(Vector3D<TypeOther> const &rhs) {
    this->vec[0] *= rhs[0];
    this->vec[1] *= rhs[1];
    this->vec[2] *= rhs[2];
    return *this;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType& operator*=(Type const &scalar) {
    this->vec[0] *= scalar;
    this->vec[1] *= scalar;
    this->vec[2] *= scalar;
    return *this;
  }

  template <typename TypeOther>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType& operator/=(Vector3D<TypeOther> const &rhs) {
    this->vec[0] /= rhs[0];
    this->vec[1] /= rhs[1];
    this->vec[2] /= rhs[2];
    return *this;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType& operator/=(Type const &scalar) {
    const Type inverse = Type(1) / scalar;
    *this *= inverse;
    return *this;
  }

  template <typename TypeOther>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType operator+(Vector3D<TypeOther> const &other) const {
    VecType result(*this);
    result += other;
    return result;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType operator+(Type const &scalar) const {
    VecType result(*this);
    result += scalar;
    return result;
  }

  template <typename TypeOther>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType operator-(Vector3D<TypeOther> const &other) const {
    VecType result(*this);
    result -= other;
    return result;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType operator-(Type const &scalar) const {
    VecType result(*this);
    result -= scalar;
    return result;
  }

  template <typename TypeOther>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType operator*(Vector3D<TypeOther> const &other) const {
    VecType result(*this);
    result *= other;
    return result;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType operator*(Type const &scalar) const {
    VecType result(*this);
    result *= scalar;
    return result;
  }

  template <typename TypeOther>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType operator/(Vector3D<TypeOther> const &other) const {
    VecType result(*this);
    result /= other;
    return result;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VecType operator/(Type const &scalar) const {
    VecType result(*this);
    result /= scalar;
    return result;
  }

  #ifdef VECGEOM_STD_CXX11
  friend
  VECGEOM_INLINE
  std::ostream& operator<<(std::ostream& os, VecType const &v) {
    os << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
    return os;
  }
  #endif /* VECGEOM_STD_CXX11 */

  /**
   * \return Length of the vector as sqrt(x^2 + y^2 + z^2).
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Length() const {
    return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
  }

  /**
   * Normalizes the vector by dividing each entry by the length.
   * \sa Vector3D::Length()
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Normalize() {
    *this /= Length();
  }

  /**
   * Maps each vector entry to a function that manipulates the entry type.
   * \param f A function of type "Type f(const Type&)" to map over entries.
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Map(Type (*f)(const Type&)) {
    vec[0] = f(vec[0]);
    vec[1] = f(vec[1]);
    vec[2] = f(vec[2]);
  }

};

#ifdef VECGEOM_VC_ACCELERATION // Activated as a compiler flag

template <>
class Vector3D<double> {

  typedef Vector3D<double> VecType;

private:

  Vc::Memory<Vc::double_v, 3> mem;

public:

  Vector3D(const double a, const double b, const double c) {
    mem[0] = a;
    mem[1] = b;
    mem[2] = c;
  }

  Vector3D(const double a) {
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
  double& operator[](const int index) {
    return mem[index];
  }

  VECGEOM_INLINE
  double const& operator[](const int index) const {
    return mem[index];
  }

  VECGEOM_INLINE
  double& x() { return mem[0]; }

  VECGEOM_INLINE
  double const& x() const { return mem[0]; }

  VECGEOM_INLINE
  double& y() { return mem[1]; }

  VECGEOM_INLINE
  double const& y() const { return mem[1]; }

  VECGEOM_INLINE
  double& z() { return mem[2]; }

  VECGEOM_INLINE
  double const& z() const { return mem[2]; }

  template <typename TypeOther>
  VECGEOM_INLINE
  VecType& operator+=(Vector3D<TypeOther> const &rhs) {
    this->mem += rhs.mem;
    return *this;
  }

  VECGEOM_INLINE
  VecType& operator+=(double const &scalar) {
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
  VecType& operator-=(double const &scalar) {
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
  VecType& operator*=(double const &scalar) {
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
  VecType& operator/=(double const &scalar) {
    const double inverse = 1.0 / scalar;
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
  VecType operator+(double const &scalar) const {
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
  VecType operator-(double const &scalar) const {
    VecType result(*this);
    result -= scalar;
    return result;
  }

  template <typename TypeOther>
  VECGEOM_INLINE
  VecType operator*(Vector3D<TypeOther> const &other) const {
    VecType result(*this);
    result *= other;
    return result;
  }

  VECGEOM_INLINE
  VecType operator*(double const &scalar) const {
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
  VecType operator/(double const &scalar) const {
    VecType result(*this);
    result /= scalar;
    return result;
  }

  #ifdef VECGEOM_STD_CXX11
  friend
  VECGEOM_INLINE
  std::ostream& operator<<(std::ostream& os, VecType const &v) {
    os << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
    return os;
  }
  #endif /* VECGEOM_STD_CXX11 */

  VECGEOM_INLINE
  double Length() const {
    return sqrt(mem[0]*mem[0] + mem[1]*mem[1] + mem[2]*mem[2]);
  }

  VECGEOM_INLINE
  void Normalize() {
    *this /= Length();
  }

  VECGEOM_INLINE
  void Map(double (*f)(const double&)) {
    mem[0] = f(mem[0]);
    mem[1] = f(mem[1]);
    mem[2] = f(mem[2]);
  }

};

#endif // Vc acceleration enabled

} // End namespace vecgeom

#endif // VECGEOM_BASE_VECTOR3D_H_