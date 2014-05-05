/**
 * @file vector3d.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_VECTOR3D_H_
#define VECGEOM_BASE_VECTOR3D_H_

#include "base/global.h"
#include "backend/backend.h"
#ifndef VECGEOM_NVCC
  #if (defined(VECGEOM_VC) || defined(VECGEOM_VC_ACCELERATION))
    #include <Vc/Vc>
  #endif
#endif

#include <string>
#include <cstdlib>

namespace VECGEOM_NAMESPACE {

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
  VECGEOM_INLINE
  Vector3D(const Type a, const Type b, const Type c) {
    vec[0] = a;
    vec[1] = b;
    vec[2] = c;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D() {
    Vector3D<Type>(Type(0.), Type(0.), Type(0.));
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D(const Type a) {
    Vector3D<Type>(a, a, a);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
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
   * @return Length of the vector as sqrt(x^2 + y^2 + z^2).
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Type Mag2() const {
	  return Dot(*this,*this);
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
    * The dot product of two Vector3D<T> objects
    * @return T ( where T is float, double, or various SIMD vector types )
    */
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   static
   Type Dot( Vector3D<Type> const & left, Vector3D<Type> const & right ) {
      return left[0]*right[0] + left[1]*right[1] + left[2]*right[2];
   }

   /**
     * The dot product of two Vector3D<T> objects
     * @return T ( where T is float, double, or various SIMD vector types )
     */
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   Type Dot( Vector3D<Type> const & right ) const {
	   return Dot(*this, right);
   }


  /**
   * The cross (vector) product of two Vector3D<T> objects
   * @return Type ( where Type is float, double, or various SIMD vector types )
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static
  Vector3D<Type> Cross( Vector3D<Type> const & left, Vector3D<Type> const & right ) {
      return Vector3D<Type>( left[1]*right[2] - left[2]*right[1],
                             left[2]*right[0] - left[0]*right[2],
                             left[0]*right[1] - left[1]*right[0] );
  }

  /**
    * The cross (vector) product of two Vector3D<T> objects
    * @return Type ( where Type is float, double, or various SIMD vector types )
    */
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   Vector3D<Type> Cross( Vector3D<Type> const & right ) const {
       return Cross(*this, right);
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
    return Vector3D<Type>(VECGEOM_NAMESPACE::Abs(vec[0]),
                          VECGEOM_NAMESPACE::Abs(vec[1]),
                          VECGEOM_NAMESPACE::Abs(vec[2]));
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

  // Inplace binary operators

  #define INPLACE_BINARY_OP(OPERATOR) \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType& operator OPERATOR(const VecType &other) { \
    this->vec[0] OPERATOR other.vec[0]; \
    this->vec[1] OPERATOR other.vec[1]; \
    this->vec[2] OPERATOR other.vec[2]; \
    return *this; \
  } \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
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
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType operator OPERATOR(const VecType &other) const { \
    VecType result(*this); \
    result INPLACE other; \
    return result; \
  } \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
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

std::ostream& operator<<(std::ostream& os, Vector3D<Precision> const &vec);

#if (defined(VECGEOM_VC_ACCELERATION) && !defined(VECGEOM_NVCC))
/**
*
* This is a template specialization of of class Vector3D<double> or Vector3D<float> in case when we have
* Vc as a backend. The goal of the class is to provide internal vectorization of common vector operations
*
*/

template <>
class Vector3D<Precision> : public Vc::VectorAlignedBase {

  typedef Vector3D<Precision> VecType;
  typedef Vector3D<bool> BoolType;
  typedef Vc::Vector<Precision> base_t;

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
  Precision Mag2() const {
	  return Dot(*this,*this);
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

  /**
    * The dot product of two Vector3D<> objects
    * TODO: This function should be internally vectorized ( if proven to be beneficial )
    * @return Precision ( float or double )
    */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static
  Precision Dot( Vector3D<Precision> const & left, Vector3D<Precision> const & right ) {
	  base_t s(Vc::Zero);
	  for (unsigned i=0; i < 1 + 3/Vc::Vector<double>::Size; ++i) {
		  //   std::cerr << "adding vector " << i <<  std::endl;
		  base_t tmp1 = left.mem.vector(i);
		  base_t tmp2 = right.mem.vector(i);
		  s+=tmp1*tmp2;
	    }
	  return s.sum();
	  //  return left[0]*right[0] + left[1]*right[1] + left[2]*right[2];
  }

  /**
     * The dot product of two Vector3D<> objects
     * TODO: This function should be internally vectorized ( if proven to be beneficial )
     * @return Precision ( float or double )
     */
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   Precision Dot( Vector3D<Precision> const & right ) const {
      return Dot( *this, right );
   }


  /**
   * The cross product of two Vector3D<T> objects
   * @return Type ( where Type is float, double, or various SIMD vector types )
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static
  VecType Cross( VecType const & left, VecType const & right ) {
      return VecType( left[1]*right[2] - left[2]*right[1],
                      left[2]*right[0] - left[0]*right[2],
                      left[0]*right[1] - left[1]*right[0] );
  }

  /**
     * The cross product with another Vector3D<T> objects
     * @return Type ( where Type is float, double, or various SIMD vector types )
     */
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    VecType Cross( VecType const & right ) const {
        return Cross(*this, right);
    }

    /**
   * returns abs of the vector ( as a vector )
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> Abs() const {
      VecType tmp;
      for (unsigned i = 0; i < 1 + 3/Vc::Vector<Precision>::Size; ++i) {
        base_t v = this->mem.vector(i);
        tmp.mem.vector(i) = Vc::abs( v );
      }
      return tmp;
  }

  // Inplace binary operators

  #define INPLACE_BINARY_OP(OPERATOR) \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType& operator OPERATOR(const VecType &other) { \
    this->mem OPERATOR other.mem; \
    return *this; \
  } \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType& operator OPERATOR(const Precision &scalar) { \
    this->mem OPERATOR scalar; \
    return *this; \
  }
  INPLACE_BINARY_OP(+=)
  INPLACE_BINARY_OP(-=)
  INPLACE_BINARY_OP(*=)
  INPLACE_BINARY_OP(/=)
  #undef INPLACE_BINARY_OP

  // Binary operators

  #define BINARY_OP(OPERATOR, INPLACE) \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType operator OPERATOR(const VecType &other) const { \
    VecType result(*this); \
    result INPLACE other; \
    return result; \
  } \
  VECGEOM_CUDA_HEADER_BOTH \
  VECGEOM_INLINE \
  VecType operator OPERATOR(const Precision &scalar) const { \
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

#endif // (defined(VECGEOM_VC_ACCELERATION) && !defined(VECGEOM_NVCC))

#define BOOLEAN_OP(OPERATOR) \
template <typename Type> \
Vector3D<bool> operator OPERATOR(Vector3D<Type> const &lhs, \
                                 Vector3D<Type> const &rhs) { \
  return Vector3D<bool>(lhs[0] OPERATOR rhs[0], \
                        lhs[1] OPERATOR rhs[1], \
                        lhs[2] OPERATOR rhs[2]); \
}
BOOLEAN_OP(<)
BOOLEAN_OP(>)
BOOLEAN_OP(<=)
BOOLEAN_OP(>=)
BOOLEAN_OP(==)
BOOLEAN_OP(!=)
#undef BOOLEAN_OP

#ifdef VECGEOM_VC

#define BOOLEAN_OP(OPERATOR) \
template <typename Type> \
Vector3D<typename Vc::Vector<Type>::Mask> operator OPERATOR( \
    Vector3D<typename Vc::Vector<Type> > const &lhs, \
    Vector3D<typename Vc::Vector<Type> > const &rhs) { \
  return Vector3D<typename Vc::Vector<Type>::Mask>(lhs[0] OPERATOR rhs[0], \
                                                   lhs[1] OPERATOR rhs[1], \
                                                   lhs[2] OPERATOR rhs[2]); \
} \
template <typename Type> \
Vector3D<typename Vc::Vector<Type>::Mask> operator OPERATOR( \
    Vector3D<typename Vc::Vector<Type> > const &lhs, \
    Vector3D<Type> const &rhs) { \
  return Vector3D<typename Vc::Vector<Type>::Mask>(lhs[0] OPERATOR rhs[0], \
                                                   lhs[1] OPERATOR rhs[1], \
                                                   lhs[2] OPERATOR rhs[2]); \
}
BOOLEAN_OP(<)
BOOLEAN_OP(>)
BOOLEAN_OP(<=)
BOOLEAN_OP(>=)
BOOLEAN_OP(==)
BOOLEAN_OP(!=)
#undef BOOLEAN_OP

#endif // VECGEOM_VC

} // End global namespace

#endif // VECGEOM_BASE_VECTOR3D_H_
