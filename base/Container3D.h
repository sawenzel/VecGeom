/// \file Container3D.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_CONTAINER3D_H_
#define VECGEOM_BASE_CONTAINER3D_H_

#include "base/Global.h"

#include "base/Vector3D.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <class Implementation>
class Container3D;

template <template<typename> class ImplementationType, typename T>
class Container3D<ImplementationType<T> > {

protected:

  VECGEOM_CUDA_HEADER_BOTH
  Container3D() {}

  VECGEOM_CUDA_HEADER_BOTH
  ~Container3D() {}

private:

  typedef ImplementationType<T> Implementation;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Implementation& implementation() {
    return *static_cast<Implementation*>(this);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Implementation& implementation_const() const {
    return *static_cast<Implementation const*>(this);
  }

  typedef T value_type;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  size_t size() const {
    return implementation_const().size();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  size_t capacity() const {
    return implementation_const().capacity();
  }

  VECGEOM_INLINE
  void resize() {
    implementation().resize();
  }

  VECGEOM_INLINE
  void reserve() {
    implementation().reserve();
  }

  VECGEOM_INLINE
  void clear() {
    implementation().clear();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<value_type> operator[](size_t index) const {
    return implementation_const().operator[](index);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  value_type const& x(size_t index) const {
    return implementation_const().x(index);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  value_type& x(size_t index) {
    return implementation().x(index);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  value_type const& y(size_t index) const {
    return implementation_const().y(index);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  value_type& y(size_t index) {
    return implementation().y(index);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  value_type z(size_t index) const {
    return implementation_const().z(index);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  value_type& z(size_t index) {
    return implementation().z(index);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void set(size_t index, value_type xval, value_type yval, value_type zval) {
    implementation().set(index, xval, yval, zval);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void set(size_t index, Vector3D<value_type> const &vec) {
    implementation().set(index, vec);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void push_back(const value_type xval, const value_type yval, const value_type zval) {
    implementation().push_back(xval, yval, zval);
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void push_back(Vector3D<value_type> const &vec) {
    implementation().push_back(vec);
  }

};

} } // End global namespace

#endif // VECGEOM_BASE_CONTAINER3D_H_
