#ifndef VECGEOM_BASE_CONTAINER_H_
#define VECGEOM_BASE_CONTAINER_H_

#include "base/utilities.h"

namespace vecgeom {

template <typename Type>
class Iterator : public std::iterator<std::forward_iterator_tag, Type> {

public:

  Iterator(Type const *const e) : element_(e) {}

  Iterator& operator=(Iterator const &other) {
    element_ = other.element_;
    return *this;
  }

  bool operator==(Iterator const &other) {
    return element_ == other.element_;
  }

  bool operator!=(Iterator const &other) {
    return !(*this == other);
  }

  virtual Iterator& operator++() {
    this->element_++;
    return *this;
  }

  Iterator& operator++(int) {
    Iterator temp(*this);
    ++(*this);
    return temp;
  }

  Type const& operator*() {
    return *element_;
  }

  Type const* operator->() {
    return element_;
  }

protected:

  Type const *element_;

};

template <typename Type>
class Container {

public:

  virtual ~Container() {}

  virtual Iterator<Type> begin() const =0;
  virtual Iterator<Type> end() const =0;

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_CONTAINER_H_