#ifndef VECGEOM_BASE_VECTOR_H_
#define VECGEOM_BASE_VECTOR_H_

#include <vector>
#include "base/container.h"

namespace vecgeom {

template <typename Type>
class Vector : public Container<Type> {

private:

  std::vector<Type> vec;

public:

  VECGEOM_INLINE
  Type& operator[](const int index) {
    return vec[index];
  }

  VECGEOM_INLINE
  Type const& operator[](const int index) const {
    return vec[index];
  }

  VECGEOM_INLINE
  int size() const {
    return vec.size();
  }

  VECGEOM_INLINE
  void push_back(Type const &item) {
    vec.push_back(item);
  }

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_CONTAINER_H_