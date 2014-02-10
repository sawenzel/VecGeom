#ifndef VECGEOM_BASE_LIST_H_
#define VECGEOM_BASE_LIST_H_

#include <list>
#include "base/container.h"

namespace vecgeom {

template <typename Type>
class List : public Container<Type> {

private:

  std::list<Type> list;

public:

  VECGEOM_CUDA_HEADER_HOST
  List() {}

  VECGEOM_CUDA_HEADER_HOST
  VECGEOM_INLINE
  Type& operator[](const int index) {
    return list[index];
  }

  VECGEOM_CUDA_HEADER_HOST
  VECGEOM_INLINE
  Type const& operator[](const int index) const {
    return list[index];
  }

  VECGEOM_CUDA_HEADER_HOST
  VECGEOM_INLINE
  int Size() const {
    return list.size();
  }

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_LIST_H_