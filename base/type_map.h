#ifndef VECGEOM_BASE_TYPEMAP_H_
#define VECGEOM_BASE_TYPEMAP_H_

#include <map>

#include "base/global.h"

namespace vecgeom {

template <typename TypeA, typename TypeB>
class TypeMap {

private:

  std::map<TypeA, TypeB> a_to_b;
  std::map<TypeB, TypeA> b_to_a;

public:

  TypeMap() {}
  TypeMap(TypeMap const &other);
  TypeMap& operator=(TypeMap const &other);

  TypeB& operator[](TypeA const &a) {
    return a_to_b[a];
  }

  TypeB const& operator[](TypeA const &a) const {
    return a_to_b[a];
  }

  TypeA& operator[](TypeB const &b) {
    return b_to_a[b];
  }

  TypeA const& operator[](TypeB const &b) const {
    return b_to_a[b];
  }

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_TYPEMAP_H_