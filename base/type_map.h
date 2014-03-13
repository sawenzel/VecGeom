#ifndef VECGEOM_BASE_TYPEMAP_H_
#define VECGEOM_BASE_TYPEMAP_H_

#include <map>

#include "base/global.h"

#include "base/iterator.h"

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

  TypeB const& operator[](TypeA const &a) const {
    const typename std::map<TypeA, TypeB>::const_iterator i = a_to_b.find(a);
    assert(i != a_to_b.end());
    return i->second;
  }

  TypeA const& operator[](TypeB const &b) const {
    const typename std::map<TypeB, TypeA>::const_iterator i = b_to_a.find(b);
    assert(i != b_to_a.end());
    return i->second;
  }

  void Set(TypeA const &a, TypeB const &b) {
    a_to_b[a] = b;
    b_to_a[b] = a;
  }

  void Set(TypeB const &b, TypeA const &a) {
    a_to_b[a] = b;
    b_to_a[b] = a;
  }

  typename std::map<TypeA, TypeB>::const_iterator begin() const {
    return a_to_b.begin();
  }

  typename std::map<TypeA, TypeB>::const_iterator end() const {
    return a_to_b.begin();
  }

  bool Contains(TypeA const &a) const {
    return a_to_b.find(a) != a_to_b.end();
  }

  bool Contains(TypeB const &b) const {
    return b_to_a.find(b) != b_to_a.end();
  }

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_TYPEMAP_H_