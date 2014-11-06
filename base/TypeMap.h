/// \file TypeMap.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_TYPEMAP_H_
#define VECGEOM_BASE_TYPEMAP_H_

#include "base/Global.h"

#include <cassert>
#include <map>

namespace VECGEOM_NAMESPACE {

/**
 * @brief Bidirectional map between two distinct types.
 * @details Implemented to accommodate mapping between two representations of
 *          geometry without having to store pointers in the geometry objects
 *          themselves.
 */
template <typename TypeA, typename TypeB>
class BidirectionalTypeMap {

private:

  std::map<TypeA, TypeB> a_to_b;
  std::map<TypeB, TypeA> b_to_a;

public:

  BidirectionalTypeMap() : a_to_b(), b_to_a() {}
  BidirectionalTypeMap(BidirectionalTypeMap const &other);
  BidirectionalTypeMap& operator=(BidirectionalTypeMap const &other);

  /**
   * Lookup variable of second type using variable of first type. The accessed
   * element is asserted to exist, crashing if it isn't found.
   * @return Constant reference to element. Unlike std::map, changes to content
   *         should be done using the Set() function.
   */
  TypeB const& operator[](TypeA const &a) const {
    const typename std::map<TypeA, TypeB>::const_iterator i = a_to_b.find(a);
    // Crash if not found. To prevent insertion of new elements by lookup.
    assert(i != a_to_b.end());
    return i->second;
  }

  /**
   * Lookup variable of first type using variable of second type. The accessed
   * element is asserted to exist, crashing if it isn't found.
   * @return Constant reference to element. Unlike std::map, changes to content
   *         should be done using the Set() function.
   */
  TypeA const& operator[](TypeB const &b) const {
    const typename std::map<TypeB, TypeA>::const_iterator i = b_to_a.find(b);
    // Crash if not found. To prevent insertion of new elements by lookup.
    assert(i != b_to_a.end());
    return i->second;
  }

  /**
   * Maps the two arguments to each other. The order of which arguments are
   * passed is inconsequential.
   */
  void Set(TypeA const &a, TypeB const &b) {
    a_to_b[a] = b;
    b_to_a[b] = a;
  }

  /**
   * Maps the two arguments to each other. The order of which arguments are
   * passed is inconsequential.
   */
  void Set(TypeB const &b, TypeA const &a) {
    a_to_b[a] = b;
    b_to_a[b] = a;
  }

  /**
   * @return Constant iterator to the beginning of the map from the first type
   *         to the second. Values of both can be accessed by accessing the
   *         fields ->first and ->second of iterator.
   */
  typename std::map<TypeA, TypeB>::const_iterator begin() const {
    return a_to_b.begin();
  }

  /**
   * @return Constant iterator to the end of the map from the first type
   *         to the second. Values of both can be accessed by accessing the
   *         fields ->first and ->second of iterator.
   */
  typename std::map<TypeA, TypeB>::const_iterator end() const {
    return a_to_b.end();
  }

    /**
     * @return True is the passed element is contined in the bidirectional map,
     *         false if it isn't.
     */
    bool Contains(TypeA const &a) const {
      return a_to_b.find(a) != a_to_b.end();
    }

    /**
     * @return True is the passed element is contined in the bidirectional map,
     *         false if it isn't.
     */
    bool Contains(TypeB const &b) const {
      return b_to_a.find(b) != b_to_a.end();
    }

    /**
     * remove any content from the containers
     */
    void Clear()
    {
        b_to_a.clear();
        a_to_b.clear();
    }
};

} // End global namespace

#endif // VECGEOM_BASE_TYPEMAP_H_
