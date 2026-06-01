/// \file NavTuple.h
/// \author Andrei Gheata (andrei.gheata@cern.ch)
/// \date 27.05.2026

#ifndef VECGEOM_NAVIGATION_NAVTUPLE_H_
#define VECGEOM_NAVIGATION_NAVTUPLE_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Global.h"

#include <ostream>

namespace vecgeom {

/**
 * @brief Fixed-size tuple of scene-local navigation-table indices.
 *
 * @details The tuple stores one encoded table index for each active scene. The
 * current scene is at `fLevel`; `Top()` returns the index in that scene.
 */
template <uint MAX_DEPTH>
struct NavTuple {
  NavIndex_t fNavInd[MAX_DEPTH]{0};
  uint fLevel{0};

  /// @brief Construct an outside tuple.
  /// @details The tuple is initialized with level zero and top index zero,
  /// which is the outside-state encoding.
  NavTuple() = default;

  /// @brief Copy a tuple.
  /// @details Only entries up to the active tuple level are copied.
  VECCORE_ATT_HOST_DEVICE
  NavTuple(NavTuple<MAX_DEPTH> const &other) : fLevel{other.fLevel}
  {
    for (uint i = 0; i <= fLevel; ++i)
      fNavInd[i] = other.fNavInd[i];
  }

  /// @brief Construct a single-component tuple.
  /// @details The tuple level is zero and the first component is initialized to
  /// the provided scene-local navigation-table index.
  VECCORE_ATT_HOST_DEVICE
  NavTuple(NavIndex_t ind) { fNavInd[0] = ind; }

  /// @brief Construct a tuple from a container of table indices.
  /// @details Container entries are copied in order and must fit the fixed
  /// tuple depth.
  template <typename Container>
  VECCORE_ATT_HOST_DEVICE NavTuple(Container const *cont)
  {
    VECGEOM_ASSERT(cont->size() <= MAX_DEPTH);
    for (NavIndex_t navind : *cont)
      fNavInd[fLevel++] = navind;
    if (fLevel > 0) fLevel--;
  }

  /// @brief Return the fixed tuple capacity.
  /// @details This is the maximum number of scene components that can be stored
  /// in the tuple object.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static constexpr uint GetMaxDepth() { return MAX_DEPTH; }

  /// @brief Assign another tuple.
  /// @details Only entries up to the active tuple level are copied.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavTuple<MAX_DEPTH> &operator=(NavTuple<MAX_DEPTH> const &other)
  {
    fLevel = other.fLevel;
    for (uint i = 0; i <= fLevel; ++i)
      fNavInd[i] = other.fNavInd[i];
    return *this;
  }

  /// @brief Assign a single-level navigation index.
  /// @details The tuple level is reset to zero and the first component is set
  /// to `ind`.
  /// @param ind Navigation index at first level.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavTuple<MAX_DEPTH> &operator=(NavIndex_t ind)
  {
    fLevel     = 0;
    fNavInd[0] = ind;
    return *this;
  }

  /// @brief Return a tuple component.
  /// @details Out-of-range access is reported through validation and returns
  /// zero when validation is compiled as non-fatal.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t operator[](uint i) const
  {
    VECGEOM_VALIDATE(i < MAX_DEPTH, << "NavTuple::operator[] out of range");
    return (i < MAX_DEPTH) ? fNavInd[i] : 0;
  }

  /// @brief Return a mutable tuple component.
  /// @details The caller is responsible for passing an index below
  /// `GetMaxDepth()`.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t &operator[](uint i) { return fNavInd[i]; }

  /// @brief Compare two tuples for exact equality.
  /// @details The active tuple level and all active components must match.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool operator==(NavTuple<MAX_DEPTH> const &other) const
  {
    if (fLevel != other.fLevel) return false;
    for (int i = fLevel; i >= 0; i--) {
      if (fNavInd[i] != other.fNavInd[i]) return false;
    }
    return true;
  }

  /// @brief Compare two tuples for inequality.
  /// @details This is the negation of exact tuple equality.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool operator!=(NavTuple<MAX_DEPTH> const &other) const { return !operator==(other); }

  /// @brief Order tuples lexicographically.
  /// @details This ordering is used by helper algorithms that need tuple keys;
  /// it does not encode geometry containment by itself.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool operator<(NavTuple<MAX_DEPTH> const &other) const
  {
    if (fLevel == other.fLevel) {
      for (unsigned i = 0; i <= fLevel; ++i) {
        if (fNavInd[i] < other.fNavInd[i]) return true;
        if (fNavInd[i] > other.fNavInd[i]) return false;
      }
    } else {
      return fLevel < other.fLevel;
    }
    return false;
  }

  /// @brief Compare with a single table index.
  /// @details The tuple is equal to the index only when it has level zero and
  /// its first component matches.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool operator==(NavIndex_t navind) const { return fLevel == 0 && fNavInd[0] == navind; }

  /// @brief Compare with a single table index for inequality.
  /// @details This is the negation of single-index equality.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool operator!=(NavIndex_t navind) const { return !operator==(navind); }

  /// @brief Reset the tuple to outside.
  /// @details The active level is set to zero and the first component is set to
  /// zero.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Clear()
  {
    fLevel     = 0;
    fNavInd[0] = 0;
  }

  /// @brief Return whether this tuple encodes outside.
  /// @details Outside is represented by level zero and top component zero.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsOutside() const { return (fLevel == 0) && (fNavInd[0] == 0); }

  /// @brief Push a new scene component.
  /// @details For a non-outside tuple, the active level is incremented before
  /// storing the new scene-local index. For outside, the first component is set.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Push(NavIndex_t value)
  {
    if (!IsOutside()) fLevel++;
    VECGEOM_VALIDATE(fLevel < MAX_DEPTH, << "NavTuple::Push out of range");
    if (fLevel < MAX_DEPTH) fNavInd[fLevel] = value;
  }

  /// @brief Set the active scene component.
  /// @details The active tuple level is unchanged.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Set(NavIndex_t value)
  {
    VECGEOM_VALIDATE(fLevel < MAX_DEPTH, << "NavTuple::Set out of range");
    if (fLevel < MAX_DEPTH) fNavInd[fLevel] = value;
  }

  /// @brief Return the active scene component.
  /// @details This is the scene-local navigation-table index at `fLevel`.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t Top() const
  {
    VECGEOM_VALIDATE(fLevel < MAX_DEPTH, << "NavTuple::Top out of range");
    return (fLevel < MAX_DEPTH) ? fNavInd[fLevel] : 0;
  }
};

/// @brief Print a tuple to a stream.
/// @details This diagnostic helper prints active tuple components in order.
template <unsigned int MAX_DEPTH>
std::ostream &operator<<(std::ostream &os, NavTuple<MAX_DEPTH> const &nav_tuple)
{
  os << "(" << nav_tuple[0];
  for (unsigned i = 1; i < nav_tuple.fLevel; ++i)
    os << " ," << nav_tuple[i];
  os << ")";
  return os;
}

using NavTuple_t = NavTuple<VECGEOM_NAVTUPLE_MAXDEPTH>;

} // namespace vecgeom

#endif // VECGEOM_NAVIGATION_NAVTUPLE_H_
