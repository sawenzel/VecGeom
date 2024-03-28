/// \file NavStateTuple.h
/// \author Andrei Gheata (andrei.gheata@cern.ch)
/// \date 20.06.2023

#ifndef VECGEOM_NAVIGATION_NAVSTATETUPLE_H_
#define VECGEOM_NAVIGATION_NAVSTATETUPLE_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/management/GeoManager.h"

#ifdef VECGEOM_ENABLE_CUDA
#include "VecGeom/management/CudaManager.h"
#endif

#include <iostream>
#include <list>
#include <sstream>

namespace vecgeom {

template <uint MAX_DEPTH>
struct NavTuple {
  NavIndex_t fNavInd[MAX_DEPTH]{0};
  uint fLevel{0};

  NavTuple() = default;

  VECCORE_ATT_HOST_DEVICE
  NavTuple(NavTuple<MAX_DEPTH> const &other) : fLevel{other.fLevel}
  {
    for (uint i = 0; i <= fLevel; ++i)
      fNavInd[i] = other.fNavInd[i];
  }

  VECCORE_ATT_HOST_DEVICE
  NavTuple(NavIndex_t ind) { fNavInd[0] = ind; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static constexpr uint GetMaxDepth() { return MAX_DEPTH; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavTuple<MAX_DEPTH> &operator=(NavTuple<MAX_DEPTH> const &other)
  {
    fLevel = other.fLevel;
    for (uint i = 0; i <= fLevel; ++i)
      fNavInd[i] = other.fNavInd[i];
    return *this;
  }

  /// @brief Assign a single-level navigation index
  /// @param ind Navigation index at first level
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavTuple<MAX_DEPTH> &operator=(NavIndex_t ind)
  {
    fLevel     = 0;
    fNavInd[0] = ind;
    return *this;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t operator[](uint i) const { return fNavInd[i]; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t &operator[](uint i) { return fNavInd[i]; }

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

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool operator!=(NavTuple<MAX_DEPTH> const &other) const { return !operator==(other); }

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

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool operator==(NavIndex_t navind) const { return fLevel == 0 && fNavInd[0] == navind; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool operator!=(NavIndex_t navind) const { return !operator==(navind); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Clear()
  {
    fLevel     = 0;
    fNavInd[0] = 0;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsOutside() const { return (fLevel == 0) && (fNavInd[0] == 0); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Push(NavIndex_t value)
  {
    if (!IsOutside()) fLevel++;
    fNavInd[fLevel] = value;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Set(NavIndex_t value) { fNavInd[fLevel] = value; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t Top() const { return fNavInd[fLevel]; }
};

template <unsigned int MAX_DEPTH>
std::ostream &operator<<(std::ostream &os, NavTuple<MAX_DEPTH> const &nav_tuple)
{
  os << "(" << nav_tuple[0];
  for (unsigned i = 1; i < nav_tuple.fLevel; ++i)
    os << " ," << nav_tuple[i];
  os << ")";
  return os;
}

using NavTuple_t = NavTuple<NAVTUPLE_MAXDEPTH>;

/**
 * @brief A class describing a current geometry state based on a tuple of indices
 */
class NavStateTuple {
public:
  using Value_t = NavTuple_t;

private:
  NavTuple_t fNavTuple{0};   ///< Navigation state tuple
  NavTuple_t fLastExited{0}; ///< Navigation state index of the last exited state
  bool fOnBoundary = false;  ///< flag indicating whether track is on boundary of the "Top()" placed volume

public:
  VECCORE_ATT_HOST_DEVICE
  NavStateTuple(NavTuple_t nav_tpl = 0) : fNavTuple(nav_tpl) {}

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static unsigned char GetMaxLevel()
  {
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
    return vecgeom::globaldevicegeomdata::gMaxDepth;
#else
    return (unsigned char)GeoManager::Instance().getMaxDepth();
#endif
  }

  // Static accessors
  VECCORE_ATT_HOST_DEVICE
  static NavStateTuple *MakeInstance(int)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    return new NavStateTuple();
  }

  VECCORE_ATT_HOST_DEVICE
  static NavStateTuple *MakeCopy(NavStateTuple const &other)
  {
    return new NavStateTuple(other);
  }

  VECCORE_ATT_HOST_DEVICE
  static NavStateTuple *MakeInstanceAt(int, void *addr)
  {
    return new (addr) NavStateTuple();
  }

  VECCORE_ATT_HOST_DEVICE
  static NavStateTuple *MakeCopy(NavStateTuple const &other, void *addr)
  {
    return new (addr) NavStateTuple(other);
  }

  VECCORE_ATT_HOST_DEVICE
  static void ReleaseInstance(NavStateTuple *state)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    delete state;
  }

  // returns the size in bytes of a NavStateTuple object with internal
  // path depth maxlevel
  VECCORE_ATT_HOST_DEVICE
  static size_t SizeOfInstance(int)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    return sizeof(NavStateTuple);
  }

  // returns the size in bytes of a NavStateIndex object with internal
  // path depth maxlevel -- including space needed for padding to next aligned object
  // of same kind
  VECCORE_ATT_HOST_DEVICE
  static size_t SizeOfInstanceAlignAware(int)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    return sizeof(NavStateTuple);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  NavIndex_t GetNavIndex() const
  {
    return fNavTuple.Top();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  NavTuple_t const &GetState() const
  {
    return fNavTuple;
  }

  VECCORE_ATT_HOST_DEVICE
  int GetObjectSize() const
  {
    return (int)sizeof(NavStateTuple);
  }

  VECCORE_ATT_HOST_DEVICE
  static size_t SizeOf(size_t)
  {
    return sizeof(NavStateTuple);
  }

  VECCORE_ATT_HOST_DEVICE
  void CopyTo(NavStateTuple *other) const
  {
    *other = *this;
  }

  // copies a fixed and predetermined number of bytes
  // might be useful for specialized navigators which know the depth + SizeOf in advance
  // N is number of bytes to be copied and can be obtained by a prior call to constexpr NavStateIndex::SizeOf( ... );
  template <size_t N>
  void CopyToFixedSize(NavStateTuple *other) const
  {
    *other = *this;
  }

  /// @brief Returns the navigation index address in the table. This is needed for calculating
  //         the addresses of components associated with nav_ind
  /// @param nav_ind Navigation index
  /// @return Start address fir the meta information associated with the index
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static NavIndex_t const *NavIndAddr(NavIndex_t nav_ind)
  {
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
    // checking here for NVCC_DEVICE since the global variable globaldevicegeomgata::gCompact...
    // is marked __device__ and can only be compiled within device compiler passes
    assert(vecgeom::globaldevicegeomdata::gNavIndex != nullptr);
    return &vecgeom::globaldevicegeomdata::gNavIndex[nav_ind];
#else
#ifndef VECCORE_CUDA
    assert(vecgeom::GeoManager::gNavIndex != nullptr);
    return &vecgeom::GeoManager::gNavIndex[nav_ind];
#else
    // this is the case when we compile with nvcc for host side
    // (failed previously due to undefined symbol vecgeom::cuda::GeoManager::gCompactPlacedVolBuffer)
    assert(false && "reached unimplement code");
    return nullptr;
#endif
#endif
  }

  /// @brief Returns the navigation index stored at a given index in the table
  /// @param i Index in the navigation table
  /// @return Navigation index stored at the requested index
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static NavIndex_t NavInd(NavIndex_t i)
  {
    return *NavIndAddr(i);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static VPlacedVolume const *ToPlacedVolume(size_t index)
  {
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
    // checking here for NVCC_DEVICE since the global variable globaldevicegeomgata::gCompact...
    // is marked __device__ and can only be compiled within device compiler passes
    assert(vecgeom::globaldevicegeomdata::gCompactPlacedVolBuffer != nullptr);
    return &vecgeom::globaldevicegeomdata::gCompactPlacedVolBuffer[index];
#else
#ifndef VECCORE_CUDA
    assert(vecgeom::GeoManager::gCompactPlacedVolBuffer == nullptr ||
           vecgeom::GeoManager::gCompactPlacedVolBuffer[index].id() == index);
    return &vecgeom::GeoManager::gCompactPlacedVolBuffer[index];
#else
    // this is the case when we compile with nvcc for host side
    // (failed previously due to undefined symbol vecgeom::cuda::GeoManager::gCompactPlacedVolBuffer)
    assert(false && "reached unimplement code");
    (void)index; // avoid unused parameter warning.
    return nullptr;
#endif
#endif
  }

  /// @brief Implementation for getting the number of daughters for a given navigation index
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned int GetLogicalIdImpl(NavIndex_t nav_index)
  {
    if (!nav_index) return 0;
    return NavInd(NavInd(nav_index + 3));
  }

  /// @brief Implementation for getting the number of daughters for a given navigation index
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned int GetLogicalIdImpl(NavTuple_t const &nav_tuple)
  {
    return GetLogicalIdImpl(nav_tuple.Top());
  }

  /// @brief Implementation for getting the number of daughters for a given navigation tuple
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned int GetNdaughtersImpl(NavTuple_t const &nav_tuple)
  {
    return NavInd(NavInd(nav_tuple.Top() + 3) + 1);
  }

  /// @brief Implementation for getting the scene id for a given navigation index
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static bool GetSceneIdImpl(NavIndex_t nav_ind, unsigned short &scene_id, unsigned short &newscene_id)
  {
    scene_id    = 0;
    newscene_id = 0;
    if (nav_ind == 0) return false;
    auto scenes = reinterpret_cast<const unsigned short *>(NavIndAddr(nav_ind + 4));
    scene_id    = scenes[0];
    newscene_id = scenes[1];
    return (newscene_id > scene_id);
  }

  /// @brief Implementation for getting the scene id for a given navigation tuple
  /// @return True if the state points to a scene volume
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static bool GetSceneIdImpl(NavTuple_t const &nav_tuple, unsigned short &scene_id, unsigned short &newscene_id)
  {
    auto top = nav_tuple.Top();
    if (top == 0) {
      scene_id    = GetParentNewSceneImpl(nav_tuple);
      newscene_id = scene_id;
      return true;
    }
    return GetSceneIdImpl(nav_tuple.Top(), scene_id, newscene_id);
  }

  /// @brief Implementation for getting if a given tuple points to a scene volume
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static bool IsSceneImpl(NavTuple_t const &nav_tuple)
  {
    unsigned short scene_id = 0, newscene_id = 0;
    return GetSceneIdImpl(nav_tuple.Top(), scene_id, newscene_id);
  }

  /// @brief Implementation for getting the scene level for a given navigation tuple
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned int GetSceneLevelImpl(NavTuple_t const &nav_tuple)
  {
    return nav_tuple.fLevel;
  }

  /// @brief Implementation for getting the parent scene id
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned short GetParentSceneImpl(NavTuple_t const &nav_tuple)
  {
    if (nav_tuple.fLevel < 1) return 0;
    auto top_parent = nav_tuple[nav_tuple.fLevel - 1];
    return *reinterpret_cast<const unsigned short *>(NavIndAddr(top_parent + 4));
  }

  /// @brief Implementation for getting the parent scene id
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned short GetParentNewSceneImpl(NavTuple_t const &nav_tuple)
  {
    if (nav_tuple.fLevel < 1) return 0;
    auto top_parent = nav_tuple[nav_tuple.fLevel - 1];
    auto scenes     = reinterpret_cast<const unsigned short *>(NavIndAddr(top_parent + 4));
    return scenes[1];
  }

  /// @brief Implementation for getting the local level associated with a navigation index in the current scene
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static unsigned char GetLevelImpl(NavIndex_t nav_ind)
  {
    auto content_level = reinterpret_cast<const unsigned char *>(NavIndAddr(nav_ind + 5));
    return *content_level;
  }

  /// @brief Implementation for getting the global level associated with a navigation tuple
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static unsigned char GetLevelImpl(NavTuple_t const &nav_tuple)
  {
    int level = 0;
    for (uint ituple = 0; ituple <= nav_tuple.fLevel; ++ituple) {
      level += GetLevelImpl(nav_tuple[ituple]);
    }
    return level;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static NavTuple_t GetNavTupleImpl(NavTuple_t nav_tuple, int level)
  {
    int level_max = GetLevelImpl(nav_tuple);
    int up        = level_max - level;
    assert(up >= 0 && "called GetNavIndexImpl for level larger than current level");
    NavIndex_t mother = nav_tuple.Top();
    while (up--) {
      mother = NavInd(mother);
      if (mother == 0) {
        if (nav_tuple.fLevel == 0) break;
        mother = nav_tuple[--nav_tuple.fLevel];
      }
    }
    nav_tuple.Set(mother);
    return nav_tuple;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static NavIndex_t GetIdImpl(NavIndex_t nav_ind)
  {
    return (nav_ind > 0) ? NavInd(nav_ind + 2) : 0;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static bool IsDescendentImpl(NavTuple_t const &child_ind, NavTuple_t const &parent_ind)
  {
    NavTuple_t ind = child_ind;
    while (parent_ind < ind) {
      PopImpl(ind);
      if (ind == parent_ind) return true;
    }
    return false;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static bool IsDescendentImpl(NavIndex_t child_ind, NavIndex_t parent_ind)
  {
    // Only search inside a scene
    auto ind = child_ind;
    while (ind > parent_ind) {
      ind = NavInd(ind);
      if (ind == parent_ind) return true;
    }
    return false;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PopImpl(NavTuple_t &nav_tuple)
  {
    auto top = nav_tuple.Top();
    if (!top) return;
    top = NavInd(top);
    nav_tuple.Set(top);
    if (!top && nav_tuple.fLevel) nav_tuple.fLevel--;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PopImpl(NavIndex_t &nav_index)
  {
    if (!nav_index) return;
    nav_index = NavInd(nav_index);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PushImpl(NavTuple_t &nav_tuple, VPlacedVolume const *v)
  {
    auto top = nav_tuple.Top();
    if (top) {
      auto child       = NavInd(NavInd(top + 3) + 2 + v->GetChildId());
      bool on_newscene = NavInd(child) == 0;
      if (on_newscene) {
        nav_tuple.fLevel++;
        assert(nav_tuple.fLevel < NavTuple_t::GetMaxDepth());
      }
      nav_tuple.Set(child);
    } else {
      nav_tuple.Set(1);
    }
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static VPlacedVolume const *TopImpl(NavTuple_t const &nav_tuple)
  {
    auto top = nav_tuple.Top();
    return (top > 0) ? ToPlacedVolume(NavInd(top + 1)) : nullptr;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void ReadTransformation(NavIndex_t nav_ind, Transformation3D &trans)
  {
    assert(trans.IsIdentity() && "ReadTransformation: destination must be an identity");
    auto record       = NavIndAddr(nav_ind);
    auto content_lhtr = reinterpret_cast<const unsigned char *>(record + 5);
    bool has_trans    = *(content_lhtr + 2) > 0;
    bool has_rot      = *(content_lhtr + 3) > 0;
    if (!(has_trans | has_rot)) return; // identity
    auto offset  = *(content_lhtr + 1);
    auto address = reinterpret_cast<const Precision *>(record + offset);
    assert(reinterpret_cast<uintptr_t>(address) % sizeof(Precision) == 0 &&
           "ReadTransformation: transformation storage not aligned");
    trans.Set(address, address + int{has_trans} * 3, has_trans, has_rot);
  }

  VECCORE_ATT_HOST_DEVICE
  static void TopMatrixImpl(NavTuple_t const &nav_tuple, Transformation3D &trans)
  {
    // Get the multiplication of all parent scenes transformations, then multiply with the current
    // m_0 * m_1 * ... * m_n
    auto top = nav_tuple.Top();
    if (!top) return;
    ReadTransformation(top, trans);
    for (int ituple = nav_tuple.fLevel - 1; ituple >= 0; --ituple) {
      Transformation3D scene_trans;
      top = nav_tuple[ituple];
      ReadTransformation(top, scene_trans);
      trans *= scene_trans;
    }
  }

  VECCORE_ATT_HOST_DEVICE
  static void TopInSceneMatrixImpl(NavTuple_t const &nav_tuple, Transformation3D &trans)
  {
    // Get transformation of the node in the top scene
    auto top = nav_tuple.Top();
    if (!top) return;
    ReadTransformation(top, trans);
  }

  VECCORE_ATT_HOST_DEVICE
  static void SceneMatrixImpl(NavTuple_t const &nav_tuple, Transformation3D &trans)
  {
    // Get the top scene transformation
    auto top = nav_tuple.Top();
    if (!top) return;
    for (int ituple = nav_tuple.fLevel - 1; ituple >= 0; --ituple) {
      Transformation3D scene_trans;
      top = nav_tuple[ituple];
      ReadTransformation(top, scene_trans);
      trans *= scene_trans;
    }
  }

  VECCORE_ATT_HOST_DEVICE
  static Vector3D<Precision> GlobalToLocalImpl(NavTuple_t const &nav_tuple, Vector3D<Precision> const &globalpoint)
  {
    Transformation3D trans;
    TopMatrixImpl(nav_tuple, trans);
    Vector3D<Precision> local = trans.Transform(globalpoint);
    return local;
  }

  // Intrerface methods
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *GetLastExited() const
  {
    return TopImpl(fLastExited);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetLastExited()
  {
    fLastExited = fNavTuple;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetNavIndex(NavTuple_t const &nav_tuple)
  {
    fNavTuple = nav_tuple;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetNavIndex(NavIndex_t const &nav_index)
  {
    fNavTuple.Set(nav_index);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned int GetLogicalId() const
  {
    return GetLogicalIdImpl(fNavTuple);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsScene() const
  {
    return IsSceneImpl(fNavTuple);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsDescendent(NavTuple_t const &parent) const
  {
    return IsDescendentImpl(fNavTuple, parent);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned int GetNdaughters() const
  {
    return GetNdaughtersImpl(fNavTuple);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t GetId() const
  {
    return GetIdImpl(fNavTuple.Top());
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t GetParentSceneTopId() const
  {
    return (fNavTuple.fLevel > 0) ? GetIdImpl(fNavTuple[fNavTuple.fLevel - 1]) : 0;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool GetSceneId(unsigned short &scene_id, unsigned short &newscene_id) const
  {
    return GetSceneIdImpl(fNavTuple, scene_id, newscene_id);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned int GetSceneLevel() const
  {
    return GetSceneLevelImpl(fNavTuple);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned short GetParentScene() const
  {
    return GetParentSceneImpl(fNavTuple);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Push(VPlacedVolume const *v)
  {
    PushImpl(fNavTuple, v);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void PushScene(NavIndex_t nav_ind)
  {
    fNavTuple.fLevel++;
    fNavTuple.Set(nav_ind);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Pop()
  {
    PopImpl(fNavTuple);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void PopScene()
  {
    if (fNavTuple.fLevel > 0) fNavTuple.fLevel--;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *Top() const
  {
    return TopImpl(fNavTuple);
  }

  /**
   * returns the number of FILLED LEVELS such that
   * state.GetNode( state.GetLevel() ) == state.Top()
   */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  unsigned char GetLevel() const
  {
    return GetLevelImpl(fNavTuple);
  }

  /** Compatibility getter for NavigationState interface */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  unsigned char GetCurrentLevel() const
  {
    return GetLevel() + 1;
  }

  /**
   * Returns the navigation index for a level smaller/equal than the current level.
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  NavTuple_t GetNavTuple(int level) const
  {
    return GetNavTupleImpl(fNavTuple, level);
  }

  /**
   * Returns the placed volume at a evel smaller/equal than the current level.
   */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *At(int level) const
  {
    auto nav_ind = GetNavTupleImpl(fNavTuple, level).Top();
    return (nav_ind > 0) ? ToPlacedVolume(NavInd(nav_ind + 1)) : nullptr;
  }

  /**
   * Returns the index of a placed volume at a evel smaller/equal than the current level.
   */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  size_t ValueAt(int level) const
  {
    auto nav_ind = GetNavTupleImpl(fNavTuple, level).Top();
    return (nav_ind > 0) ? size_t(NavInd(nav_ind + 1)) : 0;
  }

  VECCORE_ATT_HOST_DEVICE
  void TopMatrix(Transformation3D &trans) const
  {
    TopMatrixImpl(fNavTuple, trans);
  }

  VECCORE_ATT_HOST_DEVICE
  void TopInSceneMatrix(Transformation3D &trans) const
  {
    TopInSceneMatrixImpl(fNavTuple, trans);
  }

  VECCORE_ATT_HOST_DEVICE
  void SceneMatrix(Transformation3D &trans) const
  {
    SceneMatrixImpl(fNavTuple, trans);
  }

  VECCORE_ATT_HOST_DEVICE
  void TopMatrix(int tolevel, Transformation3D &trans) const
  {
    TopMatrixImpl(GetNavTupleImpl(fNavTuple, tolevel), trans);
  }

  // returning a "delta" transformation that can transform
  // coordinates given in reference frame of this->Top() to the reference frame of other->Top()
  // simply with otherlocalcoordinate = delta.Transform( thislocalcoordinate )
  VECCORE_ATT_HOST_DEVICE
  void DeltaTransformation(NavStateTuple const &other, Transformation3D &delta) const
  {
    Transformation3D g2;
    Transformation3D g1;
    other.TopMatrix(g2);
    this->TopMatrix(g1);
    delta = g1.Inverse();
    // Trans/rot properties already correctly set
    // g2.SetProperties();
    // delta.SetProperties();
    delta.FixZeroes();
    delta.MultiplyFromRight(g2);
    delta.FixZeroes();
  }

  // VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GlobalToLocal(Vector3D<Precision> const &localpoint) const
  {
    return GlobalToLocalImpl(fNavTuple, localpoint);
  }

  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GlobalToLocal(Vector3D<Precision> const &localpoint, int tolevel) const
  {
    return GlobalToLocalImpl(GetNavTupleImpl(fNavTuple, tolevel), localpoint);
  }

  /**
   * calculates if other navigation state takes a different branch in geometry path or is on same branch
   * ( two states are on same branch if one can connect the states just by going upwards or downwards ( or do nothing
   * ))
   */
  VECCORE_ATT_HOST_DEVICE
  int Distance(NavStateTuple const &other) const
  {
    int lastcommonlevel = -1;
    int thislevel       = GetLevel();
    int otherlevel      = other.GetLevel();
    int maxlevel        = Min(thislevel, otherlevel);

    //  algorithm: start on top and go down until paths split
    for (int i = 0; i < maxlevel + 1; i++) {
      if (this->At(i) == other.At(i)) {
        lastcommonlevel = i;
      } else {
        break;
      }
    }

    return (thislevel - lastcommonlevel) + (otherlevel - lastcommonlevel);
  }

  // returns a string representation of a (relative) sequence of operations/moves
  // that transforms this navigation state into the other navigation state
  // example:
  // state1 = /0/1/1/
  // state2 = /0/2/2/3
  // results in string
  // "/up/horiz/1/down/2/down/3" with 4 operations "up", "horiz", "down", "down"
  // the sequence of moves is the following
  // up: /0/1/1 --> /0/1/
  // horiz/1 : 0/1 --> /0/2 ( == /0/(1+1) )   "we are hopping from daughter 1 to 2 (which corresponds to a step of 1)"
  // down/2 : /0/2 --> /0/2/2   "going further down 2nd daughter"
  // down/3 : /0/2/2/3 --> /0/2/2/3  "going further down 2nd daughter"
  std::string RelativePath(NavStateTuple const &other) const
  {
    int lastcommonlevel = -1;
    int thislevel       = GetLevel();
    int otherlevel      = other.GetLevel();
    int maxlevel        = Min(thislevel, otherlevel);
    std::stringstream str;
    //  algorithm: start on top and go down until paths split
    for (int i = 0; i < maxlevel + 1; i++) {
      if (this->At(i) == other.At(i)) {
        lastcommonlevel = i;
      } else {
        break;
      }
    }

    // paths are the same
    if (thislevel == lastcommonlevel && otherlevel == lastcommonlevel) {
      return std::string("");
    }

    // emit only ups
    if (thislevel > lastcommonlevel && otherlevel == lastcommonlevel) {
      for (int i = 0; i < thislevel - lastcommonlevel; ++i) {
        str << "/up";
      }
      return str.str();
    }

    // emit only downs
    if (thislevel == lastcommonlevel && otherlevel > lastcommonlevel) {
      for (int i = lastcommonlevel + 1; i <= otherlevel; ++i) {
        str << "/down";
        str << "/" << other.ValueAt(i);
      }
      return str.str();
    }

    // mixed case: first up; then down
    if (thislevel > lastcommonlevel && otherlevel > lastcommonlevel) {
      // emit ups
      int level = thislevel;
      for (; level > lastcommonlevel + 1; --level) {
        str << "/up";
      }

      level = lastcommonlevel + 1;
      // emit horiz ( exists when there is a turning point )
      int delta = other.ValueAt(level) - this->ValueAt(level);
      if (delta != 0) str << "/horiz/" << delta;

      level++;
      // emit downs with index
      for (; level <= otherlevel; ++level) {
        str << "/down/" << other.ValueAt(level);
      }
    }
    return str.str();
  }

  // functions useful to "serialize" navigationstate
  // the Vector-of-Indices basically describes the path on the tree taken from top to bottom
  // an index corresponds to a daughter

  void GetPathAsListOfIndices(std::list<uint> &indices) const
  {
    indices.clear();
    if (IsOutside()) return;

    auto nav_tuple = fNavTuple;
    while (nav_tuple.Top() > 1) {
      auto pvol = TopImpl(nav_tuple);
      indices.push_front(pvol->GetChildId());
      PopImpl(nav_tuple);
    }
    // Paths start always with 0
    indices.push_front(0);
  }

  void ResetPathFromListOfIndices(VPlacedVolume const *world, std::list<uint> const &indices)
  {
    // clear current nav state
    Clear();
    auto vol    = world;
    int counter = 0;
    for (auto id : indices) {
      if (counter > 0) vol = vol->GetDaughters().operator[](id);
      Push(vol);
      counter++;
    }
  }

  // replaces the volume pointers from CPU volumes in fPath
  // to the equivalent pointers on the GPU
  // uses the CudaManager to do so
  void ConvertToGPUPointers() {}

  // replaces the pointers from GPU volumes in fPath
  // to the equivalent pointers on the CPU
  // uses the CudaManager to do so
  void ConvertToCPUPointers() {}

  // clear all information
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Clear()
  {
    fNavTuple.Clear();
    fLastExited.Clear();
    fOnBoundary = false;
  }

  VECCORE_ATT_HOST_DEVICE
  void Print() const
  {
    if (fNavTuple.Top() == 0 && fNavTuple.fLevel == 0) {
      printf("navInd=0, id=0, path=outside\n");
      return;
    }
    NavTuple_t nav_tuple;
    auto level = GetLevel();
    printf("navInd=");
    for (unsigned i = 0; i <= fNavTuple.fLevel; ++i) {
      unsigned short scene_id = 0, newscene_id = 0;
      nav_tuple.Push(fNavTuple[i]);
      GetSceneIdImpl(nav_tuple, scene_id, newscene_id);
      printf("s%u:%u", scene_id, fNavTuple[i]);
      if (i < fNavTuple.fLevel) printf(" | ");
    }
    printf(", id=%u, level=%u/%u,  onBoundary=%s, path=<", GetId(), level, GetMaxLevel(),
           (fOnBoundary ? "true" : "false"));
    int last_scene = 0;
    nav_tuple.Clear();
    for (int i = 0; i <= level; ++i) {
      unsigned short scene_id = 0, newscene_id = 0;
      nav_tuple = GetNavTupleImpl(fNavTuple, i);
      GetSceneIdImpl(nav_tuple, scene_id, newscene_id);
      if (scene_id > last_scene) {
        printf(" | ");
        last_scene = scene_id;
      }
#ifndef VECCORE_CUDA
      auto vol = At(i);
      printf("/%s", vol ? vol->GetLabel().c_str() : "TOP_SCENE");
#else
      printf("/%u", nav_tuple.Top());
#endif
    }
    printf(">\n");
  }

  VECCORE_ATT_HOST_DEVICE
  void PrintTop() const
  {
    if (fNavTuple.Top() == 0) {
      printf("navInd=0, id=0, path=outside\n");
      return;
    }
    auto level = GetLevel();
    printf("navInd=");
    auto top                = fNavTuple.Top();
    unsigned short scene_id = 0, newscene_id = 0;
    GetSceneIdImpl(fNavTuple, scene_id, newscene_id);
    printf("s%u:%u", scene_id, top);

    printf(", id=%u, level=%u/%u,  onBoundary=%s, path=<", GetId(), level, GetMaxLevel(),
           (fOnBoundary ? "true" : "false"));
    int top_scene = scene_id;
    for (int i = 0; i <= level; ++i) {
      auto nav_tuple = GetNavTupleImpl(fNavTuple, i);
      GetSceneIdImpl(nav_tuple, scene_id, newscene_id);
      if (scene_id < top_scene) continue;
#ifndef VECCORE_CUDA
      auto vol = At(i);
      printf("/%s", vol ? vol->GetLabel().c_str() : "TOP_SCENE");
#else
      printf("/%u", nav_tuple.Top());
#endif
    }
    printf(">\n");
  }

  VECCORE_ATT_HOST_DEVICE
  void Dump() const
  {
    Print();
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool HasSamePathAsOther(NavStateTuple const &other) const
  {
    return (fNavTuple == other.fNavTuple);
  }

  void printValueSequence(std::ostream & = std::cerr) const;

  // calculates a checksum along the path
  // can be used (as a quick criterion) to see whether 2 states are same
  unsigned long getCheckSum() const
  {
    unsigned long checksum = 0;
    for (uint ituple = 0; ituple <= fNavTuple.fLevel; ++ituple)
      checksum += (unsigned long)fNavTuple[ituple];

    return checksum;
  }

  /**
    function returning whether the point (current navigation state) is outside the detector setup
  */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsOutside() const
  {
    return (fNavTuple.Top() == 0);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsOnBoundary() const
  {
    return fOnBoundary;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetBoundaryState(bool b)
  {
    fOnBoundary = b;
  }
};

/**
 * encodes the geometry path as a concatenated string of ( Value_t ) present in fPath
 */
inline void NavStateTuple::printValueSequence(std::ostream &stream) const
{
  auto level = GetLevel();
  for (int i = 0; i < level + 1; ++i) {
    auto pvol = At(i);
    if (pvol) stream << "/" << ValueAt(i) << "(" << pvol->GetLabel() << ")";
  }
}

} // namespace vecgeom

#endif // VECGEOM_NAVIGATION_NAVSTATETUPLE_H_
