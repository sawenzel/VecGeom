/// \file NavStateTuple.h
/// \author Andrei Gheata (andrei.gheata@cern.ch)
/// \date 20.06.2023

#ifndef VECGEOM_NAVIGATION_NAVSTATETUPLE_H_
#define VECGEOM_NAVIGATION_NAVSTATETUPLE_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Transformation3DMP.h"
#include "VecGeom/navigation/NavTuple.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/VolumeTree.h"
#include "VecGeom/management/NavIndexTableLayout.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/DeviceGlobals.h"

#include <iostream>
#include <list>
#include <sstream>

namespace vecgeom {

/**
 * @brief Navigation state represented by a tuple of scene-local navigation indices.
 *
 * @details `NavStateTuple` stores the current touchable as a tuple of encoded
 * navigation-table addresses. Each tuple component identifies a scene-local
 * record; descending into a selected scene appends a component instead of
 * expanding the whole repeated subtree into every parent context. This keeps
 * the navigation table smaller for repeated geometries at the cost of a larger
 * fixed-size per-track state.
 *
 * The class implements the common `NavigationState` API shared with
 * `NavStateIndex`: `GetMaxLevel()`, `GetCurrentLevel()`, `At(level)`,
 * `ValueAt(level)`, `Push(...)`, `Pop()`, `Top()`, transform accessors, scene
 * accessors, and outside/boundary flags.
 */
class NavStateTuple {
public:
  using Value_t = NavTuple_t;

private:
  NavTuple_t fNavTuple{0};   ///< Navigation state tuple
  NavTuple_t fLastExited{0}; ///< Navigation state tuple of the last exited state
  bool fOnBoundary = false;  ///< flag indicating whether track is on boundary of the "Top()" placed volume

public:
  /// @brief Construct a state from a navigation tuple.
  /// @details A tuple with level zero and top index zero denotes the outside
  /// state. Non-zero entries address records in the tuple navigation table.
  VECCORE_ATT_HOST_DEVICE
  NavStateTuple(NavTuple_t nav_tpl = 0) : fNavTuple(nav_tpl) {}

  /// @brief Construct a state from a container of scene-local indices.
  /// @details The container form is part of the generic navigation-state API.
  /// Entries are copied into the fixed-depth tuple and must fit
  /// `VECGEOM_NAVTUPLE_MAXDEPTH`.
  template <typename Container>
  VECCORE_ATT_HOST_DEVICE NavStateTuple(Container const *cont) : fNavTuple(cont)
  {
  }

  /// @brief Return the maximum geometry depth.
  /// @details This is the geometry hierarchy depth, not the number of tuple
  /// components. The tuple component limit is `VECGEOM_NAVTUPLE_MAXDEPTH`.
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

  /// @brief Allocate an empty navigation state.
  /// @details Legacy allocation helper retained for existing pool and test
  /// code. New code can construct `NavStateTuple` directly. The depth argument
  /// is ignored because storage is fixed by `VECGEOM_NAVTUPLE_MAXDEPTH`.
  VECCORE_ATT_HOST_DEVICE
  static NavStateTuple *MakeInstance(int) { return new NavStateTuple(); }

  /// @brief Allocate a heap copy of another state.
  /// @details Legacy allocation helper retained for existing callers. New code
  /// can use the copy constructor directly.
  VECCORE_ATT_HOST_DEVICE
  static NavStateTuple *MakeCopy(NavStateTuple const &other) { return new NavStateTuple(other); }

  /// @brief Construct an empty state in caller-provided storage.
  /// @details Legacy placement helper retained for `NavStatePool`. The depth
  /// argument is ignored because object size is fixed by
  /// `VECGEOM_NAVTUPLE_MAXDEPTH`.
  VECCORE_ATT_HOST_DEVICE
  static NavStateTuple *MakeInstanceAt(int, void *addr) { return new (addr) NavStateTuple(); }

  /// @brief Construct a copy in caller-provided storage.
  /// @details Legacy placement helper retained for existing callers. New code
  /// can use placement-new with the copy constructor directly.
  VECCORE_ATT_HOST_DEVICE
  static NavStateTuple *MakeCopy(NavStateTuple const &other, void *addr) { return new (addr) NavStateTuple(other); }

  /// @brief Release a heap-allocated navigation state.
  /// @details Legacy counterpart to `MakeInstance` and `MakeCopy`. New code can
  /// delete directly.
  VECCORE_ATT_HOST_DEVICE
  static void ReleaseInstance(NavStateTuple *state) { delete state; }

  /// @brief Return the size in bytes of one `NavStateTuple` object.
  /// @details Legacy pool helper retained for existing callers. The depth
  /// argument is ignored because storage is fixed by
  /// `VECGEOM_NAVTUPLE_MAXDEPTH`.
  VECCORE_ATT_HOST_DEVICE
  static size_t SizeOfInstance(int) { return sizeof(NavStateTuple); }

  /// @brief Return the aligned size in bytes of one `NavStateTuple` object.
  /// @details `NavStateTuple` has no variable-length trailing storage, so this
  /// legacy pool helper is identical to `SizeOfInstance`.
  VECCORE_ATT_HOST_DEVICE
  static size_t SizeOfInstanceAlignAware(int) { return sizeof(NavStateTuple); }

  /// @brief Return the current scene-local navigation-table index.
  /// @details This is the top component of the tuple, not the complete tuple
  /// state. Use `GetState()` for the full representation-specific state.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  NavIndex_t GetNavIndex() const { return fNavTuple.Top(); }

  /// @brief Return the complete representation-specific state value.
  /// @details For `NavStateTuple`, this is the full fixed-size scene tuple.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  NavTuple_t const &GetState() const { return fNavTuple; }

  /// @brief Return the runtime object size in bytes.
  /// @details This is the fixed `sizeof(NavStateTuple)` value.
  VECCORE_ATT_HOST_DEVICE
  int GetObjectSize() const { return (int)sizeof(NavStateTuple); }

  /// @brief Return the fixed object size in bytes.
  /// @details Legacy fixed-size helper retained for generated navigation code.
  /// The argument is ignored.
  VECCORE_ATT_HOST_DEVICE
  static size_t SizeOf(size_t) { return sizeof(NavStateTuple); }

  /// @brief Copy this state to another `NavStateTuple`.
  /// @details This copies only the compact state fields, not any global table
  /// data referenced by the tuple.
  VECCORE_ATT_HOST_DEVICE
  void CopyTo(NavStateTuple *other) const { *other = *this; }

  /// @brief Copy this state when the caller already knows the fixed size.
  /// @details Legacy generated-code helper. `N` is ignored because
  /// `NavStateTuple` always copies the complete object.
  template <size_t N>
  void CopyToFixedSize(NavStateTuple *other) const
  {
    *other = *this;
  }

  /// @brief Return the address of an encoded table record.
  /// @details On CUDA builds this reads from device geometry globals; on host
  /// builds it reads from the host `GeoManager` table.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static NavIndex_t const *NavIndAddr(NavIndex_t nav_ind)
  {
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
    VECGEOM_ASSERT(vecgeom::globaldevicegeomdata::gNavIndex != nullptr);
    return &vecgeom::globaldevicegeomdata::gNavIndex[nav_ind];
#else
    VECGEOM_ASSERT(vecgeom::GeoManager::gNavIndex != nullptr);
    return &vecgeom::GeoManager::gNavIndex[nav_ind];
#endif
  }

  /// @brief Read one `NavIndex_t` from the navigation table.
  /// @details The argument is a table element offset, not necessarily a record
  /// start.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static NavIndex_t NavInd(NavIndex_t i) { return *NavIndAddr(i); }

  /// @brief Resolve a compact placed-volume id to a placed-volume pointer.
  /// @details Host builds read the compact placed-volume buffer from
  /// `GeoManager`; CUDA builds read the copied device geometry buffer.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static VPlacedVolume const *ToPlacedVolume(size_t index)
  {
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
    VECGEOM_ASSERT(vecgeom::globaldevicegeomdata::gCompactPlacedVolBuffer != nullptr);
    return &vecgeom::globaldevicegeomdata::gCompactPlacedVolBuffer[index];
#else
    VECGEOM_ASSERT(vecgeom::GeoManager::gCompactPlacedVolBuffer == nullptr ||
                   vecgeom::GeoManager::gCompactPlacedVolBuffer[index].id() == index);
    return &vecgeom::GeoManager::gCompactPlacedVolBuffer[index];
#endif
  }

  /// @brief Return the compact placed-volume id of the world.
  /// @details The world record is stored at navigation index 1 in the tuple
  /// table.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static int WorldId() { return NavInd(NavIndexTableLayout::Tuple::kPlacedVolume + 1); }

  /// @brief Resolve a compact placed-volume id to `VolumeTree` metadata.
  /// @details The metadata provides the child id used to descend through
  /// logical-volume daughter blocks.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static vecgeom::PlacedId const &ToPlacedId(size_t iplaced)
  {
    return vecgeom::VolumeTree::Instance().fPlaced[iplaced];
  }

  /// @brief Decode the logical-volume id for a table record.
  /// @details The touchable record points to a shared logical-volume record
  /// that stores the compact logical id.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned int GetLogicalIdImpl(NavIndex_t nav_index)
  {
    return nav_index ? NavInd(NavInd(nav_index + NavIndexTableLayout::Tuple::kLogicalRecord) +
                              NavIndexTableLayout::Tuple::kLogicalVolumeId)
                     : 0;
  }

  /// @brief Decode the logical-volume id for a tuple state.
  /// @details The top tuple component selects the scene-local touchable record
  /// used by the single-index overload.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned int GetLogicalIdImpl(NavTuple_t const &nav_tuple) { return GetLogicalIdImpl(nav_tuple.Top()); }

  /// @brief Decode the child id for a table record.
  /// @details The child id is stored as an `int` in the touchable record's
  /// fixed field block.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static int GetChildIdImpl(NavIndex_t const &nav_index)
  {
    auto content_ichild = reinterpret_cast<const int *>(NavIndAddr(nav_index + NavIndexTableLayout::Tuple::kChildId));
    return *content_ichild;
  }

  /// @brief Decode the child id for a tuple state.
  /// @details A tuple with no top touchable returns -1.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static int GetChildIdImpl(NavTuple_t const &nav_tuple)
  {
    auto top = nav_tuple.Top();
    return top ? GetChildIdImpl(top) : -1;
  }

  /// @brief Decode the daughter count for a table record.
  /// @details The count is stored in the shared logical-volume record
  /// referenced by the touchable record.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned int GetNdaughtersImpl(NavIndex_t const &nav_index)
  {
    return NavInd(NavInd(nav_index + NavIndexTableLayout::Tuple::kLogicalRecord) +
                  NavIndexTableLayout::Tuple::kDaughterCount);
  }

  /// @brief Decode the daughter count for a tuple state.
  /// @details A tuple with no top touchable has zero daughters.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned int GetNdaughtersImpl(NavTuple_t const &nav_tuple)
  {
    auto top = nav_tuple.Top();
    return top ? GetNdaughtersImpl(top) : 0;
  }

  /// @brief Decode scene ids for a table record.
  /// @details The scene field stores parent and current scene ids as two
  /// unsigned-short halves. The return value is true when the record selects a
  /// new scene.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static bool GetSceneIdImpl(NavIndex_t nav_ind, unsigned short &scene_id, unsigned short &newscene_id)
  {
    scene_id    = 0;
    newscene_id = 0;
    if (nav_ind == 0) return false;
    auto scenes = reinterpret_cast<const unsigned short *>(NavIndAddr(nav_ind + NavIndexTableLayout::Tuple::kScenes));
    scene_id    = scenes[NavIndexTableLayout::Tuple::kParentSceneHalf];
    newscene_id = scenes[NavIndexTableLayout::Tuple::kCurrentSceneHalf];
    return (newscene_id != scene_id);
  }

  /// @brief Decode scene ids for a tuple state.
  /// @details Scene placeholders with top index zero inherit the current scene
  /// id from the parent tuple component.
  /// @return True if the state points to a scene transition.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static bool GetSceneIdImpl(NavTuple_t const &nav_tuple, unsigned short &scene_id, unsigned short &newscene_id)
  {
    auto top = nav_tuple.Top();
    if (top == 0) {
      if (nav_tuple.IsOutside()) {
        scene_id = newscene_id = 0;
        return false;
      }
      scene_id    = GetParentNewSceneImpl(nav_tuple);
      newscene_id = scene_id;
      return true;
    }
    return GetSceneIdImpl(nav_tuple.Top(), scene_id, newscene_id);
  }

  /// @brief Report whether a tuple points to a scene transition.
  /// @details This is derived from the tuple scene ids decoded by
  /// `GetSceneIdImpl`.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static bool IsSceneImpl(NavTuple_t const &nav_tuple)
  {
    unsigned short scene_id = 0, newscene_id = 0;
    return GetSceneIdImpl(nav_tuple, scene_id, newscene_id);
  }

  /// @brief Return the tuple scene nesting level.
  /// @details This is the active component index in the fixed-size tuple.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned int GetSceneLevelImpl(NavTuple_t const &nav_tuple) { return nav_tuple.fLevel; }

  /// @brief Decode the parent scene id for a tuple state.
  /// @details The id is read from the tuple component below the active scene,
  /// or zero for the root scene.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned short GetParentSceneImpl(NavTuple_t const &nav_tuple)
  {
    if (nav_tuple.fLevel < 1) return 0;
    auto top_parent = nav_tuple[nav_tuple.fLevel - 1];
    auto scenes =
        reinterpret_cast<const unsigned short *>(NavIndAddr(top_parent + NavIndexTableLayout::Tuple::kScenes));
    return scenes[NavIndexTableLayout::Tuple::kParentSceneHalf];
  }

  /// @brief Decode the current scene id of the parent tuple component.
  /// @details This is used when the active tuple component is a scene
  /// placeholder with top index zero.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned short GetParentNewSceneImpl(NavTuple_t const &nav_tuple)
  {
    if (nav_tuple.fLevel < 1) return 0;
    auto top_parent = nav_tuple[nav_tuple.fLevel - 1];
    auto scenes =
        reinterpret_cast<const unsigned short *>(NavIndAddr(top_parent + NavIndexTableLayout::Tuple::kScenes));
    return scenes[NavIndexTableLayout::Tuple::kCurrentSceneHalf];
  }

  /// @brief Decode the local level for a scene-local table record.
  /// @details The level is stored as a byte in the record's packed metadata.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static unsigned char GetLevelImpl(NavIndex_t nav_ind)
  {
    auto content_level =
        reinterpret_cast<const unsigned char *>(NavIndAddr(nav_ind + NavIndexTableLayout::Tuple::kPacked));
    return content_level[NavIndexTableLayout::Tuple::kLevelByte];
  }

  /// @brief Decode the full-geometry level for a tuple state.
  /// @details Scene-local levels from all active tuple components are summed.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static unsigned char GetLevelImpl(NavTuple_t const &nav_tuple)
  {
    int level = 0;
    for (uint ituple = 0; ituple <= nav_tuple.fLevel; ++ituple) {
      auto nav_ind = nav_tuple[ituple];
      if (nav_ind) level += GetLevelImpl(nav_ind);
    }
    return level;
  }

  /// @brief Return the tuple state for a requested full-geometry level.
  /// @details Scene-local parent links are followed and tuple components are
  /// popped when traversal crosses scene placeholders.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static NavTuple_t GetNavTupleImpl(NavTuple_t nav_tuple, int level)
  {
    int level_max = GetLevelImpl(nav_tuple);
    int up        = level_max - level;
    VECGEOM_VALIDATE(up >= 0, << "called GetNavIndexImpl for level larger than current level");
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

  /// @brief Decode the touchable id from a table record.
  /// @details A zero navigation index returns zero.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static NavIndex_t GetIdImpl(NavIndex_t nav_ind)
  {
    return (nav_ind > 0) ? NavInd(nav_ind + NavIndexTableLayout::Tuple::kTouchableId) : 0;
  }

  /// @brief Test whether one tuple state descends from another.
  /// @details Tuple components and scene-local parent links are followed until
  /// the parent tuple is reached or the paths diverge.
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

  /// @brief Test whether one scene-local table record descends from another.
  /// @details The search follows parent links inside one scene only.
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

  /// @brief Return a child table record by child id.
  /// @details The child record address is read from the logical-volume
  /// daughter block shared by the current touchable.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static NavIndex_t GetChildNavInd(NavIndex_t nav_ind, int ichild)
  {
    return (nav_ind > 0) ? NavInd(NavInd(nav_ind + NavIndexTableLayout::Tuple::kLogicalRecord) +
                                  NavIndexTableLayout::Tuple::kDaughters + ichild)
                         : 0;
  }

  /// @brief Validate basic tuple parent/daughter links.
  /// @details The check verifies that each active scene-local record is linked
  /// from its parent through the expected daughter entry. `nprint` is kept for
  /// signature symmetry with `NavStateIndex`.
  VECCORE_ATT_HOST_DEVICE
  static bool IsValid(NavTuple_t const &nav_tuple, int nprint = 0)
  {
    bool valid = true;
    if (nav_tuple.fLevel == 0 && nav_tuple.Top() <= 1) return valid;
    for (unsigned i = 0; i <= nav_tuple.fLevel; ++i) {
      auto nav_ind = nav_tuple[i];
      if (nav_ind == 0) return false;
      if (nav_ind == 1) continue;
      auto parent = NavInd(nav_ind);
      if (parent > 0) {
        // Check the pointer to nav_ind in parent record
        auto ichild        = GetChildIdImpl(nav_ind);
        auto nav_ind_child = GetChildNavInd(parent, ichild);
        valid &= nav_ind_child == nav_ind;
      }
    }
    return valid;
  }

  /// @brief Move a tuple state to its parent volume.
  /// @details Scene-local parent links are followed. If the parent link is a
  /// scene placeholder, the tuple scene level is decremented.
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

  /// @brief Move a scene-local table index to its parent.
  /// @details A zero index remains zero.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PopImpl(NavIndex_t &nav_index)
  {
    if (!nav_index) return;
    nav_index = NavInd(nav_index);
  }

  /// @brief Descend to a daughter by placed-volume pointer.
  /// @details The volume child id selects an entry in the current
  /// logical-volume daughter block. Entering a selected scene pushes a tuple
  /// component.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PushImpl(NavTuple_t &nav_tuple, VPlacedVolume const *v)
  {
    auto top = nav_tuple.Top();
    if (top) {
      auto child       = NavInd(NavInd(top + NavIndexTableLayout::Tuple::kLogicalRecord) +
                                NavIndexTableLayout::Tuple::kDaughters + v->GetChildId());
      bool on_newscene = NavInd(child) == 0;
      if (on_newscene) {
        nav_tuple.fLevel++;
        VECGEOM_ASSERT(nav_tuple.fLevel < NavTuple_t::GetMaxDepth());
      }
      nav_tuple.Set(child);
    } else {
      nav_tuple.Set(1);
    }
  }

  /// @brief Descend to a daughter by child id.
  /// @details The child id selects an entry in the current logical-volume
  /// daughter block. Entering a selected scene pushes a tuple component.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PushDaughterImpl(NavTuple_t &nav_tuple, int idaughter)
  {
    auto top = nav_tuple.Top();
    if (top) {
      VECGEOM_ASSERT(idaughter >= 0 && idaughter < int(GetNdaughtersImpl(top)));
      auto child       = NavInd(NavInd(top + NavIndexTableLayout::Tuple::kLogicalRecord) +
                                NavIndexTableLayout::Tuple::kDaughters + idaughter);
      bool on_newscene = NavInd(child) == 0;
      if (on_newscene) {
        nav_tuple.fLevel++;
        VECGEOM_ASSERT(nav_tuple.fLevel < NavTuple_t::GetMaxDepth());
      }
      nav_tuple.Set(child);
    } else {
      nav_tuple.Set(1);
    }
  }

  /// @brief Descend to a daughter by compact placed-volume id.
  /// @details The placed-volume id is resolved through `VolumeTree` to recover
  /// the child id used by `PushDaughterImpl`.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PushImpl(NavTuple_t &nav_tuple, int iplaced)
  {
    auto const &pv_ind = ToPlacedId(iplaced);
    PushDaughterImpl(nav_tuple, pv_ind.fChildId);
  }

  /// @brief Resolve the top placed volume for a tuple state.
  /// @details A tuple without a top touchable resolves to `nullptr`.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static VPlacedVolume const *TopImpl(NavTuple_t const &nav_tuple)
  {
    auto top = nav_tuple.Top();
    return (top > 0) ? ToPlacedVolume(NavInd(top + NavIndexTableLayout::Tuple::kPlacedVolume)) : nullptr;
  }

  /// @brief Return the placed world volume.
  /// @details The world record is stored at navigation index 1.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static VPlacedVolume const *World() { return ToPlacedVolume(NavInd(NavIndexTableLayout::Tuple::kPlacedVolume + 1)); }

  /// @brief Resolve the compact placed-volume id for a tuple state.
  /// @details A tuple without a top touchable returns -1.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static int TopIdImpl(NavTuple_t const &nav_tuple)
  {
    auto top = nav_tuple.Top();
    return (top > 0) ? int(NavInd(top + NavIndexTableLayout::Tuple::kPlacedVolume)) : -1;
  }

  /// @brief Read a stored transformation into a `Transformation3D`.
  /// @details Transform coefficients are stored inline in the touchable record
  /// when the packed metadata contains a non-zero transform offset.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void ReadTransformation(NavIndex_t nav_ind, Transformation3D &trans)
  {
    VECGEOM_VALIDATE(trans.IsIdentity(), << "ReadTransformation: destination must be an identity");
    auto record       = NavIndAddr(nav_ind);
    auto content_lhtr = reinterpret_cast<const unsigned char *>(record + NavIndexTableLayout::Tuple::kPacked);
    bool has_trans    = content_lhtr[NavIndexTableLayout::Tuple::kHasTranslationByte] > 0;
    bool has_rot      = content_lhtr[NavIndexTableLayout::Tuple::kHasRotationByte] > 0;
    if (!(has_trans | has_rot)) return; // identity
    auto offset  = content_lhtr[NavIndexTableLayout::Tuple::kTransformOffsetByte];
    auto address = reinterpret_cast<const Precision *>(record + offset);
    VECGEOM_ASSERT(reinterpret_cast<uintptr_t>(address) % sizeof(Precision) == 0 &&
                   "ReadTransformation: transformation storage not aligned");
    trans.Set(address, address + int{has_trans} * 3, has_trans, has_rot);
  }

  /// @brief Read a stored transformation into a multi-precision transform.
  /// @details Transform coefficients are stored in the navigation table as
  /// `Precision` values. `Transformation3DMP<Real_t>::Set` performs the
  /// conversion to the requested arithmetic type.
  template <typename Real_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void ReadTransformation(NavIndex_t nav_ind,
                                                                              Transformation3DMP<Real_t> &trans)
  {
    VECGEOM_VALIDATE(trans.IsIdentity(), << "ReadTransformation: destination must be an identity");
    auto record       = NavIndAddr(nav_ind);
    auto content_lhtr = reinterpret_cast<const unsigned char *>(record + NavIndexTableLayout::Tuple::kPacked);
    bool has_trans    = content_lhtr[NavIndexTableLayout::Tuple::kHasTranslationByte] > 0;
    bool has_rot      = content_lhtr[NavIndexTableLayout::Tuple::kHasRotationByte] > 0;
    if (!(has_trans | has_rot)) return; // identity
    auto offset  = content_lhtr[NavIndexTableLayout::Tuple::kTransformOffsetByte];
    auto address = reinterpret_cast<const Precision *>(record + offset);
    VECGEOM_ASSERT(reinterpret_cast<uintptr_t>(address) % sizeof(Precision) == 0 &&
                   "ReadTransformation: transformation storage not aligned");
    trans.Set(address, address + int{has_trans} * 3, has_trans, has_rot);
  }

  /// @brief Reconstruct the global-to-local transform for a tuple state.
  /// @details The current scene-local transform is multiplied with the
  /// transforms of active parent scenes from innermost to outermost.
  VECCORE_ATT_HOST_DEVICE
  static void TopMatrixImpl(NavTuple_t const &nav_tuple, Transformation3D &trans)
  {
    // Get the multiplication of all parent scenes transformations, then multiply with the current
    // m_0 * m_1 * ... * m_n
    auto top = nav_tuple.Top();
    if (top) ReadTransformation(top, trans);
    for (int ituple = int(nav_tuple.fLevel) - 1; ituple >= 0; --ituple) {
      Transformation3D scene_trans;
      top = nav_tuple[ituple];
      if (!top) continue;
      ReadTransformation(top, scene_trans);
      trans *= scene_trans;
    }
  }

  /// @brief Reconstruct the global-to-local transform for a tuple state.
  /// @details The current scene-local transform is multiplied with the
  /// transforms of active parent scenes from innermost to outermost.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE static void TopMatrixImpl(NavTuple_t const &nav_tuple, Transformation3DMP<Real_t> &trans)
  {
    // Get the multiplication of all parent scenes transformations, then multiply with the current
    // m_0 * m_1 * ... * m_n
    auto top = nav_tuple.Top();
    if (top) ReadTransformation(top, trans);
    for (int ituple = int(nav_tuple.fLevel) - 1; ituple >= 0; --ituple) {
      Transformation3DMP<Real_t> scene_trans;
      top = nav_tuple[ituple];
      if (!top) continue;
      ReadTransformation(top, scene_trans);
      trans *= scene_trans;
    }
  }

  /// @brief Reconstruct the top transform inside the active scene.
  /// @details Only the active scene-local component is read; parent scene
  /// transforms are intentionally not multiplied.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE static void TopInSceneMatrixImpl(NavTuple_t const &nav_tuple,
                                                           Transformation3DMP<Real_t> &trans)
  {
    // Get transformation of the node in the top scene
    auto top = nav_tuple.Top();
    if (!top) return;
    ReadTransformation(top, trans);
  }

  /// @brief Reconstruct the top transform inside the active scene.
  /// @details Only the active scene-local component is read; parent scene
  /// transforms are intentionally not multiplied.
  VECCORE_ATT_HOST_DEVICE
  static void TopInSceneMatrixImpl(NavTuple_t const &nav_tuple, Transformation3D &trans)
  {
    // Get transformation of the node in the top scene
    auto top = nav_tuple.Top();
    if (!top) return;
    ReadTransformation(top, trans);
  }

  /// @brief Reconstruct the transform of the active scene.
  /// @details The transform is built from active parent scene components,
  /// excluding the current top scene-local touchable.
  VECCORE_ATT_HOST_DEVICE
  static void SceneMatrixImpl(NavTuple_t const &nav_tuple, Transformation3D &trans)
  {
    // Get the top scene transformation
    for (int ituple = int(nav_tuple.fLevel) - 1; ituple >= 0; --ituple) {
      Transformation3D scene_trans;
      auto top = nav_tuple[ituple];
      if (!top) continue;
      ReadTransformation(top, scene_trans);
      trans *= scene_trans;
    }
  }

  /// @brief Reconstruct the transform of the active scene.
  /// @details The transform is built from active parent scene components,
  /// excluding the current top scene-local touchable.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE static void SceneMatrixImpl(NavTuple_t const &nav_tuple, Transformation3DMP<Real_t> &trans)
  {
    // Get the top scene transformation
    for (int ituple = int(nav_tuple.fLevel) - 1; ituple >= 0; --ituple) {
      Transformation3DMP<Real_t> scene_trans;
      auto top = nav_tuple[ituple];
      if (!top) continue;
      ReadTransformation(top, scene_trans);
      trans *= scene_trans;
    }
  }

  /// @brief Transform a global point to a tuple state's local frame.
  /// @details The full top transform for `nav_tuple` is reconstructed and
  /// applied to `globalpoint`.
  VECCORE_ATT_HOST_DEVICE
  static Vector3D<Precision> GlobalToLocalImpl(NavTuple_t const &nav_tuple, Vector3D<Precision> const &globalpoint)
  {
    Transformation3D trans;
    TopMatrixImpl(nav_tuple, trans);
    Vector3D<Precision> local = trans.Transform(globalpoint);
    return local;
  }

  /// @brief Transform a global point to a tuple state's local frame.
  /// @details The full top transform for `nav_tuple` is reconstructed and
  /// applied to `globalpoint`.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE static Vector3D<Real_t> GlobalToLocalImpl(NavTuple_t const &nav_tuple,
                                                                    Vector3D<Real_t> const &globalpoint)
  {
    Transformation3DMP<Real_t> trans;
    TopMatrixImpl(nav_tuple, trans);
    Vector3D<Real_t> local = trans.Transform(globalpoint);
    return local;
  }

  /**
   * @name Common NavigationState API
   *
   * @brief Methods consumed by navigators and helper utilities independent of
   * the concrete navigation-state representation.
   *
   * @details Levels are zero-based in the full geometry path: the world is
   * level 0, its daughters are level 1, and `GetCurrentLevel()` returns the
   * number of filled path entries. `ValueAt(level)` returns the compact
   * placed-volume id at that level, not a scene-local table address.
   *
   * Scene methods expose tuple-specific state. `GetNavIndex()` returns the
   * current scene-local index; a value of 0 can denote the top of a parent scene
   * while `IsOutside()` remains false unless the tuple itself is empty.
   */
  /// @{
  /// @brief Return the placed volume stored as last exited.
  /// @details The returned volume is resolved from the stored last-exited
  /// tuple. A null pointer denotes that no exited volume is recorded.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *GetLastExited() const { return TopImpl(fLastExited); }

  /// @brief Return the representation-specific last-exited state.
  /// @details For `NavStateTuple`, this is the full scene-local tuple.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavTuple_t GetLastExitedState() const { return fLastExited; }

  /// @brief Return the compact placed-volume id of the last-exited volume.
  /// @details Returns -1 when no last-exited table index is stored.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  int GetLastIdExited() const { return TopIdImpl(fLastExited); }

  /// @brief Record the current state as last exited.
  /// @details This is used by navigators to preserve boundary-crossing
  /// context.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetLastExited() { fLastExited = fNavTuple; }

  /// @brief Set the last-exited state explicitly.
  /// @details The argument must be a valid tuple state for the current
  /// navigation table, or the outside tuple for none.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetLastExited(NavTuple_t const &nav_tuple) { fLastExited = nav_tuple; }

  /// @brief Replace the current navigation state with a tuple.
  /// @details The tuple entries must address valid scene-local table records,
  /// or represent the outside state.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetNavIndex(NavTuple_t const &nav_tuple) { fNavTuple = nav_tuple; }

  /// @brief Replace the current scene-local index.
  /// @details The value is stored in the top tuple component. This does not
  /// change the tuple scene level.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetNavIndex(NavIndex_t const &nav_index) { fNavTuple.Set(nav_index); }

  /// @brief Return the logical-volume id of the current top volume.
  /// @details The id is decoded through the current touchable record's shared
  /// logical-volume record.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned int GetLogicalId() const { return GetLogicalIdImpl(fNavTuple); }

  /// @brief Return the child id of the current top volume.
  /// @details The child id is the index of this placed volume among its
  /// parent's daughters within the current scene.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int GetChildId() const { return GetChildIdImpl(fNavTuple); }

  /// @brief Report whether the state is at a scene boundary.
  /// @details This is true when the top tuple component represents a selected
  /// scene transition.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsScene() const { return IsSceneImpl(fNavTuple); }

  /// @brief Test whether this state is below the given parent tuple.
  /// @details Tuple components and scene-local parent links are compared until
  /// the parent is reached or the paths diverge.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsDescendent(NavTuple_t const &parent) const { return IsDescendentImpl(fNavTuple, parent); }

  /// @brief Return the number of daughters of the current top volume.
  /// @details The value is decoded from the shared logical-volume record
  /// referenced by the current touchable.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned int GetNdaughters() const { return GetNdaughtersImpl(fNavTuple); }

  /// @brief Return the touchable id of the current state.
  /// @details The id is the builder-assigned touchable id stored in the current
  /// touchable record, not the placed-volume id.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t GetId() const { return GetIdImpl(fNavTuple.Top()); }

  /// @brief Return the top touchable id of the parent scene.
  /// @details The value is decoded from the preceding tuple component when a
  /// parent scene exists; otherwise it is zero.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t GetParentSceneTopId() const
  {
    return (fNavTuple.fLevel > 0) ? GetIdImpl(fNavTuple[fNavTuple.fLevel - 1]) : 0;
  }

  /// @brief Query scene transition ids for this state.
  /// @details The output ids are the parent scene and current scene encoded in
  /// the top touchable record. The return value is true for a scene transition.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool GetSceneId(unsigned short &scene_id, unsigned short &newscene_id) const
  {
    return GetSceneIdImpl(fNavTuple, scene_id, newscene_id);
  }

  /// @brief Return the active tuple scene level.
  /// @details Level zero is the root scene. Higher values correspond to nested
  /// selected scenes stored as extra tuple components.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned int GetSceneLevel() const { return GetSceneLevelImpl(fNavTuple); }

  /// @brief Return the parent scene id.
  /// @details The value is read from the tuple component below the current
  /// scene, or zero at the root scene.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned short GetParentScene() const { return GetParentSceneImpl(fNavTuple); }

  /// @brief Descend to a daughter volume by placed-volume pointer.
  /// @details The daughter child id is taken from the volume and resolved
  /// through the current logical-volume record. Entering a selected scene pushes
  /// a new tuple component.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Push(VPlacedVolume const *v) { PushImpl(fNavTuple, v); }

  /// @brief Descend to a daughter volume by compact placed-volume id.
  /// @details The child id is looked up in `VolumeTree` and then resolved
  /// through the current logical-volume record.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Push(int iplaced) { PushImpl(fNavTuple, iplaced); }

  /// @brief Descend to a daughter volume by child id.
  /// @details The child id is resolved in the current logical-volume record.
  /// Entering a selected scene pushes a new tuple component.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void PushDaughter(int idaughter) { PushDaughterImpl(fNavTuple, idaughter); }

  /// @brief Enter a new scene explicitly.
  /// @details The tuple scene level is incremented and the provided scene-local
  /// index becomes the top component.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void PushScene(NavIndex_t nav_ind)
  {
    fNavTuple.fLevel++;
    fNavTuple.Set(nav_ind);
  }

  /// @brief Move to the parent volume.
  /// @details Scene-local parent links are followed. If this leaves a selected
  /// scene, the tuple scene level is decremented.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Pop() { PopImpl(fNavTuple); }

  /// @brief Leave the current selected scene.
  /// @details This decrements the tuple scene level when one is active.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void PopScene()
  {
    if (fNavTuple.fLevel > 0) fNavTuple.fLevel--;
  }

  /// @brief Return the current top placed volume.
  /// @details The volume pointer is resolved from the top tuple component.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *Top() const { return TopImpl(fNavTuple); }

  /// @brief Return the compact placed-volume id of the current top volume.
  /// @details Returns -1 when the state is outside or at a scene placeholder
  /// without a top touchable.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  int TopId() const { return TopIdImpl(fNavTuple); }

  /// @brief Return the zero-based full-geometry level of the current top volume.
  /// @details The world has level 0. Local scene levels from all tuple
  /// components are accumulated. For the number of filled path entries, use
  /// `GetCurrentLevel()`.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  unsigned char GetLevel() const { return GetLevelImpl(fNavTuple); }

  /// @brief Return the number of filled path entries.
  /// @details This is `GetLevel() + 1` for an in-geometry state. The outside
  /// state is normally tested with `IsOutside()`.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  unsigned char GetCurrentLevel() const { return GetLevel() + 1; }

  /// @brief Return the encoded tuple state at a zero-based path level.
  /// @details The returned tuple identifies the table record for the volume at
  /// the requested full-geometry level.
  /// @param level Zero-based level, where 0 is the world.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  NavTuple_t GetNavTuple(int level) const { return GetNavTupleImpl(fNavTuple, level); }

  /// @brief Return the placed volume at a zero-based path level.
  /// @details The requested full-geometry level is resolved through tuple
  /// components and scene-local parent links.
  /// @param level Zero-based level, where 0 is the world.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *At(int level) const
  {
    auto nav_ind = GetNavTupleImpl(fNavTuple, level).Top();
    return (nav_ind > 0) ? ToPlacedVolume(NavInd(nav_ind + NavIndexTableLayout::Tuple::kPlacedVolume)) : nullptr;
  }

  /// @brief Return the compact placed-volume id at a zero-based path level.
  /// @details The requested full-geometry level is resolved through tuple
  /// components and scene-local parent links.
  /// @param level Zero-based level, where 0 is the world.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  size_t ValueAt(int level) const
  {
    auto nav_ind = GetNavTupleImpl(fNavTuple, level).Top();
    return (nav_ind > 0) ? size_t(NavInd(nav_ind + NavIndexTableLayout::Tuple::kPlacedVolume)) : 0;
  }

  /// @brief Return the global-to-local transform of the current top volume.
  /// @details Transforms are reconstructed by multiplying the current
  /// scene-local transform with the transforms of active parent scenes.
  VECCORE_ATT_HOST_DEVICE
  void TopMatrix(Transformation3D &trans) const { TopMatrixImpl(fNavTuple, trans); }

  /// @brief Return the global-to-local transform of the current top volume.
  /// @details Transforms are reconstructed by multiplying the current
  /// scene-local transform with the transforms of active parent scenes.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void TopMatrix(Transformation3DMP<Real_t> &trans) const
  {
    TopMatrixImpl(fNavTuple, trans);
  }

  /// @brief Return the transform of the current top volume within its scene.
  /// @details Only the top scene-local component is used; parent scene
  /// transforms are not multiplied.
  VECCORE_ATT_HOST_DEVICE
  void TopInSceneMatrix(Transformation3D &trans) const { TopInSceneMatrixImpl(fNavTuple, trans); }

  /// @brief Return the transform of the current top volume within its scene.
  /// @details Only the top scene-local component is used; parent scene
  /// transforms are not multiplied.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void TopInSceneMatrix(Transformation3DMP<Real_t> &trans) const
  {
    TopInSceneMatrixImpl(fNavTuple, trans);
  }

  /// @brief Return the transform of the active scene.
  /// @details The scene transform is reconstructed from the tuple component
  /// that selected the active scene.
  VECCORE_ATT_HOST_DEVICE
  void SceneMatrix(Transformation3D &trans) const { SceneMatrixImpl(fNavTuple, trans); }

  /// @brief Return the transform of the active scene.
  /// @details The scene transform is reconstructed from the tuple component
  /// that selected the active scene.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void SceneMatrix(Transformation3DMP<Real_t> &trans) const
  {
    SceneMatrixImpl(fNavTuple, trans);
  }

  /// @brief Return the global-to-local transform to a requested path level.
  /// @details The requested full-geometry level is resolved to a tuple before
  /// reconstructing the transform.
  VECCORE_ATT_HOST_DEVICE
  void TopMatrix(int tolevel, Transformation3D &trans) const
  {
    TopMatrixImpl(GetNavTupleImpl(fNavTuple, tolevel), trans);
  }

  /// @brief Return the global-to-local transform to a requested path level.
  /// @details The requested full-geometry level is resolved to a tuple before
  /// reconstructing the transform.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void TopMatrix(int tolevel, Transformation3DMP<Real_t> &trans) const
  {
    TopMatrixImpl(GetNavTupleImpl(fNavTuple, tolevel), trans);
  }

  /// @brief Return the transform from this top-volume frame to another state.
  /// @details The resulting delta converts coordinates from `this->Top()` local
  /// frame to `other.Top()` local frame.
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

  /// @brief Return the transform from this top-volume frame to another state.
  /// @details The resulting delta converts coordinates from `this->Top()` local
  /// frame to `other.Top()` local frame.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void DeltaTransformation(NavStateTuple const &other, Transformation3DMP<Real_t> &delta) const
  {
    Transformation3DMP<Real_t> g2;
    Transformation3DMP<Real_t> g1;
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

  /// @brief Transform a global point to the current top-volume frame.
  /// @details The current state's top transform is used.
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GlobalToLocal(Vector3D<Precision> const &localpoint) const
  {
    return GlobalToLocalImpl(fNavTuple, localpoint);
  }

  /// @brief Transform a global point to the frame of a requested path level.
  /// @details The requested full-geometry level is resolved to a tuple before
  /// applying the transform.
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GlobalToLocal(Vector3D<Precision> const &localpoint, int tolevel) const
  {
    return GlobalToLocalImpl(GetNavTupleImpl(fNavTuple, tolevel), localpoint);
  }

  /// @brief Return the path distance to another navigation state.
  /// @details The distance is the number of up/down steps required to move
  /// between the two states through their nearest common ancestor.
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

  /// @brief Return a textual relative path from this state to another state.
  /// @details The string encodes moves such as `/up`, `/horiz/<delta>`, and
  /// `/down/<placed-id>` through the placed-volume path. It is a diagnostic
  /// representation, not a stored navigation-state representation.
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

  /// @brief Serialize the state as a top-to-bottom list of child indices.
  /// @details The list starts with the world marker 0 and then contains child
  /// ids used to replay the path through placed-volume daughters.
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

  /// @brief Reconstruct the state from a list of child indices.
  /// @details The input must use the format produced by
  /// `GetPathAsListOfIndices`: world marker 0 followed by child ids.
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

  /// @brief Clear the current state.
  /// @details This resets the tuple to outside, clears last-exited state, and
  /// clears the boundary flag.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Clear()
  {
    fNavTuple.Clear();
    fLastExited.Clear();
    fOnBoundary = false;
  }

  /// @brief Print one raw tuple-table touchable record.
  /// @details The output includes encoded tuple table offsets, scene ids, and
  /// the shared logical-record address for diagnostics.
  VECCORE_ATT_HOST_DEVICE
  static void PrintRecord(NavIndex_t nav_ind)
  {
    if (nav_ind == 0) return;
    auto parent       = NavInd(nav_ind + NavIndexTableLayout::Tuple::kParent);
    auto placed_id    = NavInd(nav_ind + NavIndexTableLayout::Tuple::kPlacedVolume);
    auto child_id     = NavInd(nav_ind + NavIndexTableLayout::Tuple::kChildId);
    auto id           = NavInd(nav_ind + NavIndexTableLayout::Tuple::kTouchableId);
    auto logical_addr = NavInd(nav_ind + NavIndexTableLayout::Tuple::kLogicalRecord);
    auto logical_id   = NavInd(logical_addr + NavIndexTableLayout::Tuple::kLogicalVolumeId);
    auto scenes   = reinterpret_cast<const unsigned short *>(NavIndAddr(nav_ind + NavIndexTableLayout::Tuple::kScenes));
    auto scene_id = scenes[NavIndexTableLayout::Tuple::kParentSceneHalf];
    auto newscene_id = scenes[NavIndexTableLayout::Tuple::kCurrentSceneHalf];
    auto level       = GetLevelImpl(nav_ind);
    auto nd          = NavInd(logical_addr + NavIndexTableLayout::Tuple::kDaughterCount);
    printf("| navind %u |+0| parent %u |+1| placed_id %u |+2| child_id %u |+3| id %u |+4| logical_addr %u |+5| scene "
           "%hu | "
           "new_scene %hu |+6| level %u | ... |%u| logical_id %u |+1| nd %u ",
           nav_ind, parent, placed_id, child_id, id, logical_addr, scene_id, newscene_id, level, logical_addr,
           logical_id, nd);
    if (nd > 0) printf("|+2| d0 %u | ...", NavInd(logical_addr + NavIndexTableLayout::Tuple::kDaughters));
    printf("\n");
  }

  /// @brief Print this navigation state.
  /// @details When `print_names` is true on host builds, placed-volume labels
  /// are printed instead of only encoded ids.
  VECCORE_ATT_HOST_DEVICE
  void Print(bool print_names = false) const
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
    printf(" lastExited=");
    for (unsigned i = 0; i <= fLastExited.fLevel; ++i) {
      printf("%u", fLastExited[i]);
      if (i < fLastExited.fLevel) printf(" | ");
    }
    printf(", id=%u, level=%u/%u,  onBoundary=%s, path=<", GetId(), level, GetMaxLevel(),
           (fOnBoundary ? "true" : "false"));
    int last_scene = 0;
    nav_tuple.Clear();
    for (int i = 0; i <= level; ++i) {
      unsigned short scene_id = 0, newscene_id = 0;
      nav_tuple = GetNavTupleImpl(fNavTuple, i);
      GetSceneIdImpl(nav_tuple, scene_id, newscene_id);
      if (scene_id != last_scene) {
        printf(" | ");
        last_scene = scene_id;
      }
#ifndef VECCORE_CUDA
      if (print_names) {
        auto vol = At(i);
        printf("/%s", vol ? vol->GetLabel().c_str() : "TOP_SCENE");
      } else
#endif
        printf("/%u", nav_tuple.Top());
    }
    printf(">\n");
  }

  /// @brief Print the scene-local top record for an encoded navigation index.
  /// @details This diagnostic helper walks parent links inside one scene and
  /// prints the resulting table path.
  VECCORE_ATT_HOST_DEVICE
  static void PrintTopImpl(NavIndex_t top)
  {
    if (top == 0) {
      printf("navInd=0, id=0, path=TOP_SCENE\n");
      return;
    }
    // Find level
    int level         = 0;
    NavIndex_t mother = top;
    while (mother) {
      level++;
      mother = NavInd(mother);
    }
    printf("navInd=");
    unsigned short scene_id = 0, newscene_id = 0;
    GetSceneIdImpl(top, scene_id, newscene_id);
    printf("s%u:%u", scene_id, top);

    printf(", id=%u, level=%u,  path=<", GetIdImpl(top), level);
    for (int i = 0; i <= level; ++i) {
      mother = top;
      for (int j = 0; j < level - i; ++j)
        mother = NavInd(mother);
#ifndef VECCORE_CUDA
      auto vol = (mother > 0) ? ToPlacedVolume(NavInd(mother + NavIndexTableLayout::Tuple::kPlacedVolume)) : nullptr;
      printf("/%s", vol ? vol->GetLabel().c_str() : "TOP_SCENE");
#else
      printf("/%u", mother);
#endif
    }
    printf(">\n");
  }

  /// @brief Print the current top state.
  /// @details This prints only the path entries in the active top scene.
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
      if (scene_id != top_scene) continue;
#ifndef VECCORE_CUDA
      auto vol = At(i);
      printf("/%s", vol ? vol->GetLabel().c_str() : "TOP_SCENE");
#else
      printf("/%u", nav_tuple.Top());
#endif
    }
    printf(">\n");
  }

  /// @brief Dump this navigation state.
  /// @details This is a diagnostic alias for `Print()`.
  VECCORE_ATT_HOST_DEVICE
  void Dump() const { Print(); }

  /// @brief Test whether two states identify the same placed-volume path.
  /// @details In the tuple representation, equality of all tuple components is
  /// path equality.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool HasSamePathAsOther(NavStateTuple const &other) const { return (fNavTuple == other.fNavTuple); }

  /// @brief Print the compact placed-volume id sequence for this state.
  /// @details This is a debugging helper and does not print table record
  /// internals.
  void printValueSequence(std::ostream & = std::cerr) const;

  /// @brief Return a checksum for quick state comparison.
  /// @details The checksum combines all tuple components and can be used as a
  /// quick rejection test before full state comparison.
  unsigned long getCheckSum() const
  {
    unsigned long checksum = 0;
    for (uint ituple = 0; ituple <= fNavTuple.fLevel; ++ituple)
      checksum += (unsigned long)fNavTuple[ituple];

    return checksum;
  }

  /// @brief Return whether the state is outside the detector setup.
  /// @details The outside state is encoded by a level-zero tuple whose top
  /// index is zero.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsOutside() const { return fNavTuple.IsOutside(); }

  /// @brief Return whether the current state is on a boundary.
  /// @details This flag is carried by the navigation state and is managed by
  /// navigators; it is not derived from the table record.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsOnBoundary() const { return fOnBoundary; }

  /// @brief Set the boundary flag.
  /// @details The flag is stored in the state object and copied with the state.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetBoundaryState(bool b) { fOnBoundary = b; }
  /// @}
};

/**
 * @brief Print the compact placed-volume id sequence for this state.
 * @details This out-of-line definition implements the diagnostic helper
 * declared in the common navigation-state API.
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
