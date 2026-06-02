/// \file NavStateIndex
/// \author Andrei Gheata (andrei.gheata@cern.ch)
/// \date 12.03.2014

#ifndef VECGEOM_NAVIGATION_NAVSTATEINDEX_H_
#define VECGEOM_NAVIGATION_NAVSTATEINDEX_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Transformation3DMP.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/management/NavIndexTableLayout.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/VolumeTree.h"
#include "VecGeom/management/DeviceGlobals.h"

#include <iostream>
#include <string>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Navigation state represented by one touchable index.
 *
 * @details `NavStateIndex` stores one address into the global navigation table.
 * The selected record encodes the full touchable path through parent links and
 * cached metadata. This keeps per-track state compact and lookup fast, but the
 * table contains one expanded record for every touchable and can be large for
 * repeated detector geometries.
 *
 * The class implements the common `NavigationState` API shared with
 * `NavStateTuple`: `GetMaxLevel()`, `GetCurrentLevel()`, `At(level)`,
 * `ValueAt(level)`, `Push(...)`, `Pop()`, `Top()`, transform accessors, scene
 * accessors, and outside/boundary flags.
 */
class NavStateIndex {
public:
  using Value_t = unsigned int;

private:
  NavIndex_t fNavInd     = 0;     ///< Navigation state index
  NavIndex_t fLastExited = 0;     ///< Navigation state index of the last exited state
  bool fOnBoundary       = false; ///< flag indicating whether track is on boundary of the "Top()" placed volume

public:
  /// @brief Construct a state from an encoded navigation-table index.
  /// @details A value of zero denotes the outside state. Non-zero values are
  /// addresses into the expanded `NavStateIndex` navigation table.
  VECCORE_ATT_HOST_DEVICE
  NavStateIndex(NavIndex_t nav_ind = 0) { fNavInd = nav_ind; }

  /// @brief Construct a state from a one-element container.
  /// @details The container form is part of the generic navigation-state API.
  /// For `NavStateIndex` it must contain exactly one encoded table index.
  template <typename Container>
  VECCORE_ATT_HOST_DEVICE NavStateIndex(Container const *cont)
  {
    VECGEOM_ASSERT(cont->size() == 1);
    fNavInd = (*cont)[0];
  }

  /// @brief Return the maximum geometry depth.
  /// @details This is the geometry hierarchy depth, not the size of this
  /// navigation-state object. `NavStateIndex` stores a single table index
  /// independently of this value.
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

  /// @brief Return the current encoded navigation-table index.
  /// @details This is the complete state for the expanded-index
  /// representation.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  NavIndex_t GetNavIndex() const { return fNavInd; }

  /// @brief Return the representation-specific state value.
  /// @details For `NavStateIndex`, this is the same encoded table index as
  /// `GetNavIndex`.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  NavIndex_t GetState() const { return fNavInd; }

  /// @brief Copy this state to another `NavStateIndex`.
  /// @details This copies only the compact state fields, not any global table
  /// data referenced by the state.
  VECCORE_ATT_HOST_DEVICE
  void CopyTo(NavStateIndex *other) const { *other = *this; }

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
  static NavIndex_t NavInd(NavIndex_t nav_ind) { return *NavIndAddr(nav_ind); }

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
  /// @details The world record is stored at navigation index 1 in the expanded
  /// table.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static int WorldId() { return NavInd(NavIndexTableLayout::Index::kPlacedVolume + 1); }

  /// @brief Resolve a compact placed-volume id to `VolumeTree` metadata.
  /// @details The metadata provides the child id used to descend through
  /// daughter blocks.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static vecgeom::PlacedId const &ToPlacedId(size_t iplaced)
  {
    return vecgeom::VolumeTree::Instance().fPlaced[iplaced];
  }

  /// @brief Decode the daughter count from a table record.
  /// @details The count is stored in the packed metadata field of the expanded
  /// `NavStateIndex` record.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned short GetNdaughtersImpl(NavIndex_t nav_ind)
  {
    constexpr unsigned int kOffsetNd =
        NavIndexTableLayout::Index::kPacked * sizeof(NavIndex_t) + NavIndexTableLayout::Index::kDaughterCountByte;
    auto content_nd = (unsigned short *)((unsigned char *)(NavIndAddr(nav_ind)) + kOffsetNd);
    return *content_nd;
  }

  /// @brief Decode the zero-based geometry level from a table record.
  /// @details The level is stored as a byte in the packed metadata field.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static unsigned char GetLevelImpl(NavIndex_t nav_ind)
  {
    constexpr unsigned int kOffsetLevel =
        NavIndexTableLayout::Index::kPacked * sizeof(NavIndex_t) + NavIndexTableLayout::Index::kLevelByte;
    auto content_level = (unsigned char *)(NavIndAddr(nav_ind)) + kOffsetLevel;
    return *content_level;
  }

  /// @brief Decode scene ids for a table record.
  /// @details `NavStateIndex` has no scene compression, so the output ids are
  /// always zero and the return value is false.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static bool GetSceneIdImpl(NavIndex_t const & /*nav_ind*/, unsigned short &scene_id, unsigned short &newscene_id)
  {
    scene_id = newscene_id = 0;
    return false;
  }

  /// @brief Return the table record for a requested path level.
  /// @details Parent links are followed from `nav_ind` until the requested
  /// zero-based level is reached.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static NavIndex_t GetNavIndexImpl(NavIndex_t nav_ind, int level)
  {
    int up            = GetLevelImpl(nav_ind) - level;
    NavIndex_t mother = nav_ind;
    while (mother && up--)
      mother = NavInd(mother);
    return mother;
  }

  /// @brief Decode the touchable id from a table record.
  /// @details A zero navigation index returns zero.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static NavIndex_t GetIdImpl(NavIndex_t nav_ind)
  {
    return (nav_ind > 0) ? NavInd(nav_ind + NavIndexTableLayout::Index::kTouchableId) : 0;
  }

  /// @brief Test whether one table record descends from another.
  /// @details The check follows expanded-table parent links from `child_ind`
  /// toward the root until `parent_ind` is reached or the paths diverge.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static bool IsDescendentImpl(NavIndex_t child_ind, NavIndex_t parent_ind)
  {
    NavIndex_t ind = child_ind;
    while (ind > parent_ind) {
      ind = NavInd(ind);
      if (ind == parent_ind) return true;
    }
    return false;
  }

  /// @brief Decode the logical-volume id from a table record.
  /// @details A zero navigation index returns zero.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static unsigned int GetLogicalIdImpl(NavIndex_t nav_ind)
  {
    return nav_ind ? NavInd(nav_ind + NavIndexTableLayout::Index::kLogicalVolume) : 0;
  }

  /// @brief Decode the child id from a table record.
  /// @details The child id is stored as an `int` in the record's fixed field
  /// block.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static int GetChildIdImpl(NavIndex_t const &nav_index)
  {
    auto content_ichild = reinterpret_cast<const int *>(NavIndAddr(nav_index + NavIndexTableLayout::Index::kChildId));
    return *content_ichild;
  }

  /// @brief Report whether an encoded index represents a scene boundary.
  /// @details The expanded-index representation has no scenes; only zero is
  /// treated as a scene-like placeholder.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static bool IsSceneImpl(NavIndex_t nav_ind) { return nav_ind == 0; }

  /// @brief Move an encoded state to its parent.
  /// @details A zero state remains zero.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PopImpl(NavIndex_t &nav_ind) { nav_ind = (nav_ind > 0) ? NavInd(nav_ind) : 0; }

  /// @brief Descend to a daughter by placed-volume pointer.
  /// @details The volume child id selects an entry in the current record's
  /// daughter block. A zero state descends to the world record.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PushImpl(NavIndex_t &nav_ind, VPlacedVolume const *v)
  {
    nav_ind = (nav_ind > 0) ? NavInd(nav_ind + NavIndexTableLayout::Index::kDaughters + v->GetChildId()) : 1;
  }

  /// @brief Return a child table record by child id.
  /// @details The child record address is read from the current record's
  /// daughter block.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static NavIndex_t GetChildNavInd(NavIndex_t nav_ind, int ichild)
  {
    return (nav_ind > 0) ? NavInd(nav_ind + NavIndexTableLayout::Index::kDaughters + ichild) : 0;
  }

  /// @brief Descend to a daughter by child id.
  /// @details The child id selects an entry in the current record's daughter
  /// block. A zero state descends to the world record.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PushDaughterImpl(NavIndex_t &nav_ind, int idaughter)
  {
    nav_ind = (nav_ind > 0) ? NavInd(nav_ind + NavIndexTableLayout::Index::kDaughters + idaughter) : 1;
  }

  /// @brief Descend to a daughter by compact placed-volume id.
  /// @details The placed-volume id is resolved through `VolumeTree` to recover
  /// the child id used by `PushDaughterImpl`.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PushImpl(NavIndex_t &nav_ind, int iplaced)
  {
    auto const &pv_ind = ToPlacedId(iplaced);
    PushDaughterImpl(nav_ind, pv_ind.fChildId);
  }

  /// @brief Resolve the top placed volume for an encoded state.
  /// @details A zero state resolves to `nullptr`.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static VPlacedVolume const *TopImpl(NavIndex_t nav_ind)
  {
    return (nav_ind > 0) ? ToPlacedVolume(NavInd(nav_ind + NavIndexTableLayout::Index::kPlacedVolume)) : nullptr;
  }

  /// @brief Return the placed world volume.
  /// @details The world record is stored at navigation index 1.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static VPlacedVolume const *World() { return ToPlacedVolume(NavInd(NavIndexTableLayout::Index::kPlacedVolume + 1)); }

  /// @brief Resolve the compact placed-volume id for an encoded state.
  /// @details A zero state returns -1.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static int TopIdImpl(NavIndex_t const &nav_ind)
  {
    return (nav_ind > 0) ? int(NavInd(nav_ind + NavIndexTableLayout::Index::kPlacedVolume)) : -1;
  }

  /// @brief Reconstruct the global-to-local transform for an encoded state.
  /// @details Cached transforms are used when present; otherwise parent links
  /// and placed-volume transforms are multiplied until a cached transform is
  /// reached.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE static void TopMatrixImpl(NavIndex_t nav_ind, Transformation3DMP<Real_t> &trans)
  {
    constexpr unsigned int kOffsetHasm =
        NavIndexTableLayout::Index::kPacked * sizeof(NavIndex_t) + NavIndexTableLayout::Index::kMatrixFlagsByte;

    unsigned char hasm;
    while (true) {
      if (nav_ind == 0) return;
      hasm            = *((unsigned char *)(NavIndAddr(nav_ind)) + kOffsetHasm);
      bool has_matrix = (hasm & NavIndexTableLayout::Index::kHasStoredMatrixFlag) > 0;
      if (has_matrix) break;
      // note that the surface model should always have a matrix and never do the following multiplication
      auto const &t = *TopImpl(nav_ind)->GetTransformation();
      trans *= t;
      nav_ind = NavInd(nav_ind);
    }

    if ((hasm & 0x03) == 0) return;
    bool has_trans = (hasm & NavIndexTableLayout::Index::kHasTranslationFlag) > 0;
    bool has_rot   = (hasm & NavIndexTableLayout::Index::kHasRotationFlag) > 0;
    auto nd        = GetNdaughtersImpl(nav_ind);

    auto transformationDataIndex = NavIndexTableLayout::Index::TransformStart(nav_ind, nd);

    const auto address = reinterpret_cast<const Precision *>(NavIndAddr(transformationDataIndex));
    VECGEOM_ASSERT(reinterpret_cast<uintptr_t>(address) % sizeof(Precision) == 0);

    Transformation3DMP<Real_t> t;
    t.Set(address, address + 3, has_trans, has_rot);
    trans *= t;
  }

  /// @brief Reconstruct the global-to-local transform for an encoded state.
  /// @details Cached transforms are used when present; otherwise parent links
  /// and placed-volume transforms are multiplied until a cached transform is
  /// reached.
  VECCORE_ATT_HOST_DEVICE
  static void TopMatrixImpl(NavIndex_t nav_ind, Transformation3D &trans)
  {
    constexpr unsigned int kOffsetHasm =
        NavIndexTableLayout::Index::kPacked * sizeof(NavIndex_t) + NavIndexTableLayout::Index::kMatrixFlagsByte;

    unsigned char hasm;
    while (true) {
      if (nav_ind == 0) return;
      hasm            = *((unsigned char *)(NavIndAddr(nav_ind)) + kOffsetHasm);
      bool has_matrix = (hasm & NavIndexTableLayout::Index::kHasStoredMatrixFlag) > 0;
      if (has_matrix) break;
      auto const &t = *TopImpl(nav_ind)->GetTransformation();
      trans *= t;
      nav_ind = NavInd(nav_ind);
    }

    if ((hasm & 0x03) == 0) return;
    bool has_trans = (hasm & NavIndexTableLayout::Index::kHasTranslationFlag) > 0;
    bool has_rot   = (hasm & NavIndexTableLayout::Index::kHasRotationFlag) > 0;
    auto nd        = GetNdaughtersImpl(nav_ind);

    auto transformationDataIndex = NavIndexTableLayout::Index::TransformStart(nav_ind, nd);

    const auto address = reinterpret_cast<const Precision *>(NavIndAddr(transformationDataIndex));
    VECGEOM_ASSERT(reinterpret_cast<uintptr_t>(address) % sizeof(Precision) == 0);

    Transformation3D t;
    t.Set(address, address + 3, has_trans, has_rot);
    trans *= t;
  }

  /// @brief Reconstruct the top transform within the active scene.
  /// @details `NavStateIndex` has no scene tuple, so this is equivalent to
  /// `TopMatrixImpl`.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE static void TopInSceneMatrixImpl(NavIndex_t nav_ind, Transformation3DMP<Real_t> &trans)
  {
    // Get transformation of the node in the top scene
    TopMatrixImpl(nav_ind, trans);
  }

  /// @brief Reconstruct the top transform within the active scene.
  /// @details `NavStateIndex` has no scene tuple, so this is equivalent to
  /// `TopMatrixImpl`.
  VECCORE_ATT_HOST_DEVICE
  static void TopInSceneMatrixImpl(NavIndex_t nav_ind, Transformation3D &trans)
  {
    // Get transformation of the node in the top scene
    TopMatrixImpl(nav_ind, trans);
  }

  /// @brief Reconstruct the active scene transform.
  /// @details `NavStateIndex` has no scene tuple, so the scene transform is the
  /// identity and this function leaves `trans` unchanged.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE static void SceneMatrixImpl(NavIndex_t const & /*nav_tuple*/,
                                                      Transformation3DMP<Real_t> & /*trans*/)
  {
  }

  /// @brief Reconstruct the active scene transform.
  /// @details `NavStateIndex` has no scene tuple, so the scene transform is the
  /// identity and this function leaves `trans` unchanged.
  VECCORE_ATT_HOST_DEVICE
  static void SceneMatrixImpl(NavIndex_t const & /*nav_tuple*/, Transformation3D & /*trans*/) {}

  /// @brief Transform a global point to an encoded state's local frame.
  /// @details The top transform for `nav_ind` is reconstructed and applied to
  /// `globalpoint`.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE static Vector3D<Real_t> GlobalToLocalImpl(NavIndex_t nav_ind,
                                                                    Vector3D<Real_t> const &globalpoint)
  {
    Transformation3DMP<Real_t> trans;
    TopMatrixImpl(nav_ind, trans);
    Vector3D<Real_t> local = trans.Transform(globalpoint);
    return local;
  }

  /// @brief Transform a global point to an encoded state's local frame.
  /// @details The top transform for `nav_ind` is reconstructed and applied to
  /// `globalpoint`.
  VECCORE_ATT_HOST_DEVICE
  static Vector3D<Precision> GlobalToLocalImpl(NavIndex_t nav_ind, Vector3D<Precision> const &globalpoint)
  {
    Transformation3D trans;
    TopMatrixImpl(nav_ind, trans);
    Vector3D<Precision> local = trans.Transform(globalpoint);
    return local;
  }

  /**
   * @name Common NavigationState API
   *
   * @brief Methods consumed by navigators and helper utilities independent of
   * the concrete navigation-state representation.
   *
   * @details Levels are zero-based for a concrete touchable path: the world is
   * level 0, its daughters are level 1, and `GetCurrentLevel()` returns the
   * number of filled path entries. `ValueAt(level)` returns the compact
   * placed-volume id at that level, not a child-index token.
   */
  /// @{
  /// @brief Return the placed volume stored as last exited.
  /// @details The returned volume is resolved from the stored last-exited table
  /// index. A null pointer denotes that no exited volume is recorded.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *GetLastExited() const { return TopImpl(fLastExited); }

  /// @brief Return the representation-specific last-exited state.
  /// @details For `NavStateIndex`, this is one encoded navigation-table index.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t GetLastExitedState() const { return fLastExited; }

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
  void SetLastExited() { fLastExited = fNavInd; }

  /// @brief Set the last-exited state explicitly.
  /// @details The argument must be an encoded index in the `NavStateIndex`
  /// navigation table, or zero for none.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetLastExited(NavIndex_t const &navind) { fLastExited = navind; }

  /// @brief Replace the current navigation state.
  /// @details The argument must be an encoded index in the expanded
  /// navigation table, or zero for the outside state.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetNavIndex(NavIndex_t navind) { fNavInd = navind; }

  /// @brief Return the number of daughters of the current top volume.
  /// @details The value is decoded from the current table record metadata.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned short GetNdaughters() const { return GetNdaughtersImpl(fNavInd); }

  /// @brief Return the touchable id of the current state.
  /// @details The id is the builder-assigned touchable id stored in the table
  /// record, not the placed-volume id.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t GetId() const { return GetIdImpl(fNavInd); }

  /// @brief Return the top id of the parent scene.
  /// @details The expanded-index representation has no scene tuple, so this is
  /// always zero.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t GetParentSceneTopId() const { return 0; }

  /// @brief Query scene transition ids for this state.
  /// @details `NavStateIndex` does not encode scene transitions, so both output
  /// ids are set to zero and the function returns false.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool GetSceneId(unsigned short &scene_id, unsigned short &newscene_id) const
  {
    scene_id = newscene_id = 0;
    return false;
  }

  /// @brief Return the active scene nesting level.
  /// @details `NavStateIndex` has no scene tuple and therefore always reports
  /// level zero.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned int GetSceneLevel() const { return 0; }

  /// @brief Return the parent scene id.
  /// @details `NavStateIndex` has no scene tuple and therefore always reports
  /// scene zero.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned short GetParentScene() const { return 0; }

  /// @brief Return the logical-volume id of the current top volume.
  /// @details The id is decoded from the current table record.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned int GetLogicalId() const { return GetLogicalIdImpl(fNavInd); }

  /// @brief Return the child id of the current top volume.
  /// @details The child id is the index of this placed volume among its
  /// parent's daughters.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int GetChildId() const { return GetChildIdImpl(fNavInd); }

  /// @brief Report whether the state is at a scene boundary.
  /// @details Scene boundaries are only represented by `NavStateTuple`, so this
  /// is always false for `NavStateIndex`.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsScene() const { return false; }

  /// @brief Test whether this state is below the given parent state.
  /// @details Parent links are followed inside the expanded navigation table.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsDescendent(NavIndex_t parent) const { return IsDescendentImpl(fNavInd, parent); }

  /// @brief Descend to a daughter volume by placed-volume pointer.
  /// @details The daughter child id is taken from the volume and resolved
  /// through the current table record.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Push(VPlacedVolume const *v) { PushImpl(fNavInd, v); }

  /// @brief Descend to a daughter volume by compact placed-volume id.
  /// @details The child id is looked up in `VolumeTree` and then resolved
  /// through the current table record.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Push(int iplaced) { PushImpl(fNavInd, iplaced); }

  /// @brief Descend to a daughter volume by child id.
  /// @details The child id is resolved in the current table record's daughter
  /// block.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void PushDaughter(int idaughter) { PushDaughterImpl(fNavInd, idaughter); }

  /// @brief Enter a new scene.
  /// @details This is a no-op for `NavStateIndex` because the representation is
  /// fully expanded and has no scene tuple.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void PushScene(NavIndex_t) {}

  /// @brief Move to the parent volume.
  /// @details The current table index is replaced with its encoded parent
  /// index, or zero when already outside.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Pop() { PopImpl(fNavInd); }

  /// @brief Leave the current scene.
  /// @details This sets the state to outside because `NavStateIndex` has no
  /// scene stack to unwind.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void PopScene() { fNavInd = 0; }

  /// @brief Return the current top placed volume.
  /// @details The volume pointer is resolved from the current table record.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *Top() const { return TopImpl(fNavInd); }

  /// @brief Return the compact placed-volume id of the current top volume.
  /// @details Returns -1 when the state is outside.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  int TopId() const { return TopIdImpl(fNavInd); }

  /// @brief Return the zero-based level of the current top volume.
  /// @details The world has level 0. For the number of filled path entries,
  /// use `GetCurrentLevel()`.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  unsigned char GetLevel() const { return GetLevelImpl(fNavInd); }

  /// @brief Return the number of filled path entries.
  /// @details This is `GetLevel() + 1` for an in-geometry state. The outside
  /// state has implementation-defined level metadata and is normally tested
  /// with `IsOutside()`.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  unsigned char GetCurrentLevel() const { return GetLevel() + 1; }

  /// @brief Return the encoded navigation index at a zero-based path level.
  /// @details The returned value addresses the expanded navigation table record
  /// for the volume at the requested level.
  /// @param level Zero-based level, where 0 is the world.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  NavIndex_t GetNavIndex(int level) const { return GetNavIndexImpl(fNavInd, level); }

  /// @brief Return the placed volume at a zero-based path level.
  /// @details Parent links are followed from the current top record until the
  /// requested level is reached.
  /// @param level Zero-based level, where 0 is the world.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *At(int level) const
  {
    auto parent = GetNavIndexImpl(fNavInd, level);
    return (parent > 0) ? ToPlacedVolume(NavInd(parent + NavIndexTableLayout::Index::kPlacedVolume)) : nullptr;
  }

  /// @brief Return the compact placed-volume id at a zero-based path level.
  /// @details Parent links are followed from the current top record until the
  /// requested level is reached.
  /// @param level Zero-based level, where 0 is the world.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  size_t ValueAt(int level) const
  {
    auto parent = GetNavIndexImpl(fNavInd, level);
    return (parent > 0) ? (size_t)NavInd(parent + NavIndexTableLayout::Index::kPlacedVolume) : 0;
  }

  /// @brief Return the global-to-local transform of the current top volume.
  /// @details Cached table transforms are used when present; otherwise the
  /// transform is reconstructed by walking parent links.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void TopMatrix(Transformation3DMP<Real_t> &trans) const
  {
    TopMatrixImpl(fNavInd, trans);
  }

  /// @brief Return the global-to-local transform of the current top volume.
  /// @details Cached table transforms are used when present; otherwise the
  /// transform is reconstructed by walking parent links.
  VECCORE_ATT_HOST_DEVICE
  void TopMatrix(Transformation3D &trans) const { TopMatrixImpl(fNavInd, trans); }

  /// @brief Return the global-to-local transform to a requested path level.
  /// @details The requested level is resolved to a table index before reading
  /// or reconstructing the transform.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void TopMatrix(int tolevel, Transformation3DMP<Real_t> &trans) const
  {
    TopMatrixImpl(GetNavIndexImpl(fNavInd, tolevel), trans);
  }

  /// @brief Return the global-to-local transform to a requested path level.
  /// @details The requested level is resolved to a table index before reading
  /// or reconstructing the transform.
  VECCORE_ATT_HOST_DEVICE
  void TopMatrix(int tolevel, Transformation3D &trans) const
  {
    TopMatrixImpl(GetNavIndexImpl(fNavInd, tolevel), trans);
  }

  /// @brief Return the transform of the current top volume within the active scene.
  /// @details For `NavStateIndex`, there is only one expanded scene, so this is
  /// equivalent to `TopMatrix`.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void TopInSceneMatrix(Transformation3DMP<Real_t> &trans) const
  {
    TopInSceneMatrixImpl(fNavInd, trans);
  }

  /// @brief Return the transform of the current top volume within the active scene.
  /// @details For `NavStateIndex`, there is only one expanded scene, so this is
  /// equivalent to `TopMatrix`.
  VECCORE_ATT_HOST_DEVICE
  void TopInSceneMatrix(Transformation3D &trans) const { TopInSceneMatrixImpl(fNavInd, trans); }

  /// @brief Return the transform of the active scene.
  /// @details `NavStateIndex` has no scene tuple, so the scene transform is the
  /// identity and this function leaves `trans` unchanged.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void SceneMatrix(Transformation3DMP<Real_t> & /*trans*/) const
  {
  }

  /// @brief Return the transform of the active scene.
  /// @details `NavStateIndex` has no scene tuple, so the scene transform is the
  /// identity and this function leaves `trans` unchanged.
  VECCORE_ATT_HOST_DEVICE
  void SceneMatrix(Transformation3D & /*trans*/) const {}

  /// @brief Return the transform from this top-volume frame to another state.
  /// @details The resulting delta converts coordinates from `this->Top()` local
  /// frame to `other.Top()` local frame.
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void DeltaTransformation(NavStateIndex const &other, Transformation3DMP<Real_t> &delta) const;

  /// @brief Return the transform from this top-volume frame to another state.
  /// @details The resulting delta converts coordinates from `this->Top()` local
  /// frame to `other.Top()` local frame.
  VECCORE_ATT_HOST_DEVICE
  void DeltaTransformation(NavStateIndex const &other, Transformation3D &delta) const;

  /// @brief Transform a global point to the current top-volume frame.
  /// @details The current state's top transform is used.
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GlobalToLocal(Vector3D<Precision> const &localpoint) const
  {
    return GlobalToLocalImpl(fNavInd, localpoint);
  }

  /// @brief Transform a global point to the frame of a requested path level.
  /// @details The requested level is resolved to a table index before applying
  /// the transform.
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GlobalToLocal(Vector3D<Precision> const &localpoint, int tolevel) const
  {
    return GlobalToLocalImpl(GetNavIndexImpl(fNavInd, tolevel), localpoint);
  }

  /// @brief Return the path distance to another navigation state.
  /// @details The distance is the number of up/down steps required to move
  /// between the two states through their nearest common ancestor.
  VECCORE_ATT_HOST_DEVICE
  int Distance(NavStateIndex const &) const;

  /// @brief Return a textual relative path from this state to another state.
  /// @details The string encodes moves such as `/up`, `/horiz/<delta>`, and
  /// `/down/<placed-id>` through the placed-volume path. It is a diagnostic
  /// representation, not a stored navigation-state representation.
  std::string RelativePath(NavStateIndex const & /*other*/) const;

  /// @brief Serialize the state as a top-to-bottom list of child indices.
  /// @details The list starts with the world marker 0 and then contains child
  /// ids used to replay the path through placed-volume daughters.
  void GetPathAsListOfIndices(std::list<uint> &indices) const;

  /// @brief Reconstruct the state from a list of child indices.
  /// @details The input must use the format produced by
  /// `GetPathAsListOfIndices`: world marker 0 followed by child ids.
  void ResetPathFromListOfIndices(VPlacedVolume const *world, std::list<uint> const &indices);

  /// @brief Clear the current state.
  /// @details This resets the state to outside, clears last-exited state, and
  /// clears the boundary flag.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Clear()
  {
    fNavInd     = 0;
    fLastExited = 0;
    fOnBoundary = false;
  }

  /// @brief Print one raw navigation-table record.
  /// @details The output is intended for diagnostics and uses encoded table
  /// indices and ids.
  VECCORE_ATT_HOST_DEVICE
  static void PrintRecord(NavIndex_t nav_ind);

  /// @brief Print this navigation state.
  /// @details When `print_names` is true on host builds, placed-volume labels
  /// are printed instead of only encoded ids.
  VECCORE_ATT_HOST_DEVICE void Print(bool print_names = false) const;

  /// @brief Print the top record for an encoded navigation index.
  /// @details This diagnostic helper constructs a temporary state view for the
  /// supplied encoded index.
  VECCORE_ATT_HOST_DEVICE
  static void PrintTopImpl(NavIndex_t nav_ind) { NavStateIndex(nav_ind).Print(); }

  /// @brief Print the current top state.
  /// @details This is a diagnostic wrapper around `Print()`.
  VECCORE_ATT_HOST_DEVICE
  void PrintTop() const { Print(); }

  /// @brief Validate basic parent/daughter links for a table record.
  /// @details The check verifies that daughter records point back to the given
  /// parent. `nprint` controls diagnostic printing.
  VECCORE_ATT_HOST_DEVICE
  static bool IsValid(NavIndex_t nav_ind, int nprint = 0)
  {
    int nd      = GetNdaughtersImpl(nav_ind);
    auto parent = NavInd(nav_ind);
    if (nprint) printf("state %d: parent %d | %d daughters: ", nav_ind, parent, nd);
    bool valid = nav_ind == 1 || parent > 0;
    for (auto i = 0; i < nd; ++i) {
      auto nav_ind_child = GetChildNavInd(nav_ind, i);
      if (i < nprint) printf(" %d", nav_ind_child);
      valid &= NavInd(nav_ind_child) == nav_ind;
    }
    if (nprint) printf(" valid = %d\n", valid);
    return valid;
  }

  /// @brief Dump this navigation state.
  /// @details This is a diagnostic alias for `Print()`.
  VECCORE_ATT_HOST_DEVICE
  void Dump() const { Print(); }

  /// @brief Test whether two states identify the same placed-volume path.
  /// @details In the expanded-index representation, equality of table indices
  /// is path equality.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool HasSamePathAsOther(NavStateIndex const &other) const { return (fNavInd == other.fNavInd); }

  /// @brief Print the compact placed-volume id sequence for this state.
  /// @details This is a debugging helper and does not print table record
  /// internals.
  void printValueSequence(std::ostream & = std::cerr) const;

  /// @brief Return a checksum for quick state comparison.
  /// @details For `NavStateIndex`, the table index itself uniquely identifies
  /// the path and is used as the checksum.
  unsigned long getCheckSum() const { return (unsigned long)fNavInd; }

  /// @brief Return whether the state is outside the detector setup.
  /// @details The outside state is encoded by navigation index zero.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsOutside() const { return (fNavInd == 0); }

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
inline void NavStateIndex::printValueSequence(std::ostream &stream) const
{
  auto level = GetLevel();
  for (int i = 0; i < level + 1; ++i) {
    auto pvol = At(i);
    if (pvol) stream << "/" << ValueAt(i) << "(" << pvol->GetLabel() << ")";
  }
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_NAVIGATION_NAVSTATEINDEX_H_
