// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Class storing the global navigation index lookup table.
/// \file management/NavIndexTable.h
/// \author Andrei Gheata (andrei.gheata@cern.ch)

#ifndef VECGEOM_MANAGEMENT_NAVINDEXTABLE_H_
#define VECGEOM_MANAGEMENT_NAVINDEXTABLE_H_

#include "VecGeom/management/GeoVisitor.h"
#include "VecGeom/navigation/NavigationState.h"

#include <new>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class NavIndexTable;

/// @brief Visitor class for creating the navigation table
class BuildNavIndexVisitor : public GeoVisitorNavIndex {
private:
  size_t fTableSize   = sizeof(NavIndex_t); ///< The table size
  NavIndex_t *fNavInd = nullptr;            ///< Array storing the navigation info related to a given state
  int fLimitDepth     = 0;                  ///< limit depth to scache the transformations, 0 means unlimited
  int fError          = 0;                  ///< error code
  NavIndex_t fCurrent = 1;                  ///< Current navigation index being filled.
  bool fDoCount       = true;               ///< First pass to compute the table size
  bool fValidate      = false;              ///< If set, this flag will force a thorough validation upon visiting
  std::vector<NavIndex_t> fSelectedVolumes; ///< Volumes selected to be compressed in scenes
  std::vector<int> fSceneId;                ///< Scene id's for selected volumes
public:
  BuildNavIndexVisitor(int depth_limit, bool do_count)
      : GeoVisitorNavIndex(), fLimitDepth(depth_limit), fDoCount(do_count)
  {
    int ntot         = GeoManager::Instance().GetRegisteredVolumesCount();
    fSelectedVolumes = std::vector<NavIndex_t>(ntot, 0);
    fSceneId         = std::vector<int>(ntot, 0);
  }
  int GetError() const { return fError; }
  std::vector<NavIndex_t> &GetSelectedVolumes() { return fSelectedVolumes; }
  size_t GetTableSize() const { return fTableSize; }
  int GetSceneId(int ivol) const { return fSceneId[ivol]; }
  void SetSceneId(int ivol, int id) { fSceneId[ivol] = id; }
  bool IsSelected(int ivol) const { return fSelectedVolumes[ivol] > 0; }
  bool IsVisited(int ivol) const { return fSelectedVolumes[ivol] > 1; }
  void SetVisited(int ivol)
  {
    if (fSelectedVolumes[ivol] == 1) fSelectedVolumes[ivol] = 2;
  }

  void ResetVisited()
  {
    std::for_each(fSelectedVolumes.begin(), fSelectedVolumes.end(), [](NavIndex_t &i) {
      if (i > 1) i = 1;
    });
  }

  void SetSceneIndex(int ivol, NavIndex_t address) { fSelectedVolumes[ivol] = address; }
  NavIndex_t GetSceneIndex(int ivol)
  {
    auto scene_index = fSelectedVolumes[ivol];
    return (scene_index > 2) ? scene_index : 0;
  }
  void SetTable(NavIndex_t *table) { fNavInd = table; }
  void SetDoCount(bool flag) { fDoCount = flag; }
  void SetValidate(bool flag) { fValidate = flag; }
  std::vector<NavIndex_t> &SelectedVolumes() { return fSelectedVolumes; }

  /// @brief Visitor method applied for creating a single entru of the navigation table to use with indices
  /// @param state State currently visited
  /// @param level Current depth
  /// @param mother Mother navigation index
  /// @param dind Daugher index
  /// @param id Unique touchable id to be assigned
  /// @return Index of the record created for this state
  NavIndex_t apply(NavStatePath *state, int level, NavIndex_t mother, int dind, NavIndex_t &id);

  /// @brief Visitor method applied for creating a single entru of the navigation table to use with tuple states
  /// @param state State currently visited
  /// @param level Current depth
  /// @param mother Mother navigation index
  /// @param dind Daugher index
  /// @param id Unique touchable id to be assigned
  /// @param scene_id Current scene id
  /// @param new_scene_id New scene id
  /// @return Index of the record created for this state
  NavIndex_t apply_tuple(NavStatePath *state, int level, NavIndex_t mother, int dind, NavIndex_t &id, int scene_id,
                         int new_scene_id);

  /// @brief Run the optimal compression of touchables in logical scenes, constraining the maximum scene depth to
  ///        VECGEOM_NAVTUPLE_MAXDEPTH, and the minimum number of touchables per scene to NAVTUPLE_MINSCENE
  /// @param selected_volumes Container indexed with the logical volume id, returning the selection of scene volumes
  /// @param target_touchables Stop creating logical scenes if the target number is reached
  /// @param min_per_scene Minimum number of touchables of a logical volume that qualifying it to become a scene.
  void NodeReduction(int min_per_scene = 1000);
};

class NavIndexTable {
private:
  NavIndex_t *fNavInd       = nullptr; ///< address of the table
  VPlacedVolume *fVolBuffer = nullptr; ///< placed volume buffer
  NavIndex_t fWorld         = 1;       ///< index for the world volume
  size_t fTableSize         = 0;       ///< table size in bytes
  size_t fDepthLimit        = 0;       ///< depth limnit to which transformations will be cached

  NavIndexTable(NavIndex_t *table, size_t table_size) : fNavInd(table), fTableSize(table_size) {}

public:
  ~NavIndexTable() { CleanTable(); }

  /// @brief Returns the static instance of the navigation table
  /// @param nav_table Initialize with existing instance
  /// @param table_size Initialize with existing table size
  static NavIndexTable *Instance(NavIndex_t *nav_table = nullptr, size_t table_size = 0)
  {
    static NavIndexTable instance(nav_table, table_size);
    return &instance;
  }

  /// @brief Clean the current table
  void CleanTable()
  {
    delete[] fNavInd;
    fNavInd = nullptr;
  }

  /// @brief Returns the size in bytes of the table
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t GetTableSize() const { return fTableSize; }

  /// @brief Returns the address of the table
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  NavIndex_t *GetTable() const { return fNavInd; }

  /// @brief Set the volume buffer
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetVolumeBuffer(VPlacedVolume *buffer) { fVolBuffer = buffer; }

  /// @brief Get the navigation index of the world
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  NavIndex_t GetWorld() const { return fWorld; }

  /// @brief Allocate the table as a NavIndex_t array
  /// @param bytes Size to allocate
  /// @return Success flag
  bool AllocateTable(size_t bytes);

  /// @brief Create the navigation table starting from a top placed volume
  /// @param top Placed volume to start from
  /// @param maxdepth Maximum geometry depth
  /// @param depth_limit Depth limit to which to which transformations should be cached (0 = full)
  /// @param min_per_scene Minimum touchables to trigger scene creation for navigation tuple
  /// @return Success flag
  bool CreateTable(VPlacedVolume const *top, int maxdepth, int depth_limit, int min_per_scene = 1000);

  /// @brief Validate the navigation table.
  /// @param top Placed volume to start from
  /// @param maxdepth Maximum geometry depth
  /// @return Success flag
  bool Validate(VPlacedVolume const *top, int maxdepth) const;

  /// @brief Validate a single state against the built table
  /// @param state State to validate
  /// @param error Error code
  /// @return Corresponding navigation index for the deepest scene (0 = failure)
  NavIndex_t ValidateState(NavStatePath *state, int &error);

  /// @brief Recursive tree traversal keeping track of the state context and applying the injected
  /// visitor to create the navigation index table
  /// @tparam Visitor Visitor type
  /// @param currentvolume Placed volume currently visited
  /// @param visitor Visitor object
  /// @param state Current navigation state path
  /// @param id Unique touchable index (incremental)
  /// @param level Current depth in the tree
  /// @param mother Mother navigation index
  /// @param dind Child index
  template <typename Visitor>
  static int visitAllPlacedVolumesNavIndex(VPlacedVolume const *currentvolume, Visitor *visitor, NavStatePath *state,
                                           NavIndex_t &id, int level = 0, NavIndex_t mother = 0, int dind = 0)
  {
    if (currentvolume != NULL) {
      state->Push(currentvolume);
      NavIndex_t new_mother = visitor->apply(state, level, mother, dind, id);
      auto ierr             = visitor->GetError();
      if (ierr) return ierr;
      int size = currentvolume->GetDaughters().size();
      for (int i = 0; i < size; ++i) {
        ierr = visitAllPlacedVolumesNavIndex(currentvolume->GetDaughters().operator[](i), visitor, state, id, level + 1,
                                             new_mother, i);
        if (ierr) return ierr;
      }
      state->Pop();
    }
    return 0;
  }

  /// @brief Recursive tree traversal keeping track of the state context and applying the injected
  /// visitor to create the navigation tuple table
  /// @tparam Visitor Visitor type
  /// @param currentvolume Placed volume currently visited
  /// @param visitor Visitor object
  /// @param state Current navigation state path
  /// @param id Unique touchable index (incremental)
  /// @param scene_id Unique scene index (incremental)
  /// @param level Current depth in the tree
  /// @param mother Mother navigation index
  /// @param dind Child index
  /// @param new_scene Nodes have to be placed in a new scene
  template <typename Visitor>
  static int visitAllPlacedVolumesNavTuple(VPlacedVolume const *currentvolume, Visitor *visitor, NavStatePath *state,
                                           NavIndex_t &id, int &iscene, int scene_id, int level = 0,
                                           NavIndex_t mother = 0, int dind = 0, bool new_scene = false)
  {
    if (currentvolume != NULL) {
      auto lvol = currentvolume->GetLogicalVolume();
      int ivol  = lvol->id();
      state->Push(currentvolume);
      bool selected    = visitor->IsSelected(ivol);
      bool visited     = visitor->IsVisited(ivol);
      int new_scene_id = scene_id;
      // increment new scene id if this is an unvisited scene
      if (selected) {
        if (!visited) {
          new_scene_id = ++iscene;
          visitor->SetSceneId(ivol, new_scene_id);
        } else {
          new_scene_id = visitor->GetSceneId(ivol);
        }
      }
      NavIndex_t id_bak     = id;
      NavIndex_t new_mother = visitor->apply_tuple(state, level, mother, dind, id, scene_id, new_scene_id);
      auto ierr             = visitor->GetError();
      if (ierr) return ierr;
      if (!visited) {
        int size                  = currentvolume->GetDaughters().size();
        NavStatePath *scene_state = nullptr;
        if (selected) {
          scene_state = NavStatePath::MakeInstance(GeoManager::Instance().getMaxDepth());
          scene_state->Push(GeoManager::Instance().GetWorld());
          assert(visitor->IsVisited(ivol));
          id    = 1; // index 0 not used
          level = 0;
        }
        for (int i = 0; i < size; ++i) {
          ierr = visitAllPlacedVolumesNavTuple(currentvolume->GetDaughters().operator[](i), visitor,
                                               scene_state ? scene_state : state, id, iscene, new_scene_id, level + 1,
                                               new_mother, i, selected);
          if (ierr) return ierr;
        }
        if (selected) {
          NavStatePath::ReleaseInstance(scene_state);
          id = ++id_bak;
        }
      }
      state->Pop();
    }
    return 0;
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
