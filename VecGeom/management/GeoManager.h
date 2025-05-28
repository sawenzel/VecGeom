// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Declaration of the manager/registry class for VecGeom geometries.
/// \file management/GeoManager.h
/// \author created by Sandro Wenzel, Johannes de Fine Licht

#ifndef VECGEOM_MANAGEMENT_GEOMANAGER_H_
#define VECGEOM_MANAGEMENT_GEOMANAGER_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/management/GeoVisitor.h"
#include "VecGeom/navigation/NavStateFwd.h"

#include <map>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {
class UnplacedScaledShape;
class Scale3D;
} // namespace VECGEOM_IMPL_NAMESPACE

/**
 * @brief A class serving as central registry for VecGeom geometries.
 *
 * The GeoManager knows about all unplaced and placed volumes created by
 * the user and top node defining the geometry. It also offers functions to
 * iterate over the geometry tree or factory functions to create volume/shape instances.
 */
class GeoManager {

private:
  int fVolumeCount    = 0; // total number of logical volumes
  int fTotalNodeCount = 0; // total number of nodes in the geometry tree
  VPlacedVolume const *fWorld;

  // consider making these things rvalues
  std::map<unsigned int, VPlacedVolume *> fPlacedVolumesMap;
  std::map<unsigned int, LogicalVolume *> fLogicalVolumesMap;
  std::map<VPlacedVolume const *, unsigned int> fVolumeToIndexMap;
  std::vector<LogicalVolume *> fLogicalVolumesArray;
  int fMaxDepth    = 0;     // maximum geometry depth
  int fCacheDepth  = 0;     // caching level for global transformations (0 = cache all)_
  int fMinPerScene = 1000;  // minimum number of touchables to trigger scene creation in tuple mode
  bool fIsClosed   = false; // geometry closed flag

  /// Traverses the geometry tree of placed volumes and applies injected Visitor.
  template <typename Visitor>
  void visitAllPlacedVolumes(VPlacedVolume const *, Visitor *visitor, int level = 1) const;

  /// Traverses the geometry tree keeping track of the state context (volume path or navigation state)
  /// and applies the injected Visitor
  template <typename Visitor>
  void visitAllPlacedVolumesWithContext(VPlacedVolume const *, Visitor *visitor, NavigationState *state,
                                        int level = 1) const;

  /// Traverses the geometry tree keeping track of the state context (volume path or navigation state)
  /// and applies the injected Visitor for building the navigation index table
  template <typename Visitor>
  void visitAllPlacedVolumesNavIndex(VPlacedVolume const *, Visitor *visitor, NavigationState *state, int level,
                                     int dind) const;

public:
  static VPlacedVolume *gCompactPlacedVolBuffer;
  static NavIndex_t *gNavIndex;     // address of navigation index table
  static Precision gMillimeterUnit; // internal representation value for 1 milimmeter length (default is 0.1)

  /// Returns the singleton instance
  static GeoManager &Instance()
  {
    static GeoManager instance;
    return instance;
  }

  /// Returns the default length value stored as 1 mm
  static Precision GetMillimeterUnit() { return gMillimeterUnit; }

  /// Changes the default length value stored as 1 mm
  static void SetMillimeterUnit(Precision value) { gMillimeterUnit = value; }

  /**
   * Mark the current detector geometry as finished and initialize
   * important cached variables such as the maximum tree depth etc.
   */
  void CloseGeometry();

  /**
   * Returns if geometry is closed.
   */
  bool IsClosed() const { return fIsClosed; }

  /**
   * @brief Create a Index Hierarchy object
   */
  void CreateIndexHierarchy() const;
  bool CheckIndexHierarchy() const;

#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  /// A factory template for unplaced shapes.
  template <typename UnplacedShape_t, typename... ArgTypes>
  static UnplacedShape_t *MakeInstance(ArgTypes... Args);

  /// A factory for unplaced scaled shapes
  template <typename BaseShape_t, typename... ArgTypes>
  static VECGEOM_IMPL_NAMESPACE::UnplacedScaledShape *MakeScaledInstance(const VECGEOM_IMPL_NAMESPACE::Scale3D &scale,
                                                                         ArgTypes... args)
  {
    return Maker<VECGEOM_IMPL_NAMESPACE::UnplacedScaledShape>::MakeInstance<BaseShape_t>(scale, args...);
  }
#endif
  /** Compactify memory space used by VecGeom geometry objects.
   *
   * This is an internal method which should be called by ClosedGeometry.
   * It analyses the geometry and puts objects in contiguous buffers for the purpose
   * of less memory usage, better caching, fast random access to volumes
   * with indices, etc.
   *
   * Note that this procedure changes the memory location of objects. So the user
   * needs to adjust possible pointers.
   */
  void CompactifyMemory();

  /** Retrieve user-defined cache level for global transformations in USE_NAVINDEX mode. */
  int GetTransformationCacheDepth() const { return fCacheDepth; }

  /** User-defined cache level for global transformations in USE_NAVINDEX mode. */
  void SetTransformationCacheDepth(int depth) { fCacheDepth = depth; }

  /**
   * Set the world volume defining the entry point to the geometry.
   */
  void SetWorld(VPlacedVolume const *const w) { fWorld = w; }

  /**
   * Set the world volume and close geometry.
   */
  void SetWorldAndClose(VPlacedVolume const *const w)
  {
    SetWorld(w);
    CloseGeometry();
  }

  /// Returns the current world volume.
  VPlacedVolume const *GetWorld() const { return fWorld; }

  /**
   * Initialize geometry from a pre-compiled shared library
   * (such as obtained from the CppExporter)
   * This function sets the world and closes the geometry.
   */
  void LoadGeometryFromSharedLib(std::string, bool close = true);

  /// Lookup a placed volume instance from a index (logarithmic complexity).
  VPlacedVolume const *Convert(unsigned int index) { return fPlacedVolumesMap[index]; }

  /// Lookup the index of a placed volume (logarithmic complexity).
  unsigned int Convert(VPlacedVolume const *pvol) { return fVolumeToIndexMap[pvol]; }

  /**
   *  Give back a container c containing all logical volumes in the geometry.
   *  Container is supposed to be any Container that can store pointers to
   *  LogicalVolumes.
   */
  template <typename Container>
  void GetAllLogicalVolumes(Container &c) const;

  /**
   *  Give back a container c containing all placed volumes in the geometry.
   */
  template <typename Container>
  void getAllPlacedVolumes(Container &c) const;

  /**
   *  Give back a container c containing all possible geometry paths (NavigationStates)
   *  in the geometry, given a logical volume lvol.
   *  Container has to be an (stl::)container keeping pointer to NavigationStates.
   */
  template <typename Container>
  void getAllPathForLogicalVolume(LogicalVolume const *lvol, Container &c) const;

  /**
   *  Returns max depth of volume hierarchy. Can only be called after the geometry is closed.
   */
  int getMaxDepth() const
  {
    VECGEOM_ASSERT(fIsClosed == true);
    return fMaxDepth;
  }

  void RegisterPlacedVolume(VPlacedVolume *const placed_volume);

  void RegisterLogicalVolume(LogicalVolume *const logical_volume);

  void SetMinPerScene(int touchables) { fMinPerScene = touchables; }

  void DeregisterPlacedVolume(const int id);

  void DeregisterLogicalVolume(const int id);

  /**
   * \return Volume with passed id, or nullptr if the id wasn't found.
   */
  VPlacedVolume *FindPlacedVolume(const int id);

  /**
   * \return First occurrence of volume with passed label. If multiple volumes
   *         are found, their id will be printed to standard output.
   */
  VPlacedVolume *FindPlacedVolume(char const *const label);

  VECGEOM_FORCE_INLINE VPlacedVolume *GetPlacedVolume(const uint id) const { return &gCompactPlacedVolBuffer[id]; }

  /**
   * \return Volume with passed id, or NULL is the id wasn't found.
   */
  LogicalVolume *FindLogicalVolume(const int id);

  /**
   * \return First occurrence of volume with passed label. If multiple volumes
   *         are found, their id will be printed to standard output.
   */
  LogicalVolume *FindLogicalVolume(char const *const label);

  VECGEOM_FORCE_INLINE LogicalVolume *GetLogicalVolume(const uint id) const { return fLogicalVolumesArray[id]; }

  /**
   * \return Id of logical volume with passed label, or -1 if not found
   */
  int GetLogicalVolumeId(const std::string &label);

  /**
   * \return Label of logical volumen with passed Id, or empty string if not found
   */
  std::string GetLogicalVolumeLabel(int id);

  /**
   * Clear/resets the GeoManager. All geometry information will be deleted.
   */
  void Clear();

  /// Returns the number of placed volumes known to the GeoManager.
  size_t GetPlacedVolumesCount() const { return fPlacedVolumesMap.size(); }

  /** Returns the number of logical volumes registered in the GeoManager (map)
   * includes both tracking logical volumes and virtual volumes (which are part of composites for example)
   * in order to get the number of logical volumes which are seen from the perspective of a user,
   * the user should call getAllLogicalVolumes(...) and then determine the size from the resulting container
   */
  size_t GetRegisteredVolumesCount() const { return fLogicalVolumesMap.size(); }

  /// Returns the map of logical volumes.
  decltype(fLogicalVolumesMap) const &GetLogicalVolumesMap() const { return fLogicalVolumesMap; }

  /**
   * Returns the total number of leave nodes / geometry paths from top to leave in the geometry.
   */
  size_t GetTotalNodeCount() const { return fTotalNodeCount; }

  /// Creates the navigation index table, caching global transformations down to a given geometry depth
  /// min_per_scene is the minimum number of touchables to trigger scene creation for navigation tuples
#if defined(VECGEOM_USE_NAVINDEX) || defined(VECGEOM_USE_NAVTUPLE)
  bool MakeNavIndexTable(int depth_limit = 0, int min_per_scene = 1000, bool validate = true) const;
#endif

private:
  GeoManager()
      : fVolumeCount(0), fTotalNodeCount(0), fWorld(NULL), fPlacedVolumesMap(), fLogicalVolumesMap(),
        fVolumeToIndexMap(), fLogicalVolumesArray(), fMaxDepth(-1), fIsClosed(false)
  {
  }

  GeoManager(GeoManager const &);
  GeoManager &operator=(GeoManager const &);
};

template <typename Visitor>
void GeoManager::visitAllPlacedVolumes(VPlacedVolume const *currentvolume, Visitor *visitor, int level) const
{
  if (currentvolume != NULL) {
    visitor->apply(const_cast<VPlacedVolume *>(currentvolume), level);
    int size = currentvolume->GetDaughters().size();
    for (int i = 0; i < size; ++i) {
      visitAllPlacedVolumes(currentvolume->GetDaughters().operator[](i), visitor, level + 1);
    }
  }
}

template <typename Container>
void GeoManager::GetAllLogicalVolumes(Container &c) const
{
  c.clear();
  // walk all the volume hierarchy and insert
  // logical volume if not already in the container
  SimpleLogicalVolumeVisitor<Container> lv(c);
  visitAllPlacedVolumes(GetWorld(), &lv);
}

template <typename Container>
void GeoManager::getAllPlacedVolumes(Container &c) const
{
  c.clear();
  // walk all the volume hierarchy and insert
  // placed volumes if not already in the container
  SimplePlacedVolumeVisitor<Container> pv(c);
  visitAllPlacedVolumes(GetWorld(), &pv);
}

#ifndef VECCORE_CUDA_DEVICE_COMPILATION
/// A factory for unplaced shapes. Factory redirects to the "Maker" template
template <typename UnplacedShape_t, typename... Argtypes>
UnplacedShape_t *GeoManager::MakeInstance(Argtypes... args)
{
  return Maker<UnplacedShape_t>::MakeInstance(args...);
}
#endif
} // namespace vecgeom

#endif // VECGEOM_MANAGEMENT_GEOMANAGER_H_
