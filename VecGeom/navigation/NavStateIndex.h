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
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/VolumeTree.h"

#ifdef VECGEOM_ENABLE_CUDA
#include "VecGeom/management/CudaManager.h"
#endif

#include <iostream>
#include <string>

class TGeoBranchArray;

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * A class describing a current geometry state based on a single index
 * likely there will be such an object for each particle/track currently treated.
 */
class NavStateIndex {
public:
  using Value_t = unsigned int;

private:
  NavIndex_t fNavInd     = 0;     ///< Navigation state index
  NavIndex_t fLastExited = 0;     ///< Navigation state index of the last exited state
  bool fOnBoundary       = false; ///< flag indicating whether track is on boundary of the "Top()" placed volume

public:
  VECCORE_ATT_HOST_DEVICE
  NavStateIndex(NavIndex_t nav_ind = 0) { fNavInd = nav_ind; }

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
  static NavStateIndex *MakeInstance(int)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    return new NavStateIndex();
  }

  VECCORE_ATT_HOST_DEVICE
  static NavStateIndex *MakeCopy(NavStateIndex const &other) { return new NavStateIndex(other); }

  VECCORE_ATT_HOST_DEVICE
  static NavStateIndex *MakeInstanceAt(int, void *addr) { return new (addr) NavStateIndex(); }

  VECCORE_ATT_HOST_DEVICE
  static NavStateIndex *MakeCopy(NavStateIndex const &other, void *addr) { return new (addr) NavStateIndex(other); }

  VECCORE_ATT_HOST_DEVICE
  static void ReleaseInstance(NavStateIndex *state)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    delete state;
  }

  // returns the size in bytes of a NavStateIndex object with internal
  // path depth maxlevel
  VECCORE_ATT_HOST_DEVICE
  static size_t SizeOfInstance(int)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    return sizeof(NavStateIndex);
  }

  // returns the size in bytes of a NavStateIndex object with internal
  // path depth maxlevel -- including space needed for padding to next aligned object
  // of same kind
  VECCORE_ATT_HOST_DEVICE
  static size_t SizeOfInstanceAlignAware(int)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    return sizeof(NavStateIndex);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  NavIndex_t GetNavIndex() const { return fNavInd; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  NavIndex_t GetState() const { return fNavInd; }

  VECCORE_ATT_HOST_DEVICE
  int GetObjectSize() const { return (int)sizeof(NavStateIndex); }

  VECCORE_ATT_HOST_DEVICE
  static size_t SizeOf(size_t) { return sizeof(NavStateIndex); }

  VECCORE_ATT_HOST_DEVICE
  void CopyTo(NavStateIndex *other) const { *other = *this; }

  // copies a fixed and predetermined number of bytes
  // might be useful for specialized navigators which know the depth + SizeOf in advance
  // N is number of bytes to be copied and can be obtained by a prior call to constexpr NavStateIndex::SizeOf( ... );
  template <size_t N>
  void CopyToFixedSize(NavStateIndex *other) const
  {
    *other = *this;
  }

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
    assert(vecgeom::GeoManager::gNavIndex != nullptr);
    return &vecgeom::GeoManager::gNavIndex[nav_ind];
#endif
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static NavIndex_t NavInd(NavIndex_t nav_ind) { return *NavIndAddr(nav_ind); }

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
    assert(vecgeom::GeoManager::gCompactPlacedVolBuffer == nullptr ||
           vecgeom::GeoManager::gCompactPlacedVolBuffer[index].id() == index);
    return &vecgeom::GeoManager::gCompactPlacedVolBuffer[index];
#endif
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static int WorldId() { return NavInd(3); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static vecgeom::PlacedId const &ToPlacedId(size_t iplaced)
  {
    return vecgeom::VolumeTree::Instance().fPlaced[iplaced];
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static vecgeom::LogicalId const &ToLogicalId(size_t iplaced)
  {
    return vecgeom::VolumeTree::Instance().fPlaced[iplaced].fVolume;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static unsigned short GetNdaughtersImpl(NavIndex_t nav_ind)
  {
    constexpr unsigned int kOffsetNd = 5 * sizeof(NavIndex_t) + 2;
    auto content_nd                  = (unsigned short *)((unsigned char *)(NavIndAddr(nav_ind)) + kOffsetNd);
    return *content_nd;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static unsigned char GetLevelImpl(NavIndex_t nav_ind)
  {
    constexpr unsigned int kOffsetLevel = 5 * sizeof(NavIndex_t);
    auto content_level                  = (unsigned char *)(NavIndAddr(nav_ind)) + kOffsetLevel;
    return *content_level;
  }

  /// @brief Implementation for getting the scene id for a given navigation index
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static bool GetSceneIdImpl(NavIndex_t const & /*nav_ind*/, unsigned short &scene_id, unsigned short &newscene_id)
  {
    scene_id = newscene_id = 0;
    return false;
  }

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

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static NavIndex_t GetIdImpl(NavIndex_t nav_ind) { return (nav_ind > 0) ? NavInd(nav_ind + 1) : 0; }

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

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static unsigned int GetLogicalIdImpl(NavIndex_t nav_ind) { return nav_ind ? NavInd(nav_ind + 4) : 0; }

  /// @brief Implementation for getting the child id for a given navigation index
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static int GetChildIdImpl(NavIndex_t const &nav_index)
  {
    auto content_ichild = reinterpret_cast<const int *>(NavIndAddr(nav_index + 3));
    return *content_ichild;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static bool IsSceneImpl(NavIndex_t nav_ind) { return nav_ind == 0; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PopImpl(NavIndex_t &nav_ind) { nav_ind = (nav_ind > 0) ? NavInd(nav_ind) : 0; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PushImpl(NavIndex_t &nav_ind, VPlacedVolume const *v)
  {
    nav_ind = (nav_ind > 0) ? NavInd(nav_ind + 6 + v->GetChildId()) : 1;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PushDaughterImpl(NavIndex_t &nav_ind, int idaughter)
  {
    nav_ind = (nav_ind > 0) ? NavInd(nav_ind + 6 + idaughter) : 1;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void PushImpl(NavIndex_t &nav_ind, int iplaced)
  {
    auto const &pv_ind = ToPlacedId(iplaced);
    PushDaughterImpl(nav_ind, pv_ind.fChildId);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static VPlacedVolume const *TopImpl(NavIndex_t nav_ind)
  {
    return (nav_ind > 0) ? ToPlacedVolume(NavInd(nav_ind + 2)) : nullptr;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static int TopIdImpl(NavIndex_t const &nav_ind) { return (nav_ind > 0) ? int(NavInd(nav_ind + 2)) : -1; }

  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE static void TopMatrixImpl(NavIndex_t nav_ind, Transformation3DMP<Real_t> &trans)
  {
    constexpr unsigned int kOffsetHasm = 5 * sizeof(NavIndex_t) + 1;

    unsigned char hasm;
    while (true) {
      if (nav_ind == 0) return;
      hasm            = *((unsigned char *)(NavIndAddr(nav_ind)) + kOffsetHasm);
      bool has_matrix = (hasm & 0x04) > 0;
      if (has_matrix) break;
      // note that the surface model should always have a matrix and never do the following multiplication
      auto const &t = *TopImpl(nav_ind)->GetTransformation();
      trans *= t;
      nav_ind = NavInd(nav_ind);
    }

    if ((hasm & 0x03) == 0) return;
    bool has_trans = (hasm & 0x02) > 0;
    bool has_rot   = (hasm & 0x01) > 0;
    auto nd        = GetNdaughtersImpl(nav_ind);

    // Potentially skip one NavIndex_t to ensure alignment of transformation data
    auto transformationDataIndex     = nav_ind + 6 + nd + ((nd + 1) & 1);
    const bool padTransformationData = (transformationDataIndex * sizeof(NavIndex_t)) % sizeof(::Precision) != 0;
    transformationDataIndex += unsigned{padTransformationData};

    const auto address = reinterpret_cast<const Precision *>(NavIndAddr(transformationDataIndex));
    assert(reinterpret_cast<uintptr_t>(address) % sizeof(Precision) == 0);

    Transformation3DMP<Real_t> t;
    t.Set(address, address + 3, has_trans, has_rot);
    trans *= t;
  }

  VECCORE_ATT_HOST_DEVICE
  static void TopMatrixImpl(NavIndex_t nav_ind, Transformation3D &trans)
  {
    constexpr unsigned int kOffsetHasm = 5 * sizeof(NavIndex_t) + 1;

    unsigned char hasm;
    while (true) {
      if (nav_ind == 0) return;
      hasm            = *((unsigned char *)(NavIndAddr(nav_ind)) + kOffsetHasm);
      bool has_matrix = (hasm & 0x04) > 0;
      if (has_matrix) break;
      auto const &t = *TopImpl(nav_ind)->GetTransformation();
      trans *= t;
      nav_ind = NavInd(nav_ind);
    }

    if ((hasm & 0x03) == 0) return;
    bool has_trans = (hasm & 0x02) > 0;
    bool has_rot   = (hasm & 0x01) > 0;
    auto nd        = GetNdaughtersImpl(nav_ind);

    // Potentially skip one NavIndex_t to ensure alignment of transformation data
    auto transformationDataIndex     = nav_ind + 6 + nd + ((nd + 1) & 1);
    const bool padTransformationData = (transformationDataIndex * sizeof(NavIndex_t)) % sizeof(::Precision) != 0;
    transformationDataIndex += unsigned{padTransformationData};

    const auto address = reinterpret_cast<const Precision *>(NavIndAddr(transformationDataIndex));
    assert(reinterpret_cast<uintptr_t>(address) % sizeof(Precision) == 0);

    Transformation3D t;
    t.Set(address, address + 3, has_trans, has_rot);
    trans *= t;
  }

  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE static void TopInSceneMatrixImpl(NavIndex_t nav_ind, Transformation3DMP<Real_t> &trans)
  {
    // Get transformation of the node in the top scene
    TopMatrixImpl(nav_ind, trans);
  }

  VECCORE_ATT_HOST_DEVICE
  static void TopInSceneMatrixImpl(NavIndex_t nav_ind, Transformation3D &trans)
  {
    // Get transformation of the node in the top scene
    TopMatrixImpl(nav_ind, trans);
  }

  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE static void SceneMatrixImpl(NavIndex_t const & /*nav_tuple*/,
                                                      Transformation3DMP<Real_t> & /*trans*/)
  {
  }

  VECCORE_ATT_HOST_DEVICE
  static void SceneMatrixImpl(NavIndex_t const & /*nav_tuple*/, Transformation3D & /*trans*/) {}

  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE static Vector3D<Real_t> GlobalToLocalImpl(NavIndex_t nav_ind,
                                                                    Vector3D<Real_t> const &globalpoint)
  {
    Transformation3DMP<Real_t> trans;
    TopMatrixImpl(nav_ind, trans);
    Vector3D<Real_t> local = trans.Transform(globalpoint);
    return local;
  }

  VECCORE_ATT_HOST_DEVICE
  static Vector3D<Precision> GlobalToLocalImpl(NavIndex_t nav_ind, Vector3D<Precision> const &globalpoint)
  {
    Transformation3D trans;
    TopMatrixImpl(nav_ind, trans);
    Vector3D<Precision> local = trans.Transform(globalpoint);
    return local;
  }

  // Intrerface methods
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *GetLastExited() const { return TopImpl(fLastExited); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  int GetLastIdExited() const { return TopIdImpl(fLastExited); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetLastExited() { fLastExited = fNavInd; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetNavIndex(NavIndex_t navind) { fNavInd = navind; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned short GetNdaughters() const { return GetNdaughtersImpl(fNavInd); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t GetId() const { return GetIdImpl(fNavInd); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavIndex_t GetParentSceneTopId() const { return 0; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool GetSceneId(unsigned short &scene_id, unsigned short &newscene_id) const
  {
    scene_id = newscene_id = 0;
    return false;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned int GetSceneLevel() const { return 0; }

  /// @brief Implementation for getting the parent scene id
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned short GetParentScene() const { return 0; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned int GetLogicalId() const { return GetLogicalIdImpl(fNavInd); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int GetChildId() const { return GetChildIdImpl(fNavInd); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsScene() const { return false; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsDescendent(NavIndex_t parent) const { return IsDescendentImpl(fNavInd, parent); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Push(VPlacedVolume const *v) { PushImpl(fNavInd, v); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Push(int iplaced) { PushImpl(fNavInd, iplaced); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void PushDaughter(int idaughter) { PushDaughterImpl(fNavInd, idaughter); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void PushScene(NavIndex_t) {}

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Pop() { PopImpl(fNavInd); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void PopScene() { fNavInd = 0; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *Top() const { return TopImpl(fNavInd); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  int TopId() const { return TopIdImpl(fNavInd); }

  /**
   * returns the number of FILLED LEVELS such that
   * state.GetNode( state.GetLevel() ) == state.Top()
   */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  unsigned char GetLevel() const { return GetLevelImpl(fNavInd); }

  /** Compatibility getter for NavigationState interface */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  unsigned char GetCurrentLevel() const { return GetLevel() + 1; }

  /**
   * Returns the navigation index for a level smaller/equal than the current level.
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  NavIndex_t GetNavIndex(int level) const { return GetNavIndexImpl(fNavInd, level); }

  /**
   * Returns the placed volume at a evel smaller/equal than the current level.
   */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *At(int level) const
  {
    auto parent = GetNavIndexImpl(fNavInd, level);
    return (parent > 0) ? ToPlacedVolume(NavInd(parent + 2)) : nullptr;
  }

  /**
   * Returns the index of a placed volume at a evel smaller/equal than the current level.
   */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  size_t ValueAt(int level) const
  {
    auto parent = GetNavIndexImpl(fNavInd, level);
    return (parent > 0) ? (size_t)NavInd(parent + 2) : 0;
  }

  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void TopMatrix(Transformation3DMP<Real_t> &trans) const
  {
    TopMatrixImpl(fNavInd, trans);
  }

  VECCORE_ATT_HOST_DEVICE
  void TopMatrix(Transformation3D &trans) const { TopMatrixImpl(fNavInd, trans); }

  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void TopMatrix(int tolevel, Transformation3DMP<Real_t> &trans) const
  {
    TopMatrixImpl(GetNavIndexImpl(fNavInd, tolevel), trans);
  }

  VECCORE_ATT_HOST_DEVICE
  void TopMatrix(int tolevel, Transformation3D &trans) const
  {
    TopMatrixImpl(GetNavIndexImpl(fNavInd, tolevel), trans);
  }

  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void TopInSceneMatrix(Transformation3DMP<Real_t> &trans) const
  {
    TopInSceneMatrixImpl(fNavInd, trans);
  }

  VECCORE_ATT_HOST_DEVICE
  void TopInSceneMatrix(Transformation3D &trans) const { TopInSceneMatrixImpl(fNavInd, trans); }

  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void SceneMatrix(Transformation3DMP<Real_t> & /*trans*/) const
  {
  }

  VECCORE_ATT_HOST_DEVICE
  void SceneMatrix(Transformation3D & /*trans*/) const {}

  // returning a "delta" transformation that can transform
  // coordinates given in reference frame of this->Top() to the reference frame of other->Top()
  // simply with otherlocalcoordinate = delta.Transform( thislocalcoordinate )
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE void DeltaTransformation(NavStateIndex const &other, Transformation3DMP<Real_t> &delta) const;

  VECCORE_ATT_HOST_DEVICE
  void DeltaTransformation(NavStateIndex const &other, Transformation3D &delta) const;

  // VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GlobalToLocal(Vector3D<Precision> const &localpoint) const
  {
    return GlobalToLocalImpl(fNavInd, localpoint);
  }

  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GlobalToLocal(Vector3D<Precision> const &localpoint, int tolevel) const
  {
    return GlobalToLocalImpl(GetNavIndexImpl(fNavInd, tolevel), localpoint);
  }

  /**
   * calculates if other navigation state takes a different branch in geometry path or is on same branch
   * ( two states are on same branch if one can connect the states just by going upwards or downwards ( or do nothing ))
   */
  VECCORE_ATT_HOST_DEVICE
  int Distance(NavStateIndex const &) const;

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
  std::string RelativePath(NavStateIndex const & /*other*/) const;

  // functions useful to "serialize" navigationstate
  // the Vector-of-Indices basically describes the path on the tree taken from top to bottom
  // an index corresponds to a daughter

  void GetPathAsListOfIndices(std::list<uint> &indices) const;
  void ResetPathFromListOfIndices(VPlacedVolume const *world, std::list<uint> const &indices);

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
    fNavInd     = 0;
    fLastExited = 0;
    fOnBoundary = false;
  }

  VECCORE_ATT_HOST_DEVICE
  void Print() const;

  VECCORE_ATT_HOST_DEVICE
  static void PrintTopImpl(NavIndex_t nav_ind) { NavStateIndex(nav_ind).Print(); }

  VECCORE_ATT_HOST_DEVICE
  void PrintTop() const { Print(); }

  VECCORE_ATT_HOST_DEVICE
  void Dump() const { Print(); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool HasSamePathAsOther(NavStateIndex const &other) const { return (fNavInd == other.fNavInd); }

  void printValueSequence(std::ostream & = std::cerr) const;

  // calculates a checksum along the path
  // can be used (as a quick criterion) to see whether 2 states are same
  unsigned long getCheckSum() const { return (unsigned long)fNavInd; }

  /**
    function returning whether the point (current navigation state) is outside the detector setup
  */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsOutside() const { return (fNavInd == 0); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsOnBoundary() const { return fOnBoundary; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetBoundaryState(bool b) { fOnBoundary = b; }
};

struct ExitSurfState {
  int common_id;
  int frame_id;
  bool left_side;
  bool overlap;

  VECCORE_ATT_HOST_DEVICE
  ExitSurfState(int common_surf = 0, int frame = 0, bool ls = 0, bool ol = 0)
      : common_id(common_surf), frame_id(frame), left_side(ls), overlap(ol)
  {
  }
};

/**
 * encodes the geometry path as a concatenated string of ( Value_t ) present in fPath
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
