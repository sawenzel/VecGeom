// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Volume tree and hierarchical elements data structures
/// \file VolumeTree.h
/// \author Andrei Gheata (CERN)

#ifndef VECGEOM_VOLUMES_VOLUMETREE_H_
#define VECGEOM_VOLUMES_VOLUMETREE_H_

#include "VecGeom/management/DeviceGlobals.h"

namespace vecgeom {

struct PlacedId;

/// @brief Structure representing a logical volume
struct LogicalId {
  int fId{-1};                                ///< Logical volume id
  int fNplaced{0};                            ///< Number of children placed volumes
  ESolidType fSolidType{ESolidType::nosolid}; ///< Solid type (see Global.h)
  PlacedId *fChildren{nullptr};               ///< Array of children placed volumes

  LogicalId() = default;
  LogicalId(int ivol, ESolidType type, int nchildren, PlacedId *nodes) { Set(ivol, type, nchildren, nodes); }

  bool operator==(LogicalId const &other) const
  {
    if (fId != other.fId) return false;
    if (fNplaced != other.fNplaced) return false;
    if (fChildren != other.fChildren) return false;
    return true;
  }

  bool operator!=(LogicalId const &other) const { return !operator==(other); }

  void Set(int ivol, ESolidType type, int nchildren, PlacedId *nodes)
  {
    fId        = ivol;
    fNplaced   = nchildren;
    fSolidType = type;
    fChildren  = nodes;
  }

  VECGEOM_FORCE_INLINE
  PlacedId const &GetPlacedId(int ichild);
};

/// @brief Structure representing a placed volume
struct PlacedId {
  int fId{-1};       ///< Placed volume id
  int fCopyNo{-1};   ///< Copy number
  int fChildId{-1};  ///< Index in the parent volume list
  LogicalId fVolume; ///< Logical volume

  PlacedId() = default;
  PlacedId(int id, int icopy, int ichild, LogicalId const &lvol) { Set(id, icopy, ichild, lvol); }
  void Set(int id, int icopy, int ichild, LogicalId const &lvol)
  {
    fId      = id;
    fCopyNo  = icopy;
    fChildId = ichild;
    fVolume  = lvol;
  }

  bool operator==(PlacedId const &other) const
  {
    if (fId != other.fId) return false;
    if (fCopyNo != other.fCopyNo) return false;
    if (fChildId != other.fChildId) return false;
    if (fVolume != other.fVolume) return false;
    return true;
  }

  bool operator!=(PlacedId const &other) const { return !operator==(other); }

  void Place(LogicalId &parent)
  {
    VECGEOM_VALIDATE(parent.fChildren, << "Parent volume has no children space allocated");
    parent.fChildren[parent.fNplaced++] = *this;
  }

  VECCORE_ATT_HOST_DEVICE
  void Print()
  {
    printf("PlacedId %d: copyNo = %d  childId = %d  volumeId = %d  nchildren = %d\n   children: ", fId, fCopyNo,
           fChildId, fVolume.fId, fVolume.fNplaced);
    for (auto i = 0; i < fVolume.fNplaced; ++i)
      printf("%d  ", fVolume.fChildren[i].fId);
    printf("\n");
  }
};

VECGEOM_FORCE_INLINE
PlacedId const &LogicalId::GetPlacedId(int ichild) { return fChildren[ichild]; }

struct VolumeTree {
  PlacedId fWorld;
  int fNlogical{0};             ///< Number of registered logical volumes
  int fNplaced{0};              ///< Number of registered placed volumes
  int fNplacedC{0};             ///< Number of placed children
  LogicalId *fLogical{nullptr}; ///< List of logical volumes
  PlacedId *fPlaced{nullptr};   ///< List of placed volumes
  PlacedId *fChildren{nullptr}; ///< List of placed volumes
  bool fValid{false};           ///< Validity

  VolumeTree() = default;
  ~VolumeTree()
  {
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
    // Only delete arrays on host. On device, one needs to delete globaldevicegeomdata::gVolumeTree
    delete[] fLogical;
    delete[] fPlaced;
    delete[] fChildren;
    fLogical  = nullptr;
    fPlaced   = nullptr;
    fChildren = nullptr;
#endif
  }

  size_t GetSize() const
  {
    size_t sizeInBytes = sizeof(VolumeTree) + fNlogical * sizeof(LogicalId) + (fNplaced + fNplacedC) * sizeof(PlacedId);
    return sizeInBytes;
  }

  /// @brief Allocate the volume tree in a contiguous buffer
  /// @param nlogical
  /// @param nplaced
  void Allocate(int nlogical, int nplaced, int nplaced_children)
  {
    fNlogical = nlogical;
    fNplaced  = nplaced;
    fNplacedC = nplaced_children;
    fLogical  = new LogicalId[nlogical];
    fPlaced   = new PlacedId[nplaced];
    fChildren = new PlacedId[nplaced_children];
  }

  /// @brief Relocate the internal data
  /// @param shift_bytes shift of placed id children array
  VECCORE_ATT_HOST_DEVICE
  void Relocate(long shift_bytes)
  {
    auto moveAddress = [shift_bytes](PlacedId *address) {
      return reinterpret_cast<PlacedId *>((char *)address + shift_bytes);
    };
    if (fWorld.fVolume.fChildren) fWorld.fVolume.fChildren = moveAddress(fWorld.fVolume.fChildren);
    for (auto i = 0; i < fNlogical; ++i)
      if (fLogical[i].fChildren) fLogical[i].fChildren = moveAddress(fLogical[i].fChildren);
    for (auto i = 0; i < fNplaced; ++i)
      if (fPlaced[i].fVolume.fChildren) fPlaced[i].fVolume.fChildren = moveAddress(fPlaced[i].fVolume.fChildren);
    // Debug only
    // for (auto i = 0; i < fNplaced; ++i)
    //   fPlaced[i].Print();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static VolumeTree &Instance()
  {
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
    return *globaldevicegeomdata::gVolumeTree;
#else
    static VolumeTree gVolumeTree;
    return gVolumeTree;
#endif
  }
};

} // namespace vecgeom
#endif