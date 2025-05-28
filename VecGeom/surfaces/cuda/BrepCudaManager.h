#ifndef VECGEOM_SURFACE_BREPCUDAMANAGER_H_
#define VECGEOM_SURFACE_BREPCUDAMANAGER_H_

#include <VecGeom/surfaces/SurfData.h>
#include "VecGeom/surfaces/bvh/AABBsurf.h"
#include "VecGeom/surfaces/bvh/BVHsurf.h"
#include "VecGeom/volumes/VolumeTree.h"
#include "VecGeom/management/Logger.h"
#include "VecGeom/base/Assert.h"

namespace vgbrep {

template <typename Real_t>
__global__ void FinishBVHCopy(bvh::BVHsurf<Real_t> *dBVH, int *dPrimId, int *dOffset, int *dNChild,
                              bvh::AABBsurf<typename SurfData<Real_t>::Real_b> *dAABBs,
                              bvh::AABBsurf<typename SurfData<Real_t>::Real_b> *dNodes)
{
  if (dBVH == nullptr) {
    printf("Error: Null pointer 'dBVH' encountered in FinishBVHCopy\n");
    return;
  }
  if (dPrimId == nullptr) {
    printf("Error: Null pointer 'dPrimId' encountered in FinishBVHCopy\n");
    return;
  }
  if (dOffset == nullptr) {
    printf("Error: Null pointer 'dOffset' encountered in FinishBVHCopy\n");
    return;
  }
  if (dNChild == nullptr) {
    printf("Error: Null pointer 'dNChild' encountered in FinishBVHCopy\n");
    return;
  }
  if (dAABBs == nullptr) {
    printf("Error: Null pointer 'dAABBs' encountered in FinishBVHCopy\n");
    return;
  }
  if (dNodes == nullptr) {
    printf("Error: Null pointer 'dNodes' encountered in FinishBVHCopy\n");
    return;
  }
  dBVH->SetPointers(dPrimId, dOffset, dNChild, dAABBs, dNodes);
}

template <typename Real_t>
void CopyBVH(const bvh::BVHsurf<Real_t> &hBVH, bvh::BVHsurf<Real_t> *dBVH)
{
  int *dPrimId;
  int *dOffset;
  int *dNChild;
  bvh::AABBsurf<Real_t> *dNodes;
  bvh::AABBsurf<Real_t> *dAABBs;

  int rootNChild = hBVH.GetRootNChild();
  if (rootNChild <= 0) {
    std::ostringstream oss;
    oss << "Invalid number of root children: " << rootNChild;
    throw std::logic_error(oss.str());
  }
  int nodes = (2 << hBVH.GetDepth()) - 1;
  if (nodes <= 0) {
    std::ostringstream oss;
    oss << "Invalid number of nodes: " << nodes;
    throw std::logic_error(oss.str());
  }

  VECGEOM_DEVICE_API_CALL(Malloc(&dPrimId, hBVH.GetRootNChild() * sizeof(int)));
  VECGEOM_DEVICE_API_CALL(Malloc(&dOffset, nodes * sizeof(int)));
  VECGEOM_DEVICE_API_CALL(Malloc(&dNChild, nodes * sizeof(int)));
  VECGEOM_DEVICE_API_CALL(Malloc(&dNodes, nodes * sizeof(bvh::AABBsurf<Real_t>)));
  VECGEOM_DEVICE_API_CALL(Malloc(&dAABBs, hBVH.GetRootNChild() * sizeof(bvh::AABBsurf<Real_t>)));

  // Ensure pointers are not null after allocation
  if (!dPrimId || !dAABBs || !dOffset || !dNChild || !dNodes) {
    throw std::runtime_error("Memory allocation failed: One or more pointers are null.");
  }

  VECGEOM_DEVICE_API_CALL(Memcpy(dPrimId, hBVH.GetPrimId(), hBVH.GetRootNChild() * sizeof(int),
                                 VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
  VECGEOM_DEVICE_API_CALL(
      Memcpy(dOffset, hBVH.GetOffset(), nodes * sizeof(int), VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
  VECGEOM_DEVICE_API_CALL(
      Memcpy(dNChild, hBVH.GetNChild(), nodes * sizeof(int), VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
  VECGEOM_DEVICE_API_CALL(Memcpy(dNodes, hBVH.GetNodes(), nodes * sizeof(bvh::AABBsurf<Real_t>),
                                 VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
  VECGEOM_DEVICE_API_CALL(Memcpy(dAABBs, hBVH.GetAABBs(), hBVH.GetRootNChild() * sizeof(bvh::AABBsurf<Real_t>),
                                 VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

  // Adjust pointers in the GPU instance
  FinishBVHCopy<<<1, 1>>>(dBVH, dPrimId, dOffset, dNChild, dAABBs, dNodes);
}

// This function sets the correct pointers on device memory in the data structures that were copied
static __global__ void FinishVolumeTreeTransfer(vecgeom::VolumeTree *volumeTree, long shift_children)
{
  volumeTree->Relocate(shift_children);
  vecgeom::globaldevicegeomdata::gVolumeTree = volumeTree;
}

// This function sets the correct pointers on device memory in the data structures that were copied
template <typename Real_t>
static __global__ void BrepCudaManagerFinishTransfer(SurfData<Real_t> *surfData)
{
  int *current, *current_exiting_surface, *current_entering_surface, *current_entering_surface_pvol,
      *current_entering_surface_pvol_trans, *current_entering_surface_lvol_id, *current_daughter_pvol_id,
      *current_daughter_pvol_trans;
  logic_int *current_logic;
  globaldevicesurfdata::gSurfDataDevice<Real_t> = surfData;

  // Write pointers into fShells[i].fSurfaces, fShells[i].fLogic, fShells[i].fShellEnteringSurfaceList,
  // fShells[i].fShellEnteringSurfaceTransList, fShells[i].fShellEnteringSurfacePvolList
  current                             = surfData->fSurfShellList;
  current_logic                       = surfData->fLogicList;
  current_exiting_surface             = surfData->fShellExitingSurfaceList;
  current_entering_surface            = surfData->fShellEnteringSurfaceList;
  current_entering_surface_pvol       = surfData->fShellEnteringSurfacePvolList;
  current_entering_surface_pvol_trans = surfData->fShellEnteringSurfacePvolTransList;
  current_entering_surface_lvol_id    = surfData->fShellEnteringSurfaceLvolIdList;
  current_daughter_pvol_id            = surfData->fShellDaughterPvolIdList;
  current_daughter_pvol_trans         = surfData->fShellDaughterPvolTransList;
  for (int i = 0; i < surfData->fNshells; i++) {
    surfData->fShells[i].fSurfaces = current;
    current += surfData->fShells[i].fNsurf;
    surfData->fShells[i].fLogic.data_ = current_logic;
    current_logic += surfData->fShells[i].fLogic.size();
    surfData->fShells[i].fExitingSurfaces = current_exiting_surface;
    current_exiting_surface += surfData->fShells[i].fNExitingSurfaces;
    surfData->fShells[i].fEnteringSurfaces = current_entering_surface;
    current_entering_surface += surfData->fShells[i].fNEnteringSurfaces;
    surfData->fShells[i].fEnteringSurfacesPvol = current_entering_surface_pvol;
    current_entering_surface_pvol += surfData->fShells[i].fNEnteringSurfaces;
    surfData->fShells[i].fEnteringSurfacesPvolTrans = current_entering_surface_pvol_trans;
    current_entering_surface_pvol_trans += surfData->fShells[i].fNEnteringSurfaces;
    surfData->fShells[i].fEnteringSurfacesLvolIds = current_entering_surface_lvol_id;
    current_entering_surface_lvol_id += surfData->fShells[i].fNEnteringSurfaces;
    surfData->fShells[i].fDaughterPvolIds = current_daughter_pvol_id;
    current_daughter_pvol_id += surfData->fShells[i].fNDaughterPvols;
    surfData->fShells[i].fDaughterPvolTrans = current_daughter_pvol_trans;
    current_daughter_pvol_trans += surfData->fShells[i].fNDaughterPvols;
  }

  // Write pointers into fCommonSurfaces[i].f{Left,Right}Side.fSurfaces
  current = surfData->fSides;
  for (int i = 0; i < surfData->fNcommonSurf; i++) {
    surfData->fCommonSurfaces[i].fLeftSide.fSurfaces = current;
    current += surfData->fCommonSurfaces[i].fLeftSide.fNsurf;
    surfData->fCommonSurfaces[i].fRightSide.fSurfaces = current;
    current += surfData->fCommonSurfaces[i].fRightSide.fNsurf;
  }

  // Write pointers into fSideDivisions[i].fSlices and fSlices[i].fCandidates
  auto currentSlice = surfData->fSlices;
  current           = surfData->fSliceCandidates;
  for (int i = 0; i < surfData->fNsideDivisions; i++) {
    surfData->fSideDivisions[i].fSlices = currentSlice;
    for (int j = 0; j < surfData->fSideDivisions[i].fNslices; ++j) {
      currentSlice->fCandidates = current;
      current += currentSlice->fNcand;
      currentSlice++;
    }
  }

  // Write pointers into fCandidates[i].{fCandidates,fFrameInd, fSides}
  current = surfData->fCandList;
  for (int i = 0; i < surfData->fNStates; i++) {
    surfData->fCandidates[i].fCandidates = current;
    // Move the pointer to the start of the Frame index list
    int ncand = surfData->fCandidates[i].fNcand;
    current += ncand;
    surfData->fCandidates[i].fFrameInd = current;
    // Move the pointer to the start of the sides index list
    current += ncand;
    surfData->fCandidates[i].fSides = reinterpret_cast<char *>(current);
    // Move the pointer to the start of the next Candidate index list
    int add_one = ((ncand * sizeof(char)) % sizeof(int)) > 0 ? 1 : 0;
    current += ncand * sizeof(char) / sizeof(int) + add_one;
  }
}

// Manager class for synchronizing the surface data to the GPU.
template <typename Real_t>
class BrepCudaManager {
  using SurfData_t = SurfData<Real_t>;

  SurfData_t fSurfDataStaging;               ///< Host memory to stage data for the GPU
  SurfData_t *fSurfData{nullptr};            ///< Device pointer to the data structure
  vecgeom::VolumeTree fVolumeTreeStaging;    ///< Host memory to stage the volume tree for the GPU
  vecgeom::VolumeTree *fVolumeTree{nullptr}; ///< Device pointer to the volume tree

public:
  static BrepCudaManager &Instance()
  {
    static BrepCudaManager instance;
    return instance;
  }

  const SurfData_t *GetDevicePtr() const { return fSurfData; }
  const vecgeom::VolumeTree *GetVolumeTreeDevicePtr() const { return fVolumeTree; }

  void TransferVolumeTree(const vecgeom::VolumeTree &volumeTree)
  {
    fVolumeTreeStaging  = volumeTree;
    size_t sizeLogical  = volumeTree.fNlogical * sizeof(vecgeom::LogicalId);
    size_t sizePlaced   = volumeTree.fNplaced * sizeof(vecgeom::PlacedId);
    size_t sizeChildren = volumeTree.fNplacedC * sizeof(vecgeom::PlacedId);
    // Allocate the buffer on device
    VECGEOM_DEVICE_API_CALL(Malloc(&fVolumeTreeStaging.fLogical, sizeLogical));
    VECGEOM_DEVICE_API_CALL(Malloc(&fVolumeTreeStaging.fPlaced, sizePlaced));
    VECGEOM_DEVICE_API_CALL(Malloc(&fVolumeTreeStaging.fChildren, sizeChildren));
    VECGEOM_DEVICE_API_CALL(Memcpy(fVolumeTreeStaging.fLogical, volumeTree.fLogical, sizeLogical,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
    VECGEOM_DEVICE_API_CALL(Memcpy(fVolumeTreeStaging.fPlaced, volumeTree.fPlaced, sizePlaced,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
    VECGEOM_DEVICE_API_CALL(Memcpy(fVolumeTreeStaging.fChildren, volumeTree.fChildren, sizeChildren,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
    // long shift_placed = reinterpret_cast<long>(fVolumeTreeStaging.fPlaced) -
    // reinterpret_cast<long>(volumeTree.fPlaced);
    long shift_children =
        reinterpret_cast<long>(fVolumeTreeStaging.fChildren) - reinterpret_cast<long>(volumeTree.fChildren);

    // Now copy the staged data to the GPU
    VECGEOM_DEVICE_API_CALL(Malloc(&fVolumeTree, sizeof(vecgeom::VolumeTree)));
    VECGEOM_DEVICE_API_CALL(Memcpy(fVolumeTree, &fVolumeTreeStaging, sizeof(vecgeom::VolumeTree),
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Finally finish the transfer by calling a kernel to write some pointers
    FinishVolumeTreeTransfer<<<1, 1>>>(fVolumeTree, shift_children);
    VECGEOM_DEVICE_API_CALL(DeviceSynchronize());
    // The arrays in the staging area are device pointers, null them to avoid deletion
    fVolumeTreeStaging.fLogical  = nullptr;
    fVolumeTreeStaging.fPlaced   = nullptr;
    fVolumeTreeStaging.fChildren = nullptr;
  }

  void TransferSurfData(const SurfData_t &surfData)
  {
    using Real_b = typename SurfData_t::Real_b;
    size_t sizeInBytes;

    // Transfer the volume tree first
    auto const &volumeTree = vecgeom::VolumeTree::Instance();
    if (!volumeTree.fValid) {
      VECGEOM_LOG(critical) << "Volume tree is invalid";
      return;
    }
    TransferVolumeTree(volumeTree);

    // Allocate and copy transformations
    fSurfDataStaging.fNvolTrans = surfData.fNvolTrans;
    sizeInBytes                 = sizeof(surfData.fPVolTrans[0]) * surfData.fNvolTrans;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fPVolTrans, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fPVolTrans, surfData.fPVolTrans, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Allocate and copy surface data
    fSurfDataStaging.fNellip = surfData.fNellip;
    sizeInBytes              = sizeof(surfData.fEllipData[0]) * surfData.fNellip;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fEllipData, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fEllipData, surfData.fEllipData, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    fSurfDataStaging.fNtorus = surfData.fNtorus;
    sizeInBytes              = sizeof(surfData.fTorusData[0]) * surfData.fNtorus;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fTorusData, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fTorusData, surfData.fTorusData, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    fSurfDataStaging.fNarb4 = surfData.fNarb4;
    sizeInBytes             = sizeof(surfData.fArb4Data[0]) * surfData.fNarb4;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fArb4Data, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fArb4Data, surfData.fArb4Data, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Allocate and copy volume shells
    fSurfDataStaging.fNshells = surfData.fNshells;
    sizeInBytes               = sizeof(surfData.fShells[0]) * surfData.fNshells;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fShells, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(
        Memcpy(fSurfDataStaging.fShells, surfData.fShells, sizeInBytes, VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Nota bene: the fShells[i].fSurfaces are backed by the following array
    // and set via BrepCudaManagerFinishTransfer.
    sizeInBytes = sizeof(surfData.fSurfShellList[0]) * surfData.fNlocalSurf;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fSurfShellList, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fSurfShellList, surfData.fSurfShellList, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Nota bene: fShells[i].fLogic are backed by the following array
    // and set via BrepCudaManagerFinishTransfer.
    sizeInBytes = sizeof(logic_int) * surfData.fNlogic;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fLogicList, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fLogicList, surfData.fLogicList, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Nota bene: fShells[i].fShellExiting/EnteringSurfaceList are backed by the following array
    // and set via BrepCudaManagerFinishTransfer.
    fSurfDataStaging.fNExitingSurfaces  = surfData.fNExitingSurfaces;
    fSurfDataStaging.fNEnteringSurfaces = surfData.fNEnteringSurfaces;

    sizeInBytes = sizeof(surfData.fShellExitingSurfaceList[0]) * surfData.fNExitingSurfaces;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fShellExitingSurfaceList, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fShellExitingSurfaceList, surfData.fShellExitingSurfaceList,
                                   sizeInBytes, VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
    sizeInBytes = sizeof(surfData.fShellEnteringSurfaceList[0]) * surfData.fNEnteringSurfaces;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fShellEnteringSurfaceList, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fShellEnteringSurfaceList, surfData.fShellEnteringSurfaceList,
                                   sizeInBytes, VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Nota bene: fShells[i].fShellEnteringSurfacePvolList are backed by the following array
    // and set via BrepCudaManagerFinishTransfer.
    sizeInBytes = sizeof(surfData.fShellEnteringSurfacePvolList[0]) * surfData.fNEnteringSurfaces;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fShellEnteringSurfacePvolList, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fShellEnteringSurfacePvolList,
                                   surfData.fShellEnteringSurfacePvolList, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    sizeInBytes = sizeof(surfData.fShellEnteringSurfacePvolTransList[0]) * surfData.fNEnteringSurfaces;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fShellEnteringSurfacePvolTransList, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fShellEnteringSurfacePvolTransList,
                                   surfData.fShellEnteringSurfacePvolTransList, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    sizeInBytes = sizeof(surfData.fShellEnteringSurfaceLvolIdList[0]) * surfData.fNEnteringSurfaces;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fShellEnteringSurfaceLvolIdList, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fShellEnteringSurfaceLvolIdList,
                                   surfData.fShellEnteringSurfaceLvolIdList, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    sizeInBytes = sizeof(surfData.fShellDaughterPvolIdList[0]) * surfData.fNPlacedVolumes;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fShellDaughterPvolIdList, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fShellDaughterPvolIdList, surfData.fShellDaughterPvolIdList,
                                   sizeInBytes, VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    sizeInBytes = sizeof(surfData.fShellDaughterPvolTransList[0]) * surfData.fNPlacedVolumes;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fShellDaughterPvolTransList, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fShellDaughterPvolTransList, surfData.fShellDaughterPvolTransList,
                                   sizeInBytes, VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Allocate space for the BVHs
    // Surface BVHs
    sizeInBytes = sizeof(surfData.fBVH[0]) * surfData.fNshells;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fBVH, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(
        Memcpy(fSurfDataStaging.fBVH, surfData.fBVH, sizeInBytes, VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
    // Solid BVHs
    sizeInBytes = sizeof(surfData.fBVHSolids[0]) * surfData.fNshells;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fBVHSolids, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fBVHSolids, surfData.fBVHSolids, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Allocate and copy BVH members
    for (int i = 0; i < surfData.fNshells; ++i) {
      // Surface BVHs
      if (surfData.fShells[i].fNsurf > 0) { // this checks that the BVH is actually populated and not null
        auto const &hBVH = surfData.fBVH[i];
        auto dBVH        = &(fSurfDataStaging.fBVH[i]);
        CopyBVH<Real_b>(hBVH, dBVH);
      }
      // Solid BVHs
      if (surfData.fShells[i].fNEnteringSurfaces >
          0) { // If there are no entering surfaces, the volume has no daughters
        auto const &hBVH = surfData.fBVHSolids[i];
        auto dBVH        = &(fSurfDataStaging.fBVHSolids[i]);
        CopyBVH<Real_b>(hBVH, dBVH);
      }
    }

    // Allocate and copy surfaces
    fSurfDataStaging.fNlocalSurf = surfData.fNlocalSurf;
    sizeInBytes                  = sizeof(surfData.fLocalSurf[0]) * surfData.fNlocalSurf;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fLocalSurf, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fLocalSurf, surfData.fLocalSurf, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    fSurfDataStaging.fNglobalSurf = surfData.fNglobalSurf;
    sizeInBytes                   = sizeof(surfData.fFramedSurf[0]) * surfData.fNglobalSurf;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fFramedSurf, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fFramedSurf, surfData.fFramedSurf, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Allocate and copy masks
    fSurfDataStaging.fNwindows = surfData.fNwindows;
    sizeInBytes                = sizeof(surfData.fWindowMasks[0]) * surfData.fNwindows;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fWindowMasks, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fWindowMasks, surfData.fWindowMasks, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    fSurfDataStaging.fNrings = surfData.fNrings;
    sizeInBytes              = sizeof(surfData.fRingMasks[0]) * surfData.fNrings;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fRingMasks, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fRingMasks, surfData.fRingMasks, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    fSurfDataStaging.fNzphis = surfData.fNzphis;
    sizeInBytes              = sizeof(surfData.fZPhiMasks[0]) * surfData.fNzphis;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fZPhiMasks, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fZPhiMasks, surfData.fZPhiMasks, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    fSurfDataStaging.fNquads = surfData.fNquads;
    sizeInBytes              = sizeof(surfData.fQuadMasks[0]) * surfData.fNquads;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fQuadMasks, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fQuadMasks, surfData.fQuadMasks, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    fSurfDataStaging.fNtriangs = surfData.fNtriangs;
    sizeInBytes                = sizeof(surfData.fTriangleMasks[0]) * surfData.fNtriangs;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fTriangleMasks, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fTriangleMasks, surfData.fTriangleMasks, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Allocate and copy scene indices
    fSurfDataStaging.fNscenes = surfData.fNscenes;
    sizeInBytes               = sizeof(int) * surfData.fNscenes;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fSceneStartIndex, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fSceneTouchables, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fSceneStartIndex, surfData.fSceneStartIndex, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fSceneTouchables, surfData.fSceneTouchables, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Allocate and copy common surfaces
    fSurfDataStaging.fNcommonSurf = surfData.fNcommonSurf;
    sizeInBytes                   = sizeof(surfData.fCommonSurfaces[0]) * surfData.fNcommonSurf;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fCommonSurfaces, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fCommonSurfaces, surfData.fCommonSurfaces, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Nota bene: the fCommonSurfaces[i].f{Left,Right}Side.fSurfaces are backed
    // by the following array and set via BrepCudaManagerFinishTransfer.
    fSurfDataStaging.fNsides = surfData.fNsides;
    sizeInBytes              = sizeof(surfData.fSides[0]) * surfData.fNsides;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fSides, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(
        Memcpy(fSurfDataStaging.fSides, surfData.fSides, sizeInBytes, VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Allocate and copy side divisions
    // Nota bene: the fSideDivisions[i].fSlices and fSlices[i].fCandidates are backed
    // by the following arrays and set via BrepCudaManagerFinishTransfer.
    fSurfDataStaging.fNsideDivisions   = surfData.fNsideDivisions;
    fSurfDataStaging.fNslices          = surfData.fNslices;
    fSurfDataStaging.fNsliceCandidates = surfData.fNsliceCandidates;
    sizeInBytes                        = sizeof(surfData.fSideDivisions[0]) * surfData.fNsideDivisions;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fSideDivisions, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fSideDivisions, surfData.fSideDivisions, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
    sizeInBytes = sizeof(surfData.fSlices[0]) * surfData.fNslices;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fSlices, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(
        Memcpy(fSurfDataStaging.fSlices, surfData.fSlices, sizeInBytes, VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
    sizeInBytes = sizeof(int) * surfData.fNsliceCandidates;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fSliceCandidates, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fSliceCandidates, surfData.fSliceCandidates, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Allocate and copy candidates lists
    fSurfDataStaging.fNStates = surfData.fNStates;
    sizeInBytes               = sizeof(surfData.fCandidates[0]) * surfData.fNStates;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fCandidates, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fCandidates, surfData.fCandidates, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Nota bene: the fCandidates[i].{fCandidates,fFrameInd,fSides} are backed by the
    // following array and set via BrepCudaManagerFinishTransfer.
    fSurfDataStaging.fSizeCandList = surfData.fSizeCandList;
    sizeInBytes                    = sizeof(surfData.fCandList[0]) * surfData.fSizeCandList;
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfDataStaging.fCandList, sizeInBytes));
    VECGEOM_DEVICE_API_CALL(Memcpy(fSurfDataStaging.fCandList, surfData.fCandList, sizeInBytes,
                                   VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Now copy the staged data to the GPU
    VECGEOM_DEVICE_API_CALL(Malloc(&fSurfData, sizeof(SurfData_t)));
    VECGEOM_DEVICE_API_CALL(
        Memcpy(fSurfData, &fSurfDataStaging, sizeof(SurfData_t), VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // Finally finish the transfer by calling a kernel to write some pointers
    BrepCudaManagerFinishTransfer<<<1, 1>>>(fSurfData);
    VECGEOM_DEVICE_API_CALL(DeviceSynchronize());
  }

  void Cleanup()
  {
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fPVolTrans));
    fSurfDataStaging.fPVolTrans = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fEllipData));
    fSurfDataStaging.fEllipData = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fTorusData));
    fSurfDataStaging.fTorusData = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fArb4Data));
    fSurfDataStaging.fArb4Data = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fShells));
    fSurfDataStaging.fShells = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fSurfShellList));
    fSurfDataStaging.fSurfShellList = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fLogicList));
    fSurfDataStaging.fLogicList = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fShellExitingSurfaceList));
    fSurfDataStaging.fShellExitingSurfaceList = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fShellEnteringSurfaceList));
    fSurfDataStaging.fShellEnteringSurfaceList = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fShellEnteringSurfacePvolList));
    fSurfDataStaging.fShellEnteringSurfacePvolList = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fShellEnteringSurfacePvolTransList));
    fSurfDataStaging.fShellEnteringSurfacePvolTransList = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fLocalSurf));
    fSurfDataStaging.fLocalSurf = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fFramedSurf));
    fSurfDataStaging.fFramedSurf = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fWindowMasks));
    fSurfDataStaging.fWindowMasks = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fRingMasks));
    fSurfDataStaging.fRingMasks = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fZPhiMasks));
    fSurfDataStaging.fZPhiMasks = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fQuadMasks));
    fSurfDataStaging.fQuadMasks = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fCommonSurfaces));
    fSurfDataStaging.fCommonSurfaces = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fSides));
    fSurfDataStaging.fSides = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fCandidates));
    fSurfDataStaging.fCandidates = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfDataStaging.fCandList));
    fSurfDataStaging.fCandList = nullptr;
    VECGEOM_DEVICE_API_CALL(Free(fSurfData));
    fSurfData = nullptr;
  }
};

} // namespace vgbrep

#endif
