#ifndef VECGEOM_SURFACE_BREPCUDAMANAGER_H_
#define VECGEOM_SURFACE_BREPCUDAMANAGER_H_

// This header file can only be used from CUDA sources because it needs to call
// functions from the CUDA Runtime API and invoke a kernel.
#ifdef __CUDACC__

#include <VecGeom/surfaces/SurfData.h>
#include "VecGeom/surfaces/bvh/AABBsurf.h"
#include "VecGeom/surfaces/bvh/BVHsurf.h"

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

// This function sets the correct pointers on device memory in the data structures that were copied
template <typename Real_t>
static __global__ void BrepCudaManagerFinishTransfer(SurfData<Real_t> *surfData)
{
  int *current, *current_exiting_surface, *current_entering_surface, *current_entering_surface_pvol;
  int *currentShellDaughterPvolTrans, *currentShellDaughterLvolId;
  logic_int *current_logic;
  globaldevicesurfdata::gSurfDataDevice<Real_t> = surfData;

  // Write pointers into fShells[i].fSurfaces, fShells[i].fLogic, fShells[i].fShellEnteringSurfaceList,
  // fShells[i].fShellEnteringSurfaceTransList, fShells[i].fShellEnteringSurfacePvolList
  current                       = surfData->fSurfShellList;
  current_logic                 = surfData->fLogicList;
  current_exiting_surface       = surfData->fShellExitingSurfaceList;
  current_entering_surface      = surfData->fShellEnteringSurfaceList;
  current_entering_surface_pvol = surfData->fShellEnteringSurfacePvolList;
  currentShellDaughterPvolTrans = surfData->fShellDaughterPvolTransList;
  currentShellDaughterLvolId    = surfData->fShellDaughterLvolIdList;
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
    surfData->fShells[i].fDaughterPvolTrans = currentShellDaughterPvolTrans;
    currentShellDaughterPvolTrans += surfData->fShells[i].fNEnteringSurfaces;
    surfData->fShells[i].fDaughterLvolIds = currentShellDaughterLvolId;
    currentShellDaughterLvolId += surfData->fShells[i].fNEnteringSurfaces;
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

#define BREP_CUDA_CHECK(cmd)                                                       \
  do {                                                                             \
    cudaError_t err = cmd;                                                         \
    if (err != cudaSuccess) {                                                      \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                     \
    }                                                                              \
  } while (0)

// Manager class for synchronizing the surface data to the GPU.
template <typename Real_t>
class BrepCudaManager {
  using SurfData_t = SurfData<Real_t>;

  SurfData_t fSurfDataStaging;     ///< Host memory to stage data for the GPU
  SurfData_t *fSurfData = nullptr; ///< Device pointer to the data structure

public:
  static BrepCudaManager &Instance()
  {
    static BrepCudaManager instance;
    return instance;
  }

  const SurfData_t *GetDevicePtr() const { return fSurfData; }

  void TransferSurfData(const SurfData_t &surfData)
  {
    using Real_b = typename SurfData_t::Real_b;
    size_t sizeInBytes;

    // Allocate and copy transformations
    fSurfDataStaging.fNlocalTrans = surfData.fNlocalTrans;
    sizeInBytes                   = sizeof(surfData.fLocalTrans[0]) * surfData.fNlocalTrans;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fLocalTrans, sizeInBytes));
    BREP_CUDA_CHECK(
        cudaMemcpy(fSurfDataStaging.fLocalTrans, surfData.fLocalTrans, sizeInBytes, cudaMemcpyHostToDevice));

    fSurfDataStaging.fNglobalTrans = surfData.fNglobalTrans;
    sizeInBytes                    = sizeof(surfData.fGlobalTrans[0]) * surfData.fNglobalTrans;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fGlobalTrans, sizeInBytes));
    BREP_CUDA_CHECK(
        cudaMemcpy(fSurfDataStaging.fGlobalTrans, surfData.fGlobalTrans, sizeInBytes, cudaMemcpyHostToDevice));

    fSurfDataStaging.fNvolTrans = surfData.fNvolTrans;
    sizeInBytes                 = sizeof(surfData.fPVolTrans[0]) * surfData.fNvolTrans;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fPVolTrans, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fPVolTrans, surfData.fPVolTrans, sizeInBytes, cudaMemcpyHostToDevice));

    // Allocate and copy surface data
    fSurfDataStaging.fNcylsph = surfData.fNcylsph;
    sizeInBytes               = sizeof(surfData.fCylSphData[0]) * surfData.fNcylsph;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fCylSphData, sizeInBytes));
    BREP_CUDA_CHECK(
        cudaMemcpy(fSurfDataStaging.fCylSphData, surfData.fCylSphData, sizeInBytes, cudaMemcpyHostToDevice));

    fSurfDataStaging.fNcone = surfData.fNcone;
    sizeInBytes             = sizeof(surfData.fConeData[0]) * surfData.fNcone;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fConeData, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fConeData, surfData.fConeData, sizeInBytes, cudaMemcpyHostToDevice));

    fSurfDataStaging.fNellip = surfData.fNellip;
    sizeInBytes              = sizeof(surfData.fEllipData[0]) * surfData.fNellip;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fEllipData, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fEllipData, surfData.fEllipData, sizeInBytes, cudaMemcpyHostToDevice));

    fSurfDataStaging.fNtorus = surfData.fNtorus;
    sizeInBytes              = sizeof(surfData.fTorusData[0]) * surfData.fNtorus;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fTorusData, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fTorusData, surfData.fTorusData, sizeInBytes, cudaMemcpyHostToDevice));

    fSurfDataStaging.fNarb4 = surfData.fNarb4;
    sizeInBytes             = sizeof(surfData.fArb4Data[0]) * surfData.fNarb4;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fArb4Data, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fArb4Data, surfData.fArb4Data, sizeInBytes, cudaMemcpyHostToDevice));

    // Allocate and copy volume shells
    fSurfDataStaging.fNshells = surfData.fNshells;
    sizeInBytes               = sizeof(surfData.fShells[0]) * surfData.fNshells;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fShells, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fShells, surfData.fShells, sizeInBytes, cudaMemcpyHostToDevice));

    // Nota bene: the fShells[i].fSurfaces are backed by the following array
    // and set via BrepCudaManagerFinishTransfer.
    sizeInBytes = sizeof(surfData.fSurfShellList[0]) * surfData.fNlocalSurf;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fSurfShellList, sizeInBytes));
    BREP_CUDA_CHECK(
        cudaMemcpy(fSurfDataStaging.fSurfShellList, surfData.fSurfShellList, sizeInBytes, cudaMemcpyHostToDevice));

    // Nota bene: fShells[i].fLogic are backed by the following array
    // and set via BrepCudaManagerFinishTransfer.
    sizeInBytes = sizeof(logic_int) * surfData.fNlogic;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fLogicList, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fLogicList, surfData.fLogicList, sizeInBytes, cudaMemcpyHostToDevice));

    // Nota bene: fShells[i].fShellExiting/EnteringSurfaceList are backed by the following array
    // and set via BrepCudaManagerFinishTransfer.
    fSurfDataStaging.fNExitingSurfaces  = surfData.fNExitingSurfaces;
    fSurfDataStaging.fNEnteringSurfaces = surfData.fNEnteringSurfaces;

    sizeInBytes = sizeof(surfData.fShellExitingSurfaceList[0]) * surfData.fNExitingSurfaces;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fShellExitingSurfaceList, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fShellExitingSurfaceList, surfData.fShellExitingSurfaceList,
                               sizeInBytes, cudaMemcpyHostToDevice));
    sizeInBytes = sizeof(surfData.fShellEnteringSurfaceList[0]) * surfData.fNEnteringSurfaces;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fShellEnteringSurfaceList, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fShellEnteringSurfaceList, surfData.fShellEnteringSurfaceList,
                               sizeInBytes, cudaMemcpyHostToDevice));

    // Nota bene: fShells[i].fShellEnteringSurfacePvolList are backed by the following array
    // and set via BrepCudaManagerFinishTransfer.
    sizeInBytes = sizeof(surfData.fShellEnteringSurfacePvolList[0]) * surfData.fNEnteringSurfaces;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fShellEnteringSurfacePvolList, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fShellEnteringSurfacePvolList, surfData.fShellEnteringSurfacePvolList,
                               sizeInBytes, cudaMemcpyHostToDevice));

    sizeInBytes = sizeof(surfData.fShellDaughterPvolTransList[0]) * surfData.fNEnteringSurfaces;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fShellDaughterPvolTransList, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fShellDaughterPvolTransList, surfData.fShellDaughterPvolTransList,
                               sizeInBytes, cudaMemcpyHostToDevice));

    sizeInBytes = sizeof(surfData.fShellDaughterLvolIdList[0]) * surfData.fNEnteringSurfaces;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fShellDaughterLvolIdList, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fShellDaughterLvolIdList, surfData.fShellDaughterLvolIdList,
                               sizeInBytes, cudaMemcpyHostToDevice));

    // Allocate space for the BVHs
    sizeInBytes = sizeof(surfData.fBVH[0]) * surfData.fNshells;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fBVH, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fBVH, surfData.fBVH, sizeInBytes, cudaMemcpyHostToDevice));

    // Allocate and copy BVH members
    for (int i = 0; i < surfData.fNshells; ++i) {
      if (surfData.fShells[i].fNsurf > 0) { // this checks that the BVH is actually populated and not null
        auto const &hBVH = surfData.fBVH[i];
        auto dBVH        = &(fSurfDataStaging.fBVH[i]);

        int *dPrimId;
        int *dOffset;
        int *dNChild;
        bvh::AABBsurf<Real_b> *dNodes;
        bvh::AABBsurf<Real_b> *dAABBs;

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

        BREP_CUDA_CHECK(cudaMalloc(&dPrimId, hBVH.GetRootNChild() * sizeof(int)));
        BREP_CUDA_CHECK(cudaMalloc(&dOffset, nodes * sizeof(int)));
        BREP_CUDA_CHECK(cudaMalloc(&dNChild, nodes * sizeof(int)));
        BREP_CUDA_CHECK(cudaMalloc(&dNodes, nodes * sizeof(bvh::AABBsurf<Real_b>)));
        BREP_CUDA_CHECK(cudaMalloc(&dAABBs, hBVH.GetRootNChild() * sizeof(bvh::AABBsurf<Real_b>)));

        // Ensure pointers are not null after allocation
        if (!dPrimId || !dAABBs || !dOffset || !dNChild || !dNodes) {
          throw std::runtime_error("Memory allocation failed: One or more pointers are null.");
        }

        BREP_CUDA_CHECK(
            cudaMemcpy(dPrimId, hBVH.GetPrimId(), hBVH.GetRootNChild() * sizeof(int), cudaMemcpyHostToDevice));
        BREP_CUDA_CHECK(cudaMemcpy(dOffset, hBVH.GetOffset(), nodes * sizeof(int), cudaMemcpyHostToDevice));
        BREP_CUDA_CHECK(cudaMemcpy(dNChild, hBVH.GetNChild(), nodes * sizeof(int), cudaMemcpyHostToDevice));
        BREP_CUDA_CHECK(
            cudaMemcpy(dNodes, hBVH.GetNodes(), nodes * sizeof(bvh::AABBsurf<Real_b>), cudaMemcpyHostToDevice));
        BREP_CUDA_CHECK(cudaMemcpy(dAABBs, hBVH.GetAABBs(), hBVH.GetRootNChild() * sizeof(bvh::AABBsurf<Real_b>),
                                   cudaMemcpyHostToDevice));

        // Adjust pointers in the GPU instance
        FinishBVHCopy<<<1, 1>>>(dBVH, dPrimId, dOffset, dNChild, dAABBs, dNodes);
      }
    }

    // Allocate and copy surfaces
    fSurfDataStaging.fNlocalSurf = surfData.fNlocalSurf;
    sizeInBytes                  = sizeof(surfData.fLocalSurf[0]) * surfData.fNlocalSurf;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fLocalSurf, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fLocalSurf, surfData.fLocalSurf, sizeInBytes, cudaMemcpyHostToDevice));

    fSurfDataStaging.fNglobalSurf = surfData.fNglobalSurf;
    sizeInBytes                   = sizeof(surfData.fFramedSurf[0]) * surfData.fNglobalSurf;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fFramedSurf, sizeInBytes));
    BREP_CUDA_CHECK(
        cudaMemcpy(fSurfDataStaging.fFramedSurf, surfData.fFramedSurf, sizeInBytes, cudaMemcpyHostToDevice));

    // Allocate and copy masks
    fSurfDataStaging.fNwindows = surfData.fNwindows;
    sizeInBytes                = sizeof(surfData.fWindowMasks[0]) * surfData.fNwindows;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fWindowMasks, sizeInBytes));
    BREP_CUDA_CHECK(
        cudaMemcpy(fSurfDataStaging.fWindowMasks, surfData.fWindowMasks, sizeInBytes, cudaMemcpyHostToDevice));

    fSurfDataStaging.fNrings = surfData.fNrings;
    sizeInBytes              = sizeof(surfData.fRingMasks[0]) * surfData.fNrings;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fRingMasks, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fRingMasks, surfData.fRingMasks, sizeInBytes, cudaMemcpyHostToDevice));

    fSurfDataStaging.fNzphis = surfData.fNzphis;
    sizeInBytes              = sizeof(surfData.fZPhiMasks[0]) * surfData.fNzphis;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fZPhiMasks, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fZPhiMasks, surfData.fZPhiMasks, sizeInBytes, cudaMemcpyHostToDevice));

    fSurfDataStaging.fNquads = surfData.fNquads;
    sizeInBytes              = sizeof(surfData.fQuadMasks[0]) * surfData.fNquads;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fQuadMasks, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fQuadMasks, surfData.fQuadMasks, sizeInBytes, cudaMemcpyHostToDevice));

    fSurfDataStaging.fNtriangs = surfData.fNtriangs;
    sizeInBytes                = sizeof(surfData.fTriangleMasks[0]) * surfData.fNtriangs;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fTriangleMasks, sizeInBytes));
    BREP_CUDA_CHECK(
        cudaMemcpy(fSurfDataStaging.fTriangleMasks, surfData.fTriangleMasks, sizeInBytes, cudaMemcpyHostToDevice));

    // Allocate and copy scene indices
    fSurfDataStaging.fNscenes = surfData.fNscenes;
    sizeInBytes               = sizeof(int) * surfData.fNscenes;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fSceneStartIndex, sizeInBytes));
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fSceneTouchables, sizeInBytes));
    BREP_CUDA_CHECK(
        cudaMemcpy(fSurfDataStaging.fSceneStartIndex, surfData.fSceneStartIndex, sizeInBytes, cudaMemcpyHostToDevice));
    BREP_CUDA_CHECK(
        cudaMemcpy(fSurfDataStaging.fSceneTouchables, surfData.fSceneTouchables, sizeInBytes, cudaMemcpyHostToDevice));

    // Allocate and copy common surfaces
    fSurfDataStaging.fNcommonSurf = surfData.fNcommonSurf;
    sizeInBytes                   = sizeof(surfData.fCommonSurfaces[0]) * surfData.fNcommonSurf;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fCommonSurfaces, sizeInBytes));
    BREP_CUDA_CHECK(
        cudaMemcpy(fSurfDataStaging.fCommonSurfaces, surfData.fCommonSurfaces, sizeInBytes, cudaMemcpyHostToDevice));

    // Nota bene: the fCommonSurfaces[i].f{Left,Right}Side.fSurfaces are backed
    // by the following array and set via BrepCudaManagerFinishTransfer.
    fSurfDataStaging.fNsides = surfData.fNsides;
    sizeInBytes              = sizeof(surfData.fSides[0]) * surfData.fNsides;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fSides, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fSides, surfData.fSides, sizeInBytes, cudaMemcpyHostToDevice));

    // Allocate and copy side divisions
    // Nota bene: the fSideDivisions[i].fSlices and fSlices[i].fCandidates are backed
    // by the following arrays and set via BrepCudaManagerFinishTransfer.
    fSurfDataStaging.fNsideDivisions   = surfData.fNsideDivisions;
    fSurfDataStaging.fNslices          = surfData.fNslices;
    fSurfDataStaging.fNsliceCandidates = surfData.fNsliceCandidates;
    sizeInBytes                        = sizeof(surfData.fSideDivisions[0]) * surfData.fNsideDivisions;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fSideDivisions, sizeInBytes));
    BREP_CUDA_CHECK(
        cudaMemcpy(fSurfDataStaging.fSideDivisions, surfData.fSideDivisions, sizeInBytes, cudaMemcpyHostToDevice));
    sizeInBytes = sizeof(surfData.fSlices[0]) * surfData.fNslices;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fSlices, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fSlices, surfData.fSlices, sizeInBytes, cudaMemcpyHostToDevice));
    sizeInBytes = sizeof(int) * surfData.fNsliceCandidates;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fSliceCandidates, sizeInBytes));
    BREP_CUDA_CHECK(
        cudaMemcpy(fSurfDataStaging.fSliceCandidates, surfData.fSliceCandidates, sizeInBytes, cudaMemcpyHostToDevice));

    // Allocate and copy candidates lists
    fSurfDataStaging.fNStates = surfData.fNStates;
    sizeInBytes               = sizeof(surfData.fCandidates[0]) * surfData.fNStates;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fCandidates, sizeInBytes));
    BREP_CUDA_CHECK(
        cudaMemcpy(fSurfDataStaging.fCandidates, surfData.fCandidates, sizeInBytes, cudaMemcpyHostToDevice));

    // Nota bene: the fCandidates[i].{fCandidates,fFrameInd,fSides} are backed by the
    // following array and set via BrepCudaManagerFinishTransfer.
    fSurfDataStaging.fSizeCandList = surfData.fSizeCandList;
    sizeInBytes                    = sizeof(surfData.fCandList[0]) * surfData.fSizeCandList;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fCandList, sizeInBytes));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfDataStaging.fCandList, surfData.fCandList, sizeInBytes, cudaMemcpyHostToDevice));

    // Now copy the staged data to the GPU
    BREP_CUDA_CHECK(cudaMalloc(&fSurfData, sizeof(SurfData_t)));
    BREP_CUDA_CHECK(cudaMemcpy(fSurfData, &fSurfDataStaging, sizeof(SurfData_t), cudaMemcpyHostToDevice));

    // Finally finish the transfer by calling a kernel to write some pointers
    BrepCudaManagerFinishTransfer<<<1, 1>>>(fSurfData);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());
  }

  void Cleanup()
  {
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fLocalTrans));
    fSurfDataStaging.fLocalTrans = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fGlobalTrans));
    fSurfDataStaging.fGlobalTrans = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fPVolTrans));
    fSurfDataStaging.fPVolTrans = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fCylSphData));
    fSurfDataStaging.fCylSphData = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fConeData));
    fSurfDataStaging.fConeData = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fEllipData));
    fSurfDataStaging.fEllipData = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fTorusData));
    fSurfDataStaging.fTorusData = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fArb4Data));
    fSurfDataStaging.fArb4Data = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fShells));
    fSurfDataStaging.fShells = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fSurfShellList));
    fSurfDataStaging.fSurfShellList = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fLogicList));
    fSurfDataStaging.fLogicList = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fShellExitingSurfaceList));
    fSurfDataStaging.fShellExitingSurfaceList = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fShellEnteringSurfaceList));
    fSurfDataStaging.fShellEnteringSurfaceList = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fShellEnteringSurfacePvolList));
    fSurfDataStaging.fShellEnteringSurfacePvolList = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fShellDaughterPvolTransList));
    fSurfDataStaging.fShellDaughterPvolTransList = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fLocalSurf));
    fSurfDataStaging.fLocalSurf = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fFramedSurf));
    fSurfDataStaging.fFramedSurf = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fWindowMasks));
    fSurfDataStaging.fWindowMasks = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fRingMasks));
    fSurfDataStaging.fRingMasks = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fZPhiMasks));
    fSurfDataStaging.fZPhiMasks = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fQuadMasks));
    fSurfDataStaging.fQuadMasks = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fCommonSurfaces));
    fSurfDataStaging.fCommonSurfaces = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fSides));
    fSurfDataStaging.fSides = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fCandidates));
    fSurfDataStaging.fCandidates = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fCandList));
    fSurfDataStaging.fCandList = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfData));
    fSurfData = nullptr;
  }
};

} // namespace vgbrep

#endif

#endif
