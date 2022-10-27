#ifndef VECGEOM_SURFACE_BREPCUDAMANAGER_H_
#define VECGEOM_SURFACE_BREPCUDAMANAGER_H_

// This header file can only be used from CUDA sources because it needs to call
// functions from the CUDA Runtime API and invoke a kernel.
#ifdef __CUDACC__

#include <VecGeom/surfaces/Model.h>

namespace vgbrep {

template <typename Real_t>
static __global__ void BrepCudaManagerFinishTransfer(SurfData<Real_t> *surfData)
{
  int *current;

  // Write pointers into fShells[i].fSurfaces
  current = surfData->fSurfShellList;
  for (int i = 0; i < surfData->fNshells; i++) {
    surfData->fShells[i].fSurfaces = current;
    current += surfData->fShells[i].fNsurf;
  }

  // Write pointers into fCommonSurfaces[i].f{Left,Right}Side.fSurfaces
  current = surfData->fSides;
  for (int i = 0; i < surfData->fNcommonSurf; i++) {
    surfData->fCommonSurfaces[i].fLeftSide.fSurfaces = current;
    current += surfData->fCommonSurfaces[i].fLeftSide.fNsurf;
    surfData->fCommonSurfaces[i].fRightSide.fSurfaces = current;
    current += surfData->fCommonSurfaces[i].fRightSide.fNsurf;
  }

  // Write pointers into fCandidates[i].{fCandidates,fFrameInd}
  current = surfData->fCandList;
  for (int i = 0; i < surfData->fNcandidates; i++) {
    surfData->fCandidates[i].fCandidates = current;
    current += surfData->fCandidates[i].fNcand;
    surfData->fCandidates[i].fFrameInd = current;
    current += surfData->fCandidates[i].fNcand;
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

    // Allocate and copy candidates lists
    fSurfDataStaging.fNcandidates = surfData.fNcandidates;
    sizeInBytes                   = sizeof(surfData.fCandidates[0]) * surfData.fNcandidates;
    BREP_CUDA_CHECK(cudaMalloc(&fSurfDataStaging.fCandidates, sizeInBytes));
    BREP_CUDA_CHECK(
        cudaMemcpy(fSurfDataStaging.fCandidates, surfData.fCandidates, sizeInBytes, cudaMemcpyHostToDevice));

    // Nota bene: the fCandidates[i].{fCandidates,fFrameInd} are backed by the
    // following array and set via BrepCudaManagerFinishTransfer.
    fSurfDataStaging.fNcandList = surfData.fNcandList;
    sizeInBytes                 = sizeof(surfData.fCandList[0]) * surfData.fNcandList;
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
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fCylSphData));
    fSurfDataStaging.fCylSphData = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fConeData));
    fSurfDataStaging.fConeData = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fShells));
    fSurfDataStaging.fShells = nullptr;
    BREP_CUDA_CHECK(cudaFree(fSurfDataStaging.fSurfShellList));
    fSurfDataStaging.fSurfShellList = nullptr;
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
