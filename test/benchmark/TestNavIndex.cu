/// \file TestNavIndex.cu
/// \author Andrei Gheata (andrei.gheata@cern.ch)

#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/management/ReferenceNavState.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/base/Stopwatch.h>

#include <iomanip>
#include "VecGeom/base/Assert.h"
#include <cstdio>

using namespace vecgeom;

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
  if (result) {
    fprintf(stderr, "CUDA error = %s at %s:%d\n", cudaGetErrorString(result), file, line);
    cudaDeviceReset();
    exit(1);
  }
}

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

namespace visitorcuda {

VECCORE_ATT_DEVICE
int ReportIncompatibleDaughter(VPlacedVolume const *parent, VPlacedVolume const *daughter)
{
  auto const parent_id = parent ? static_cast<long long>(parent->id()) : -1LL;
  auto const child_id  = daughter ? static_cast<long long>(daughter->id()) : -1LL;
  printf("=== EEE === TestNavIndex: error code %d\n",
         static_cast<int>(ReferenceNavValidationError::kIncompatibleDaughter));
  printf("    expected daughter child id >= 0 for descent from %lld to %lld, got %d\n", parent_id, child_id,
         daughter ? daughter->GetChildId() : -1);
  return static_cast<int>(ReferenceNavValidationError::kIncompatibleDaughter);
}

template <typename EncodedNavState, typename EncodedState>
VECCORE_ATT_DEVICE int ReportValidationError(ReferenceNavValidationError error, ReferenceNavState const &reference,
                                             EncodedState encoded_state)
{
  printf("=== EEE === TestNavIndex: error code %d\n", static_cast<int>(error));
  PrintValidationFailure<EncodedNavState>(error, reference, encoded_state);
  return static_cast<int>(error);
}

template <typename EncodedNavState, typename EncodedState>
VECCORE_ATT_DEVICE int ReportSceneTransitionError(EncodedState parent_state, EncodedState child_state,
                                                  VPlacedVolume const *parent, VPlacedVolume const *daughter)
{
  printf("=== EEE === TestNavIndex: error code %d\n",
         static_cast<int>(ReferenceNavValidationError::kIncompatibleScene));
  PrintSceneTransitionFailure<EncodedNavState>(parent_state, child_state, parent, daughter);
  return static_cast<int>(ReferenceNavValidationError::kIncompatibleScene);
}

VECCORE_ATT_DEVICE
int visitAllPlacedVolumesPassNavIndex(VPlacedVolume const *currentvolume, ReferenceNavState &reference,
                                      NavIndex_t nav_ind, int &niter)
{
  auto validation_error = ValidateEncodedState<NavStateIndex>(reference, nav_ind);
  if (validation_error != ReferenceNavValidationError::kNone) {
    return ReportValidationError<NavStateIndex>(validation_error, reference, nav_ind);
  }

  ++niter;
  for (auto daughter : currentvolume->GetDaughters()) {
    if (daughter->GetChildId() < 0) {
      return ReportIncompatibleDaughter(currentvolume, daughter);
    }
    auto child_nav_ind = nav_ind;
    NavStateIndex::PushImpl(child_nav_ind, daughter);
    reference.Push(daughter);
    auto ierr = visitAllPlacedVolumesPassNavIndex(daughter, reference, child_nav_ind, niter);
    reference.Pop();
    if (ierr > 0) return ierr;
  }
  return 0;
}

VECCORE_ATT_DEVICE
int visitAllPlacedVolumesPassNavTuple(VPlacedVolume const *currentvolume, ReferenceNavState &reference,
                                      NavTuple_t nav_tuple, int &niter)
{
  auto validation_error = ValidateEncodedState<NavStateTuple>(reference, nav_tuple);
  if (validation_error != ReferenceNavValidationError::kNone) {
    return ReportValidationError<NavStateTuple>(validation_error, reference, nav_tuple);
  }

  ++niter;
  for (auto daughter : currentvolume->GetDaughters()) {
    if (daughter->GetChildId() < 0) {
      return ReportIncompatibleDaughter(currentvolume, daughter);
    }
    auto child_nav_tuple = nav_tuple;
    NavStateTuple::PushImpl(child_nav_tuple, daughter);
    auto scene_error = ValidateSceneTransition<NavStateTuple>(nav_tuple, child_nav_tuple);
    if (scene_error != ReferenceNavValidationError::kNone)
      return ReportSceneTransitionError<NavStateTuple>(nav_tuple, child_nav_tuple, currentvolume, daughter);
    reference.Push(daughter);
    auto ierr = visitAllPlacedVolumesPassNavTuple(daughter, reference, child_nav_tuple, niter);
    reference.Pop();
    if (ierr) return ierr;
  }
  return 0;
}

} // namespace visitorcuda

__global__ void TestNavIndexGPUKernel(vecgeom::cuda::VPlacedVolume const *const gpu_world, int *ierr)
{
  using namespace visitorcuda;
  auto reference     = ReferenceNavState::MakeWorld(gpu_world);
  NavIndex_t nav_ind = 1; // The navigation index corresponding to the world
  int niter          = 0;

#ifdef VECGEOM_USE_NAVTUPLE
  *ierr = visitAllPlacedVolumesPassNavTuple(gpu_world, reference, NavTuple_t{nav_ind}, niter);
#else
  *ierr = visitAllPlacedVolumesPassNavIndex(gpu_world, reference, nav_ind, niter);
#endif
}

int TestNavIndexGPU(vecgeom::cxx::VPlacedVolume const *const world, int maxdepth)
{
  // Load and synchronize the geometry on the GPU
  (void)maxdepth;

  vecgeom::cxx::CudaManager::Instance().LoadGeometry(world);
  vecgeom::cxx::CudaManager::Instance().Synchronize();

  auto gpu_world = vecgeom::cxx::CudaManager::Instance().world_gpu();
  VECGEOM_VALIDATE(gpu_world, << "GPU world volume is a null pointer");

  int ierr;
  int *d_ierr;
  cudaMalloc(&d_ierr, sizeof(int));

  Stopwatch timer;
  timer.Start();
  TestNavIndexGPUKernel<<<1, 1>>>(gpu_world, d_ierr);
  cudaMemcpy(&ierr, d_ierr, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_ierr);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  auto tvalidate = timer.Stop();
  if (!ierr) std::cout << "=== Info navigation table validation on GPU took: " << tvalidate << " sec.\n";

  return ierr;
}
