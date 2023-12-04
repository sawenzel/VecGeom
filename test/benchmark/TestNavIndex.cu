/// \file TestNavIndex.cu
/// \author Andrei Gheata (andrei.gheata@cern.ch)

#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/base/Stopwatch.h>

#include <iomanip>
#include <cassert>
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

class GlobalToLocalVisitor {
private:
  int fError = 0; ///< error code
  int fNiter = 0; ///< number of iterations
public:
  VECCORE_ATT_HOST_DEVICE
  GlobalToLocalVisitor() {}

  VECCORE_ATT_HOST_DEVICE
  int GetError() const { return fError; }

  VECCORE_ATT_HOST_DEVICE
  int GetNiter() const { return fNiter; }

  VECCORE_ATT_HOST_DEVICE
  void apply(NavStatePath *state, NavIndex_t nav_index)
  {
    unsigned char level            = state->GetLevel();
    int dind                       = 0;
    NavIndex_t nav_ind             = 1;
    VPlacedVolume const *pdaughter = nullptr;
    for (int i = 1; i < level + 1; ++i) {
      pdaughter = state->At(i);
      dind      = pdaughter->GetChildId();
      if (dind < 0) {
        fError = 1;
        return;
      }
      NavStateIndex::PushImpl(nav_ind, pdaughter);
    }

    // Check if navigation index matches input
    if (nav_ind != nav_index) {
      fError = 2;
      return;
    }

    // Check if the physical volume is correct
    if (NavStateIndex::TopImpl(nav_ind) != state->Top()) {
      fError = 3;
      return;
    }

    // Check if the current level is valid
    if (level != NavStateIndex::GetLevelImpl(nav_ind)) {
      fError = 4;
      return;
    }

    if (level > 0) {
      auto nav_ind_m = nav_ind;
      NavStateIndex::PopImpl(nav_ind_m);
      NavStateIndex::PushImpl(nav_ind_m, pdaughter);
      if (nav_ind_m != nav_ind) {
        fError = 5;
        return;
      }
    }

    // Check if the number of daughters is correct
    if (NavStateIndex::GetNdaughtersImpl(nav_ind) != state->Top()->GetDaughters().size()) {
      fError = 6;
      return;
    }

    Transformation3D trans, trans_nav_ind;
    state->TopMatrix(trans);
    NavStateIndex::TopMatrixImpl(nav_ind, trans_nav_ind);
    // If transformations are not cached to the full depth, strict equality does not stand:
    // rounded(t1 * t2) * t3 * t4 != t1 * t2 * t3 * t4
    if (!trans.ApproxEqual(trans_nav_ind)) {
      fError = 7;
      return;
    }

    // success
    fError = 0;
    fNiter++;
  }

  VECCORE_ATT_HOST_DEVICE
  void apply_tuple(NavStatePath *state, NavTuple_t &nav_tuple)
  {
    unsigned char level            = state->GetLevel();
    int dind                       = 0;
    NavTuple_t nav_tpl             = 1;
    VPlacedVolume const *pdaughter = nullptr;
    for (int i = 1; i < level + 1; ++i) {
      pdaughter = state->At(i);
      dind      = pdaughter->GetChildId();
      if (dind < 0) {
        fError = 1;
        return;
      }
      NavStateTuple::PushImpl(nav_tpl, pdaughter);
    }

    // Check if navigation index matches input
    if (nav_tpl != nav_tuple) {
      fError = 2;
      return;
    }

    // Check if the physical volume is correct
    if (NavStateTuple::TopImpl(nav_tuple) != state->Top()) {
      fError = 3;
      NavStateTuple::TopImpl(nav_tuple);
      return;
    }

    // Check if the current level is valid
    if (level != NavStateTuple::GetLevelImpl(nav_tuple)) {
      NavStateTuple::GetLevelImpl(nav_tuple);
      fError = 4;
      return;
    }

    // Check if mother navigation index is consistent
    if (level > 0) {
      auto nav_tuple_m = nav_tuple;
      NavStateTuple::PopImpl(nav_tuple_m);
      NavStateTuple::PushImpl(nav_tuple_m, pdaughter);
      if (nav_tuple_m != nav_tuple) {
        fError      = 5;
        nav_tuple_m = nav_tuple;
        NavStateTuple::PopImpl(nav_tuple_m);
        NavStateTuple::PushImpl(nav_tuple_m, pdaughter);
        return;
      }
    }

    // Check if the number of daughters is correct
    if (NavStateTuple::GetNdaughtersImpl(nav_tuple) != state->Top()->GetDaughters().size()) {
      fError = 6;
      return;
    }

    // Check the top transformation
    Transformation3D trans, trans_nav_tuple;
    state->TopMatrix(trans);
    NavStateTuple::TopMatrixImpl(nav_tuple, trans_nav_tuple);
    // If transformations are not cached to the full depth, strict equality does not stand:
    // rounded(t1 * t2) * t3 * t4 != t1 * t2 * t3 * t4
    if (!trans.ApproxEqual(trans_nav_tuple)) {
      Transformation3D trans_test, trans_nav_tuple_test;
      state->TopMatrix(trans_test);
      NavStateTuple::TopMatrixImpl(nav_tuple, trans_nav_tuple_test);
      fError = 7;
      return;
    }

    // Check the volume id
    if (level > 0) {
      auto ivol = pdaughter->GetLogicalVolume()->id();
      if (ivol != NavStateTuple::GetLogicalIdImpl(nav_tuple)) {
        NavStateTuple::GetLevelImpl(nav_tuple);
        fError = 8;
        return;
      }
    }

    // success
    fError = 0;
  }
};

/// Traverses the geometry tree keeping track of the state context (volume path or navigation state)
/// and applies the injected Visitor
template <typename Visitor>
VECCORE_ATT_DEVICE
int visitAllPlacedVolumesPassNavIndex(VPlacedVolume const *currentvolume, Visitor *visitor, NavStatePath *state,
                                      NavIndex_t nav_ind)
{
  const char *errcodes[] = {"incompatible daughter pointer",
                            "navigation index mismatch",
                            "top placed volume pointer mismatch",
                            "level mismatch",
                            "navigation index inconsistency for Push/Pop",
                            "number of daughters mismatch",
                            "transformation matrix mismatch"};
  constexpr int maxiter  = 100000; // limit the maximum number of iterations (slow on 1 GPU thread)
  if (currentvolume != NULL) {
    state->Push(currentvolume);
    visitor->apply(state, nav_ind);
    auto ierr = visitor->GetError();
    if (ierr) {
      printf("=== EEE === TestNavIndex: %s\n", errcodes[ierr - 1]);
      return ierr;
    }
    if (visitor->GetNiter() > maxiter) return 0;
    for (auto daughter : currentvolume->GetDaughters()) {
      auto nav_ind_d = nav_ind;
      NavStateIndex::PushImpl(nav_ind_d, daughter);
      ierr           = visitAllPlacedVolumesPassNavIndex(daughter, visitor, state, nav_ind_d);
      if (ierr > 0) return ierr;
      if (visitor->GetNiter() > maxiter) return 0;
    }
    state->Pop();
  }
  return 0;
}

/// Traverses the geometry tree keeping track of the state context (volume path or navigation state)
/// and applies the injected Visitor
template <typename Visitor>
VECCORE_ATT_DEVICE
int visitAllPlacedVolumesPassNavTuple(VPlacedVolume const *currentvolume, Visitor *visitor, NavStatePath *state,
                                      NavTuple_t nav_tuple)
{
  const char *errcodes[] = {"incompatible daughter pointer",
                            "navigation index mismatch",
                            "top placed volume pointer mismatch",
                            "level mismatch",
                            "navigation index inconsistency for Push/Pop",
                            "number of daughters mismatch",
                            "transformation matrix mismatch"};
  constexpr int maxiter  = 100000; // limit the maximum number of iterations (slow on 1 GPU thread)
  if (currentvolume != NULL) {
    state->Push(currentvolume);
    visitor->apply_tuple(state, nav_tuple);
    auto ierr = visitor->GetError();
    if (ierr) {
      printf("=== EEE === TestNavIndex: %s\n", errcodes[ierr - 1]);
      return ierr;
    }
    for (auto daughter : currentvolume->GetDaughters()) {
      NavStateTuple::PushImpl(nav_tuple, daughter);
      ierr = visitAllPlacedVolumesPassNavTuple(daughter, visitor, state, nav_tuple);
      if (ierr) return ierr;
      NavStateTuple::PopImpl(nav_tuple);
      if (visitor->GetNiter() > maxiter) return 0;
    }
    state->Pop();
  }
  return 0;
}

} // namespace visitorcuda

__global__ void TestNavIndexGPUKernel(vecgeom::cuda::VPlacedVolume const *const gpu_world,
                                      vecgeom::cuda::NavStatePath *const state, int *ierr)
{
  using namespace visitorcuda;

  state->Clear();
  GlobalToLocalVisitor visitor;

  NavIndex_t nav_ind_top = 1; // The navigation index corresponding to the world

#ifdef VECGEOM_USE_NAVTUPLE
  *ierr = visitAllPlacedVolumesPassNavTuple(gpu_world, &visitor, state, NavTuple_t{nav_ind_top});
#else
  *ierr = visitAllPlacedVolumesPassNavIndex(gpu_world, &visitor, state, nav_ind_top);
#endif
}

int TestNavIndexGPU(vecgeom::cxx::VPlacedVolume const *const world, int maxdepth)
{
  // Load and synchronize the geometry on the GPU
  size_t statesize = NavigationState::SizeOfInstance(maxdepth);

  vecgeom::cxx::CudaManager::Instance().LoadGeometry(world);
  vecgeom::cxx::CudaManager::Instance().Synchronize();

  auto gpu_world = vecgeom::cxx::CudaManager::Instance().world_gpu();
  assert(gpu_world && "GPU world volume is a null pointer");

  char *input_buffer = nullptr;
  checkCudaErrors(cudaMallocManaged((void **)&input_buffer, statesize));
  auto state = NavStatePath::MakeInstanceAt(maxdepth, (void *)(input_buffer));

  int ierr;
  int *d_ierr;
  cudaMalloc(&d_ierr, sizeof(int));

  Stopwatch timer;
  timer.Start();
  TestNavIndexGPUKernel<<<1, 1>>>(gpu_world, state, d_ierr);
  cudaMemcpy(&ierr, d_ierr, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_ierr);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(input_buffer));
  auto tvalidate = timer.Stop();
  if (!ierr) std::cout << "=== Info navigation table validation on GPU took: " << tvalidate << " sec.\n";

  return ierr;
}
