#include <VecGeom/surfaces/BrepCudaManager.h>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/Navigator.h>
#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/SimpleSafetyEstimator.h>
#include <VecGeom/navigation/BVHSafetyEstimator.h>
#include <VecGeom/base/Stopwatch.h>

using namespace vecgeom;
using BrepCudaManager = vgbrep::BrepCudaManager<vecgeom::Precision>;
using SurfData        = vgbrep::SurfData<vecgeom::Precision>;
using vecCore::math::Abs;
using Vec3D  = vecgeom::Vector3D<vecgeom::Precision>;
using Vec3Dc = Precision[3];

//==================================================================================
__global__ void LocateSolids(int nrays, Vector3D<Precision> const *points, NavStateIndex *in_states,
                             const VPlacedVolume *world)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    Vector3D<Precision> const &pos = points[i];
    // Locate with solid-based model
    GlobalLocator::LocateGlobalPoint(world, pos, in_states[i], true);
  }
}
//==================================================================================
__global__ void LocateSolidsBVH(int nrays, Vector3D<Precision> const *points, NavStateIndex *in_states,
                                const VPlacedVolume *world)
{
  auto nav = static_cast<BVHNavigator<> *>(BVHNavigator<>::Instance());
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    Vector3D<Precision> const &pos = points[i];
    // Locate with solid-based model
    nav->LocateGlobalPoint(world, pos, in_states[i], true);
  }
}
//==================================================================================
__global__ void LocateSurf(int nrays, SurfData const *surfDataPtr, Vector3D<Precision> const *points,
                           NavStateIndex *out_states, const VPlacedVolume *world)
{
  SurfData const &surfdata = *surfDataPtr;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    // Locate with surface-based model
    vgbrep::protonav::LocatePointIn(world, points[i], out_states[i], surfdata, true);
  }
}
//==================================================================================
__global__ void ResetStates(int nrays, NavStateIndex *out_states)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x)
    out_states[i].Clear();
}
//==================================================================================
__global__ void ValidateLocate(int nrays, NavStateIndex const *in_states, NavStateIndex const *out_states,
                               int *num_errors)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    if (out_states[i].GetNavIndex() != in_states[i].GetNavIndex()) atomicAdd(num_errors, 1);
  }
}
//==================================================================================
__global__ void ComputeSafetiesSurf(int nrays, SurfData const *surfDataPtr, Vector3D<Precision> const *points,
                                    NavStateIndex *in_states, Precision *safeties)
{
  SurfData const &surfdata = *surfDataPtr;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    int exit_surf;
    safeties[i] = vgbrep::protonav::ComputeSafety(points[i], in_states[i], surfdata, exit_surf);
  }
}
//==================================================================================
__global__ void ComputeSafetiesSolid(int nrays, Vector3D<Precision> const *points, NavStateIndex const *in_states,
                                     Precision *ref_safeties)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    ref_safeties[i] = SimpleSafetyEstimator::Instance()->ComputeSafety(points[i], in_states[i]);
  }
}
//==================================================================================
__global__ void ComputeSafetiesSolidBVH(int nrays, Vector3D<Precision> const *points, NavStateIndex const *in_states,
                                        Precision *ref_safeties)
{
  auto safety_estimator = BVHSafetyEstimator::Instance();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    safety_estimator->ComputeSafety(points[i], in_states[i]);
  }
}
//==================================================================================
__global__ void ValidateSafety(int nrays, Precision const *safeties, Precision const *refSafeties,
                               int *num_better_safety, int *num_worse_safety)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    atomicAdd(num_better_safety, int(safeties[i] > refSafeties[i] + kTolerance));
    atomicAdd(num_worse_safety, int(safeties[i] < refSafeties[i] - kTolerance));
  }
}
//==================================================================================
template <typename Navigator>
static __global__ void PropagateRaysSolid(int nrays, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                                          NavStateIndex const *in_states, Precision *length_over_crossings)
{
  auto nav = static_cast<Navigator *>(Navigator::Instance());
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    NavStateIndex start_state = in_states[i];
    NavStateIndex out_state;
    int num_cross   = 0;
    double dist_tot = 0;
    auto const &dir = dirs[i];
    auto pt         = points[i] + kTolerance * dir; // push the start and subsequent crossing points
    do {
      double distance;
      nav->FindNextBoundaryAndStep(pt, dir, start_state, out_state, kInfLength, distance);
      distance += kTolerance; // compensate for the push
      dist_tot += (num_cross + 1) * distance;
      pt += distance * dir; // this is pushed with kTolerance beyond the boundary
      start_state = out_state;
      num_cross++;
    } while (!out_state.IsOutside());

    length_over_crossings[i] = num_cross ? dist_tot / (num_cross + 1) : 0;
  }
}
//==================================================================================
__global__ void PropagateRaysSurf(int nrays, SurfData const *surfDataPtr, Vector3D<Precision> const *points,
                                  Vector3D<Precision> const *dirs, NavStateIndex const *in_states,
                                  Precision *length_over_crossings)
{
  SurfData const &surfdata = *surfDataPtr;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    NavStateIndex start_state = in_states[i];
    NavStateIndex out_state;
    int num_cross   = 0;
    int exit_surf   = 0;
    double dist_tot = 0;
    auto pt         = points[i];
    auto const &dir = dirs[i];
    do {
      exit_surf     = 0; // need to reset because the same inner tube surface can be crossed twice in a row
      auto distance = vgbrep::protonav::ComputeStepAndHit(pt, dir, start_state, out_state, surfdata, exit_surf);
      dist_tot += (num_cross + 1) * distance;
      pt += distance * dir;
      start_state = out_state;
      num_cross++;
    } while (!out_state.IsOutside());

    length_over_crossings[i] = num_cross ? dist_tot / (num_cross + 1) : 0;
  }
}
//==================================================================================
__global__ void ValidateTraversal(int nrays, Precision const *length_over_crossings,
                                  Precision const *refLength_over_crossings, int *num_errors)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    bool error_dist = Abs(length_over_crossings[i] - refLength_over_crossings[i]) > kTolerance;
    atomicAdd(num_errors, int(error_dist));
  }
}
//==================================================================================
int testRaytracingCUDA(int nrays, Vec3Dc const *pointsc, Vec3Dc const *dirsc, const SurfData &surfdata, bool debug)
{
  BrepCudaManager::Instance().TransferSurfData(surfdata);
  const SurfData *surfDataDevice = BrepCudaManager::Instance().GetDevicePtr();

  auto pointsh = new Vec3D[nrays];
  auto dirsh   = new Vec3D[nrays];
  for (auto i = 0; i < nrays; ++i) {
    pointsh[i].Set(pointsc[i][0], pointsc[i][1], pointsc[i][2]);
    dirsh[i].Set(dirsc[i][0], dirsc[i][1], dirsc[i][2]);
  }

  // Allocate/copy data on device
  Vec3D *points;
  BREP_CUDA_CHECK(cudaMalloc(&points, nrays * sizeof(Vec3D)));
  BREP_CUDA_CHECK(cudaMemcpy(points, pointsh, nrays * sizeof(Vec3D), cudaMemcpyHostToDevice));
  Vec3D *dirs;
  BREP_CUDA_CHECK(cudaMalloc(&dirs, nrays * sizeof(Vec3D)));
  BREP_CUDA_CHECK(cudaMemcpy(dirs, dirsh, nrays * sizeof(Vec3D), cudaMemcpyHostToDevice));
  NavStateIndex *origStates;
  BREP_CUDA_CHECK(cudaMalloc(&origStates, nrays * sizeof(NavStateIndex)));
  NavStateIndex *outputStates;
  BREP_CUDA_CHECK(cudaMalloc(&outputStates, nrays * sizeof(NavStateIndex)));
  Precision *refSafeties;
  BREP_CUDA_CHECK(cudaMalloc(&refSafeties, nrays * sizeof(Precision)));
  Precision *safeties;
  BREP_CUDA_CHECK(cudaMalloc(&safeties, nrays * sizeof(Precision)));
  Precision *refLength_over_crossings;
  BREP_CUDA_CHECK(cudaMalloc(&refLength_over_crossings, nrays * sizeof(Precision)));
  Precision *length_over_crossings;
  BREP_CUDA_CHECK(cudaMalloc(&length_over_crossings, nrays * sizeof(Precision)));

  int num_errors        = 0;
  int num_better_safety = 0;
  int num_worse_safety  = 0;
  int num_errors_dist   = 0;
  int *num_errors_d, *num_better_safety_d, *num_worse_safety_d, *num_errors_dist_d;
  BREP_CUDA_CHECK(cudaMalloc(&num_errors_d, sizeof(int)));
  BREP_CUDA_CHECK(cudaMalloc(&num_better_safety_d, sizeof(int)));
  BREP_CUDA_CHECK(cudaMalloc(&num_worse_safety_d, sizeof(int)));
  BREP_CUDA_CHECK(cudaMalloc(&num_errors_dist_d, sizeof(int)));

  constexpr int initThreads = 32;
  int initBlocks            = (nrays + initThreads - 1) / initThreads;

  Stopwatch timer;
  // Locating the global points with solid model
  auto world_dev = vecgeom::cxx::CudaManager::Instance().world_gpu();
  timer.Start();
  LocateSolids<<<initBlocks, initThreads>>>(nrays, points, origStates, world_dev);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_locate_solids = timer.Stop();

  // Locating the global points with solid model + BVH
  timer.Start();
  LocateSolidsBVH<<<initBlocks, initThreads>>>(nrays, points, outputStates, world_dev);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_locate_solids_bvh = timer.Stop();

  ResetStates<<<initBlocks, initThreads>>>(nrays, outputStates);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());

  // Locating the global points with surface model
  timer.Start();
  LocateSurf<<<initBlocks, initThreads>>>(nrays, surfDataDevice, points, outputStates, world_dev);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_locate_surf = timer.Stop();

  // Corectness for locating points
  ValidateLocate<<<initBlocks, initThreads>>>(nrays, origStates, outputStates, num_errors_d);
  BREP_CUDA_CHECK(cudaMemcpy(&num_errors, num_errors_d, sizeof(int), cudaMemcpyDeviceToHost));
  BREP_CUDA_CHECK(cudaDeviceSynchronize());

  if (num_errors > 0) std::cout << "CUDA: Point locate errors: " << num_errors << "\n";
  if (!debug)
    std::cout << "CUDA: locate_solids: " << time_locate_solids << "  locate_solids_BVH: " << time_locate_solids_bvh
              << "  locate_surf: " << time_locate_surf << "\n";

  // Safety for solids model (reference)
  timer.Start();
  ComputeSafetiesSolid<<<initBlocks, initThreads>>>(nrays, points, origStates, refSafeties);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_safety_solids = timer.Stop();

  // Safety for solids model with BVH
  timer.Start();
  ComputeSafetiesSolidBVH<<<initBlocks, initThreads>>>(nrays, points, origStates, safeties);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_safety_solids_bvh = timer.Stop();

  // Safety for surface model
  timer.Start();
  ComputeSafetiesSurf<<<initBlocks, initThreads>>>(nrays, surfDataDevice, points, origStates, safeties);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_safety_surf = timer.Stop();

  ValidateSafety<<<initBlocks, initThreads>>>(nrays, safeties, refSafeties, num_better_safety_d, num_worse_safety_d);
  BREP_CUDA_CHECK(cudaMemcpy(&num_better_safety, num_better_safety_d, sizeof(int), cudaMemcpyDeviceToHost));
  BREP_CUDA_CHECK(cudaMemcpy(&num_worse_safety, num_worse_safety_d, sizeof(int), cudaMemcpyDeviceToHost));
  BREP_CUDA_CHECK(cudaDeviceSynchronize());

  if (!debug)
    std::cout << "CUDA: safety_solids: " << time_safety_solids << " safety_solids_BVH: " << time_safety_solids_bvh
              << "  safety_surf: " << time_safety_surf << "\n";
  if (num_better_safety > 0) std::cout << "CUDA:    number of better safety values: " << num_better_safety << "\n";
  if (num_worse_safety > 0) std::cout << "CUDA:    number of worse safety values: \n" << num_worse_safety << "\n";

  // Traversal for solids model (reference)
  timer.Start();
  PropagateRaysSolid<NewSimpleNavigator<>>
      <<<initBlocks, initThreads>>>(nrays, points, dirs, origStates, refLength_over_crossings);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_traverse_solids = timer.Stop();

  // Traversal for solids model with BVH
  timer.Start();
  PropagateRaysSolid<BVHNavigator<>>
      <<<initBlocks, initThreads>>>(nrays, points, dirs, origStates, length_over_crossings);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_traverse_solids_bvh = timer.Stop();

  // Traversal for the surface model
  timer.Start();
  PropagateRaysSurf<<<initBlocks, initThreads>>>(nrays, surfDataDevice, points, dirs, origStates,
                                                 length_over_crossings);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_traverse_surf = timer.Stop();

  ValidateTraversal<<<initBlocks, initThreads>>>(nrays, length_over_crossings, refLength_over_crossings,
                                                 num_errors_dist_d);
  BREP_CUDA_CHECK(cudaMemcpy(&num_errors_dist, num_errors_dist_d, sizeof(int), cudaMemcpyDeviceToHost));
  BREP_CUDA_CHECK(cudaDeviceSynchronize());

  if (num_errors_dist > 0) std::cout << "CUDA: traversal errors: " << num_errors_dist << "\n";
  if (!debug)
    std::cout << "CUDA: traverse_solids: " << time_traverse_solids
              << " traverse_solids_bvh: " << time_traverse_solids_bvh << "  traverse_surf: " << time_traverse_surf
              << "\n";

  num_errors += num_errors_dist;
  if (num_errors > 0) printf("CUDA: num_erros = %d / %d\n", num_errors, nrays);

  BrepCudaManager::Instance().Cleanup();
  delete[] pointsh;
  delete[] dirsh;
  BREP_CUDA_CHECK(cudaFree(num_errors_d));
  BREP_CUDA_CHECK(cudaFree(num_better_safety_d));
  BREP_CUDA_CHECK(cudaFree(num_worse_safety_d));
  BREP_CUDA_CHECK(cudaFree(num_errors_dist_d));
  BREP_CUDA_CHECK(cudaFree(points));
  BREP_CUDA_CHECK(cudaFree(dirs));
  BREP_CUDA_CHECK(cudaFree(origStates));
  BREP_CUDA_CHECK(cudaFree(outputStates));
  BREP_CUDA_CHECK(cudaFree(refSafeties));
  BREP_CUDA_CHECK(cudaFree(safeties));
  BREP_CUDA_CHECK(cudaFree(refLength_over_crossings));
  BREP_CUDA_CHECK(cudaFree(length_over_crossings));
  return num_errors;
}
