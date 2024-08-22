#include "testRaytracing.h"

#include <VecGeom/surfaces/cuda/BrepCudaManager.h>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/Navigator.h>
#include <VecGeom/surfaces/BVHSurfNavigator.h>
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/LoopNavigator.h>
#include <VecGeom/base/Stopwatch.h>

using namespace vecgeom;
// surface model is based on Real_t precision defined in testRaytracing.h
using BrepCudaManager = vgbrep::BrepCudaManager<Real_t>;
using SurfData        = vgbrep::SurfData<Real_t>;
using vecCore::math::Abs;
// generation of rays etc is based on vecgeom::Precision as these are used also in the solid model
using Vec3D  = vecgeom::Vector3D<vecgeom::Precision>;
using Vec3Dc = Precision[3];

//==================================================================================
__global__ void LocateSolids(int nrays, Vector3D<Precision> const *points, NavigationState *in_states,
                             const VPlacedVolume *world)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    Vector3D<Precision> const &pos = points[i];
    // Locate with solid-based model
    LoopNavigator::LocatePointIn(world, pos, in_states[i], true);
  }
}
//==================================================================================
__global__ void LocateSolidsBVH(int nrays, Vector3D<Precision> const *points, NavigationState *in_states,
                                const VPlacedVolume *world)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    Vector3D<Precision> const &pos = points[i];
    // Locate with solid-based model
    BVHNavigator::LocatePointIn(world, pos, in_states[i], true);
  }
}
//==================================================================================
__global__ void LocateSurf(int nrays, Vector3D<Precision> const *points, NavigationState *out_states,
                           const VPlacedVolume *world)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    // Locate with surface-based model
    vgbrep::protonav::LocatePointIn<Precision, Real_t>(world, points[i], out_states[i], true);
  }
}
//==================================================================================
__global__ void LocateSurfBVH(int nrays, Vector3D<Precision> const *points, NavigationState *out_states,
                              const VPlacedVolume *world)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    // Locate with surface-based model + BVH
    vgbrep::protonav::BVHSurfNavigator<Real_t>::LocatePointIn(world->id(), points[i], out_states[i], true);
  }
}
//==================================================================================
__global__ void ResetStates(int nrays, NavigationState *out_states)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x)
    out_states[i].Clear();
}
//==================================================================================
__global__ void ValidateLocate(int nrays, NavigationState const *in_states, NavigationState const *out_states,
                               int *num_errors)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    if (out_states[i].GetNavIndex() != in_states[i].GetNavIndex()) atomicAdd(num_errors, 1);
  }
}
//==================================================================================
__global__ void ComputeSafetiesSurf(int nrays, Vector3D<Precision> const *points, NavigationState *in_states,
                                    Precision *safeties, bool validate_results)
{
  if (validate_results) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
      int exit_surf;
      safeties[i] = vgbrep::protonav::ComputeSafety<Precision, Real_t>(points[i], in_states[i], exit_surf);
    }
  } else {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
      int exit_surf;
      vgbrep::protonav::ComputeSafety<Precision, Real_t>(points[i], in_states[i], exit_surf);
    }
  }
}
//==================================================================================
__global__ void ComputeSafetiesSurfBVH(int nrays, Vector3D<Precision> const *points, NavigationState *in_states,
                                    Precision *safeties, bool validate_results)
{
  if (validate_results) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
      safeties[i] = vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeSafety(points[i], in_states[i]);
    }
  } else {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
      vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeSafety(points[i], in_states[i]);
    }
  }
}
//==================================================================================
__global__ void ComputeSafetiesSolid(int nrays, Vector3D<Precision> const *points, NavigationState const *in_states,
                                     Precision *ref_safeties, bool validate_results)
{
  if (validate_results) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
      ref_safeties[i] = LoopNavigator::ComputeSafety(points[i], in_states[i]);
    }
  } else {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
      LoopNavigator::ComputeSafety(points[i], in_states[i]);
    }
  }
}
//==================================================================================
__global__ void ComputeSafetiesSolidBVH(int nrays, Vector3D<Precision> const *points, NavigationState const *in_states,
                                        Precision *ref_safeties, bool validate_results)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    BVHNavigator::ComputeSafety(points[i], in_states[i]);
  }
}
//==================================================================================
__global__ void ValidateSafety(int nrays, Precision const *safeties, Precision const *refSafeties,
                               int *num_better_safety, int *num_worse_safety)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    atomicAdd(num_better_safety, int(safeties[i] > refSafeties[i] + kToleranceBVH));
    atomicAdd(num_worse_safety, int(safeties[i] < refSafeties[i] - kToleranceBVH));
  }
}
//==================================================================================
template <typename Navigator>
__device__ void PropagateRaySolid(int i, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                                  NavigationState const *in_states, Precision *length_over_crossings,
                                  int max_cross = vecgeom::kMaximumInt, bool debug = false, bool accept_zeros = false,
                                  bool validate_results = true)
{
  constexpr double kPushDistance = 1000 * vecgeom::kToleranceDist<Precision>;
  if (debug) {
    printf("CUDA PropagateRaysSolid debug ray %d:\n", i);
    printf("   ");
    in_states[i].Print();
  }
  NavigationState start_state = in_states[i];
  NavigationState out_state;
  int num_cross   = 0;
  double dist_tot = 0;
  auto const &dir = dirs[i];
  auto pt         = points[i] + kTolerance * dir; // push the start and subsequent crossing points
  do {
    auto distance =
        Navigator::ComputeStepAndPropagatedState(pt, dir, kInfLength, start_state, out_state, kPushDistance);
    if (debug) {
      printf("     dist = %15.10f\n", distance);
      printf("   ");
      out_state.Print();
    }
    dist_tot += (num_cross + 1) * distance;
    pt += distance * dir;
    start_state = out_state;
    num_cross++;
    if (accept_zeros && distance < 1000 * vecgeom::kTolerance) num_cross--;
  } while (!out_state.IsOutside() && num_cross < max_cross);

  if (validate_results) length_over_crossings[i] = num_cross ? dist_tot / (num_cross + 1) : 0;
}
//==================================================================================
__device__ void PropagateRaySurf(int i, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                                 NavigationState const *in_states, Precision *length_over_crossings,
                                 const VPlacedVolume *world, const SurfData *surfdata, bool debug = false,
                                 int max_cross = vecgeom::kMaximumInt, bool use_bvh = false, bool accept_zeros = false,
                                 bool validate_results = true)
{
  if (debug) {
    printf("PropagateRaysSurf debug ray %d:\n", i);
    printf("   ");
    in_states[i].Print();
  }
  NavigationState start_state = in_states[i];
  NavigationState out_state;
  int num_cross = 0;
  vgbrep::CrossedSurface crossed_surf;
  double dist_tot = 0;
  auto pt         = points[i];
  auto const &dir = dirs[i];
  do {
    crossed_surf.Set(0, 0, 0); // need to reset because the same inner tube surface can be crossed twice in a row

    Precision distance{0};
    if (!use_bvh) {
      distance = vgbrep::protonav::ComputeStepAndHit<Precision, Real_t>(pt, dir, start_state, out_state, crossed_surf);
    } else {
      distance =
          vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeStepAndHit(pt, dir, start_state, out_state, crossed_surf);
    }

    // exiting framed surface marked as overlapping, need to relocate
    if (surfdata->IsFSOverlapping(crossed_surf.hit_surf) && crossed_surf.hit_surf.GetFSindex() != -1) {
      vgbrep::protonav::ReLocatePointIn<Precision, Real_t>(start_state, pt + distance * dir, dir, out_state,
                                                           crossed_surf.exit_surf, distance);
    }
    if (crossed_surf.hit_surf.GetFSindex() == -1) {
      // Extruding overlap detected, relocating to correct starting state
      // Find true location for the crossing point
      NavigationState true_state;
      vgbrep::protonav::ReLocatePointIn<Precision, Real_t>(start_state, pt, dir, true_state, crossed_surf.exit_surf,
                                                           /*distance=*/Precision(0.));

      // Now replay to get correct distance
      if (!use_bvh) {
        distance = vgbrep::protonav::ComputeStepAndHit<Precision, Real_t>(pt, dir, true_state, out_state, crossed_surf);
      } else {
        distance =
            vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeStepAndHit(pt, dir, true_state, out_state, crossed_surf);
      }
      assert(distance != 0 && distance != vecgeom::InfinityLength<Precision>() &&
             "Distance after relocation shouldn't be 0 or infinity");
    }
    if (debug) {
      printf("     dist = %.16f  surf = %d\n", distance, crossed_surf.hit_surf.common_id);
      printf("   ");
      out_state.Print();
    }
    dist_tot += (num_cross + 1) * distance;
    pt += distance * dir;
    start_state = out_state;
    num_cross++;
    if (accept_zeros && distance < 1000 * vecgeom::kTolerance) num_cross--;
  } while (!out_state.IsOutside() && num_cross < max_cross);

  if (validate_results) length_over_crossings[i] = num_cross ? dist_tot / (num_cross + 1) : 0;
}
//==================================================================================
template <typename Navigator>
__global__ void PropagateRaysSolid(int nrays, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                                   NavigationState const *in_states, Precision *length_over_crossings,
                                   int max_cross = vecgeom::kMaximumInt, bool accept_zeros = false,
                                   bool validate_results = true)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    PropagateRaySolid<Navigator>(i, points, dirs, in_states, length_over_crossings, max_cross, /*debug=*/false,
                                 accept_zeros, validate_results);
  }
}
//==================================================================================
__global__ void PropagateRaysSurf(int nrays, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                                  NavigationState const *in_states, Precision *length_over_crossings,
                                  const VPlacedVolume *world, const SurfData *surfdata, bool debug = false,
                                  int max_cross = vecgeom::kMaximumInt, bool use_bvh = false, bool accept_zeros = false,
                                  bool validate_results = true)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    PropagateRaySurf(i, points, dirs, in_states, length_over_crossings, world, surfdata, debug, max_cross, use_bvh,
                     accept_zeros, validate_results);
  }
}
//==================================================================================
__global__ void ValidateTraversal(int nrays, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                                  NavigationState const *in_states, Precision *length_over_crossings,
                                  Precision *refLength_over_crossings, int *num_errors, bool debug,
                                  const VPlacedVolume *world, const SurfData *surfdata,
                                  int max_cross = vecgeom::kMaximumInt, bool use_bvh = false, bool accept_zeros = false)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    bool error_dist =
        Abs(length_over_crossings[i] - refLength_over_crossings[i]) >
        vgbrep::RoundingError(static_cast<Precision>(refLength_over_crossings[i]), 1000 * kToleranceDist<Precision>);
    if (error_dist && debug && *num_errors == 0) {
      printf("Error on GPU with BVH: %i in ray %i: length_over_crossings[i] %f refLength_over_crossings[i] %f -point "
             "%.10f %.10f %.10f -direction %.10f %.10f %.10f \n",
             use_bvh, i, length_over_crossings[i], refLength_over_crossings[i], points[i][0], points[i][1],
             points[i][2], dirs[i][0], dirs[i][1], dirs[i][2]);
      // note that the printouts in the propagation would be a mess since there is no GPU synchronization in between.
      // printf("Replaying solid model...\n");
      // PropagateRaySolid<LoopNavigator>(i, points, dirs, in_states, refLength_over_crossings, debug, max_cross,
      // accept_zeros); printf("Replaying surface model with BVH %i", use_bvh); PropagateRaySurf(i, points, dirs,
      // in_states, length_over_crossings, world, surfdata, debug, max_cross,
      //                  use_bvh, accept_zeros);
    }
    atomicAdd(num_errors, int(error_dist));
  }
}
//==================================================================================
int testRaytracingCUDA(int nrays, Vec3Dc const *pointsc, Vec3Dc const *dirsc, const SurfData &surfdata, bool debug,
                       bool accept_zeros = 0, int max_cross = vecgeom::kMaximumInt, bool test_bvh = false,
                       bool validate_results = true)
{
  BrepCudaManager::Instance().TransferSurfData(surfdata);
  auto surfdata_D = BrepCudaManager::Instance().GetDevicePtr();

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
  NavigationState *origStates;
  BREP_CUDA_CHECK(cudaMalloc(&origStates, nrays * sizeof(NavigationState)));
  NavigationState *outputStates;
  BREP_CUDA_CHECK(cudaMalloc(&outputStates, nrays * sizeof(NavigationState)));
  NavigationState *outputStatesBVH;
  BREP_CUDA_CHECK(cudaMalloc(&outputStatesBVH, nrays * sizeof(NavigationState)));
  Precision *refSafeties{nullptr};
  Precision *safeties{nullptr};
  Precision *bvhSafeties{nullptr};
  Precision *refLength_over_crossings{nullptr};
  Precision *length_over_crossings{nullptr};
  Precision *length_over_crossings_bvh{nullptr};

  if (validate_results) {
    BREP_CUDA_CHECK(cudaMalloc(&refSafeties, nrays * sizeof(Precision)));
    BREP_CUDA_CHECK(cudaMalloc(&safeties, nrays * sizeof(Precision)));
    BREP_CUDA_CHECK(cudaMalloc(&bvhSafeties, nrays * sizeof(Precision)));
    BREP_CUDA_CHECK(cudaMalloc(&refLength_over_crossings, nrays * sizeof(Precision)));
    BREP_CUDA_CHECK(cudaMalloc(&length_over_crossings, nrays * sizeof(Precision)));
    BREP_CUDA_CHECK(cudaMalloc(&length_over_crossings_bvh, nrays * sizeof(Precision)));
  }

  int num_errors          = 0;
  int num_errors_bvh_loc  = 0;
  int num_better_safety   = 0;
  int num_worse_safety    = 0;
  int num_errors_dist     = 0;
  int num_errors_dist_bvh = 0;
  int *num_errors_d, *num_better_safety_d, *num_worse_safety_d, *num_errors_dist_d, *num_errors_dist_bvh_d;
  BREP_CUDA_CHECK(cudaMalloc(&num_errors_d, sizeof(int)));
  BREP_CUDA_CHECK(cudaMalloc(&num_better_safety_d, sizeof(int)));
  BREP_CUDA_CHECK(cudaMalloc(&num_worse_safety_d, sizeof(int)));
  BREP_CUDA_CHECK(cudaMalloc(&num_errors_dist_d, sizeof(int)));
  BREP_CUDA_CHECK(cudaMalloc(&num_errors_dist_bvh_d, sizeof(int)));

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
  LocateSurf<<<initBlocks, initThreads>>>(nrays, points, outputStates, world_dev);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_locate_surf = timer.Stop();

  // Locating the global points with surface model + BVH
  timer.Start();
  if (test_bvh) LocateSurfBVH<<<initBlocks, initThreads>>>(nrays, points, outputStatesBVH, world_dev);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_locate_surf_bvh = timer.Stop();

  // Corectness for locating points
  ValidateLocate<<<initBlocks, initThreads>>>(nrays, origStates, outputStates, num_errors_d);
  BREP_CUDA_CHECK(cudaMemcpy(&num_errors, num_errors_d, sizeof(int), cudaMemcpyDeviceToHost));
  BREP_CUDA_CHECK(cudaDeviceSynchronize());

  if (num_errors > 0) std::cout << "CUDA: Point locate errors: " << num_errors << "\n";

  // Validate BVH Locate
  ValidateLocate<<<initBlocks, initThreads>>>(nrays, origStates, outputStatesBVH, num_errors_d);
  BREP_CUDA_CHECK(cudaMemcpy(&num_errors_bvh_loc, num_errors_d, sizeof(int), cudaMemcpyDeviceToHost));
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  if (num_errors_bvh_loc > 0) std::cout << "CUDA: BVH point locate errors: " << num_errors_bvh_loc << "\n";

  if (!debug) {
    std::cout << "CUDA: locate_solids: " << time_locate_solids << "  locate_solids_BVH: " << time_locate_solids_bvh
              << "  locate_surf: " << time_locate_surf;
    if (test_bvh) {
      std::cout << " locate_surf_BVH: " << time_locate_surf_bvh << "\n";
    } else {
      std::cout << "\n";
    }
  }

  // Safety for solids model (reference)
  timer.Start();
  ComputeSafetiesSolid<<<initBlocks, initThreads>>>(nrays, points, origStates, refSafeties, validate_results);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_safety_solids = timer.Stop();

  // Safety for solids model with BVH
  timer.Start();
  ComputeSafetiesSolidBVH<<<initBlocks, initThreads>>>(nrays, points, origStates, safeties, validate_results);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_safety_solids_bvh = timer.Stop();

  // Safety for surface model
  timer.Start();
  ComputeSafetiesSurf<<<initBlocks, initThreads>>>(nrays, points, origStates, safeties, validate_results);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_safety_surf = timer.Stop();

  timer.Start();
  if(test_bvh)
    ComputeSafetiesSurfBVH<<<initBlocks, initThreads>>>(nrays, points, origStates, bvhSafeties, validate_results);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_safety_surf_bvh = timer.Stop();

  if (validate_results) {
    ValidateSafety<<<initBlocks, initThreads>>>(nrays, safeties, refSafeties, num_better_safety_d, num_worse_safety_d);
    BREP_CUDA_CHECK(cudaMemcpy(&num_better_safety, num_better_safety_d, sizeof(int), cudaMemcpyDeviceToHost));
    BREP_CUDA_CHECK(cudaMemcpy(&num_worse_safety, num_worse_safety_d, sizeof(int), cudaMemcpyDeviceToHost));
    BREP_CUDA_CHECK(cudaDeviceSynchronize());

    if (num_better_safety > 0) std::cout << "CUDA:    number of better safety values: " << num_better_safety << "\n";
    if (num_worse_safety > 0) std::cout << "CUDA:    number of worse safety values: " << num_worse_safety << "\n";
  
    if(test_bvh)
    {
      // Init counters
      num_better_safety = num_worse_safety = 0;
      BREP_CUDA_CHECK(cudaMemcpy(num_better_safety_d, &num_better_safety, sizeof(int), cudaMemcpyHostToDevice));
      BREP_CUDA_CHECK(cudaMemcpy(num_worse_safety_d, &num_worse_safety, sizeof(int), cudaMemcpyHostToDevice));

      ValidateSafety<<<initBlocks, initThreads>>>(nrays, bvhSafeties, refSafeties, num_better_safety_d, num_worse_safety_d);
      BREP_CUDA_CHECK(cudaMemcpy(&num_better_safety, num_better_safety_d, sizeof(int), cudaMemcpyDeviceToHost));
      BREP_CUDA_CHECK(cudaMemcpy(&num_worse_safety, num_worse_safety_d, sizeof(int), cudaMemcpyDeviceToHost));
      BREP_CUDA_CHECK(cudaDeviceSynchronize());

      if (num_better_safety > 0) std::cout << "CUDA:    BVH number of better safety values: " << num_better_safety << "\n";
      if (num_worse_safety > 0) std::cout << "CUDA:    BVH number of worse safety values: " << num_worse_safety << "\n";
    }
  }

  if (!debug)
  {
    std::cout << "CUDA: safety_solids: " << time_safety_solids << " safety_solids_BVH: " << time_safety_solids_bvh
              << "  safety_surf: " << time_safety_surf;
    if(test_bvh)
    {
      std::cout << "  safety_surf_bvh: " << time_safety_surf_bvh << "\n";
    }
    else
    {
      std::cout << "\n";
    }
  }
  // Traversal for solids model (reference)
  timer.Start();
  PropagateRaysSolid<LoopNavigator><<<initBlocks, initThreads>>>(
      nrays, points, dirs, origStates, refLength_over_crossings, max_cross, accept_zeros, validate_results);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_traverse_solids = timer.Stop();

  // Traversal for solids model with BVH
  timer.Start();
  PropagateRaysSolid<BVHNavigator><<<initBlocks, initThreads>>>(nrays, points, dirs, origStates, length_over_crossings,
                                                                max_cross, accept_zeros, validate_results);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_traverse_solids_bvh = timer.Stop();

  // Traversal for the surface model
  timer.Start();
  PropagateRaysSurf<<<initBlocks, initThreads>>>(nrays, points, dirs, origStates, length_over_crossings, world_dev,
                                                 surfdata_D, false, max_cross, false, accept_zeros, validate_results);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_traverse_surf = timer.Stop();

  // Traversal for the surface model with BVH
  timer.Start();
  if (test_bvh)
    PropagateRaysSurf<<<initBlocks, initThreads>>>(nrays, points, dirs, origStates, length_over_crossings_bvh,
                                                   world_dev, surfdata_D, false, max_cross, true, accept_zeros,
                                                   validate_results);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_traverse_surf_bvh = timer.Stop();

  if (validate_results) {
    ValidateTraversal<<<initBlocks, initThreads>>>(nrays, points, dirs, origStates, length_over_crossings,
                                                   refLength_over_crossings, num_errors_dist_d, debug, world_dev,
                                                   surfdata_D, max_cross, /*use_bvh=*/false, accept_zeros);
    BREP_CUDA_CHECK(cudaMemcpy(&num_errors_dist, num_errors_dist_d, sizeof(int), cudaMemcpyDeviceToHost));
    BREP_CUDA_CHECK(cudaDeviceSynchronize());

    if (test_bvh)
      ValidateTraversal<<<initBlocks, initThreads>>>(nrays, points, dirs, origStates, length_over_crossings_bvh,
                                                     refLength_over_crossings, num_errors_dist_bvh_d, debug, world_dev,
                                                     surfdata_D, max_cross, /*use_bvh=*/true, accept_zeros);
    BREP_CUDA_CHECK(cudaMemcpy(&num_errors_dist_bvh, num_errors_dist_bvh_d, sizeof(int), cudaMemcpyDeviceToHost));
    BREP_CUDA_CHECK(cudaDeviceSynchronize());

    if (num_errors_dist > 0 || num_errors_dist_bvh > 0)
      std::cout << "CUDA: traversal errors looper: " << num_errors_dist
                << " traversal errors BVH: " << num_errors_dist_bvh << "\n";
  }
  if (!debug) {
    std::cout << "CUDA: traverse_solids: " << time_traverse_solids
              << " traverse_solids_bvh: " << time_traverse_solids_bvh << "  traverse_surf: " << time_traverse_surf;
    if (test_bvh)
      std::cout << "  traverse_surf BVH: " << time_traverse_surf_bvh << "\n";
    else
      std::cout << "\n";
  }

  num_errors += num_errors_dist + num_errors_dist_bvh;
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
  BREP_CUDA_CHECK(cudaFree(outputStatesBVH));
  if (validate_results) // These arrays will be nullptr if we are not doing validation
  {
    BREP_CUDA_CHECK(cudaFree(refSafeties));
    BREP_CUDA_CHECK(cudaFree(safeties));
    BREP_CUDA_CHECK(cudaFree(refLength_over_crossings));
    BREP_CUDA_CHECK(cudaFree(length_over_crossings));
    BREP_CUDA_CHECK(cudaFree(length_over_crossings_bvh));
  }
  return num_errors;
}
