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
__global__ void ResetStates(int nrays, NavigationState *out_states)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x)
    out_states[i].Clear();
}
//==================================================================================
__global__ void LocateSurf(int nrays, Vector3D<Precision> const *points, NavigationState *out_states)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    // Locate with surface-based model
    vgbrep::protonav::LocatePointIn<Precision, Real_t>(NavigationState::WorldId(), points[i], out_states[i], true);
  }
}
//==================================================================================
__global__ void LocateSurfBVH(int nrays, Vector3D<Precision> const *points, NavigationState *out_states)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    // Locate with surface-based model + BVH
    vgbrep::protonav::BVHSurfNavigator<Real_t>::LocatePointIn(NavigationState::WorldId(), points[i], out_states[i],
                                                              true);
  }
}
//==================================================================================
__device__ void PropagateRaySurf(int i, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                                 NavigationState const *in_states, Precision *length_over_crossings,
                                 const SurfData *surfdata, bool debug = false, int max_cross = vecgeom::kMaximumInt,
                                 bool accept_zeros = false, bool validate_results = true)
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
    // USING THE LOOPER
    distance = vgbrep::protonav::ComputeStepAndHit<Precision, Real_t>(pt, dir, start_state, out_state, crossed_surf);

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
      distance = vgbrep::protonav::ComputeStepAndHit<Precision, Real_t>(pt, dir, true_state, out_state, crossed_surf);
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
__device__ void PropagateRaySurfBVH(int i, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                                    NavigationState const *in_states, Precision *length_over_crossings,
                                    const SurfData *surfdata, bool debug = false, int max_cross = vecgeom::kMaximumInt,
                                    bool accept_zeros = false, bool validate_results = true)
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
    // USING THE BVH
    distance =
        vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeStepAndHit(pt, dir, start_state, out_state, crossed_surf);

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
      distance =
          vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeStepAndHit(pt, dir, true_state, out_state, crossed_surf);

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

__device__ void PropagateRaySurfBVHSingle(int i, Vector3D<Precision> *points, Vector3D<Precision> const *dirs,
                                          NavigationState *in_states, Precision *length_over_crossings,
                                          bool debug = false, bool validate_results = true)
{
  if (in_states[i].IsOutside()) return;
  if (debug) {
    printf("PropagateRaySurfBVHSingle debug ray %d:\n", i);
    printf("   ");
    in_states[i].Print();
  }
  NavigationState start_state = in_states[i];
  NavigationState out_state;
  vgbrep::CrossedSurface crossed_surf;
  // commented out, needed if validation should be added
  // double dist_tot = 0;
  // int num_cross = 0;
  auto &pt        = points[i];
  auto const &dir = dirs[i];

  Precision distance{0};
  // USING THE BVH
  distance =
      vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeStepAndHit(pt, dir, start_state, out_state, crossed_surf);

  if (debug) {
    printf("     dist = %.16f  surf = %d\n", distance, crossed_surf.hit_surf.common_id);
    printf("   ");
    out_state.Print();
  }
  // dist_tot += (num_cross + 1) * distance;
  points[i] += distance * dir;
  in_states[i] = out_state;
  // num_cross++;
  // if (accept_zeros && distance < 1000 * vecgeom::kTolerance) num_cross--;
  // if (validate_results) length_over_crossings[i] = num_cross ? dist_tot / (num_cross + 1) : 0;
}

__device__ void PropagateRayFS(int i, Vector3D<Precision> *points, Vector3D<Precision> const *dirs,
                               NavigationState *in_states, long *hitcandidate_index, Precision *distances,
                               Precision *length_over_crossings, bool debug = false, bool validate_results = true)
{
  if (in_states[i].IsOutside()) return;
  if (debug) {
    printf("PropagateRayFS debug ray %d:\n", i);
    printf("   ");
    in_states[i].Print();
  }
  NavigationState start_state = in_states[i];
  NavigationState out_state;
  // int num_cross = 0;
  vgbrep::CrossedSurface crossed_surf;
  // double dist_tot = 0;
  auto &pt        = points[i];
  auto const &dir = dirs[i];
  auto &hit_index = hitcandidate_index[i];
  auto &distance  = distances[i];

  distance = vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeFSStepAndHit(pt, dir, start_state, hit_index);
  if (debug) {
    printf("     dist = %.16f\n", distance);
  }
}

__device__ void RelocateRayCS(int i, Vector3D<Precision> *points, Vector3D<Precision> const *dirs,
                              NavigationState *in_states, long *hitcandidate_index, Precision *distances,
                              Precision *length_over_crossings, bool debug = false, bool validate_results = true)
{
  if (in_states[i].IsOutside()) return;
  if (debug) {
    printf("RelocateRayCS debug ray %d:\n", i);
    printf("   ");
    in_states[i].Print();
  }
  NavigationState start_state = in_states[i];
  NavigationState out_state;
  vgbrep::CrossedSurface crossed_surf;
  auto &pt        = points[i];
  auto const &dir = dirs[i];
  auto &hit_index = hitcandidate_index[i];
  auto &distance  = distances[i];

  vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeCSRelocation(pt, dir, start_state, out_state, crossed_surf,
                                                                  hit_index, distance);

  points[i] += distance * dir;
  in_states[i] = out_state;
}

//==================================================================================
__global__ void PropagateRaysSurf(int nrays, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                                  NavigationState const *in_states, Precision *length_over_crossings,
                                  const SurfData *surfdata, bool debug = false, int max_cross = vecgeom::kMaximumInt,
                                  bool accept_zeros = false, bool validate_results = true)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    PropagateRaySurf(i, points, dirs, in_states, length_over_crossings, surfdata, debug, max_cross, accept_zeros,
                     validate_results);
  }
}

//==================================================================================
__global__ void PropagateRaysSurfBVH(int nrays, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                                     NavigationState const *in_states, Precision *length_over_crossings,
                                     const SurfData *surfdata, bool debug = false, int max_cross = vecgeom::kMaximumInt,
                                     bool accept_zeros = false, bool validate_results = true)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    PropagateRaySurfBVH(i, points, dirs, in_states, length_over_crossings, surfdata, debug, max_cross, accept_zeros,
                        validate_results);
  }
}

__global__ void PropagateRaysSurfBVHSingle(int nrays, int *alive_indices, Vector3D<Precision> *points,
                                           Vector3D<Precision> const *dirs, NavigationState *in_states,
                                           Precision *length_over_crossings, bool debug = false,
                                           int max_cross = vecgeom::kMaximumInt, bool accept_zeros = false,
                                           bool validate_results = true)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    PropagateRaySurfBVHSingle(alive_indices[i], points, dirs, in_states, length_over_crossings, debug,
                              validate_results);
  }
}

__global__ void PropagateRaysFS(int nrays, int *alive_indices, Vector3D<Precision> *points,
                                Vector3D<Precision> const *dirs, NavigationState *in_states, long *hitcandidate_index,
                                Precision *distances, Precision *length_over_crossings, bool debug = false,
                                int max_cross = vecgeom::kMaximumInt, bool validate_results = true)
{

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    PropagateRayFS(alive_indices[i], points, dirs, in_states, hitcandidate_index, distances, length_over_crossings,
                   debug, validate_results);
  }
}

__global__ void RelocateRaysCS(int nrays, int *alive_indices, Vector3D<Precision> *points,
                               Vector3D<Precision> const *dirs, NavigationState *in_states, long *hitcandidate_index,
                               Precision *distances, Precision *length_over_crossings, bool debug = false,
                               int max_cross = vecgeom::kMaximumInt, bool validate_results = true)
{

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nrays; i += blockDim.x * gridDim.x) {
    RelocateRayCS(alive_indices[i], points, dirs, in_states, hitcandidate_index, distances, length_over_crossings,
                  debug, validate_results);
  }
}

__global__ void checkInsideStatesKernel(int nrays, const vecgeom::NavStateTuple *in_states, bool *result)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nrays) {
    if (!in_states[i].IsOutside()) {
      *result = true;
    }
  }
}

__global__ void filterAliveRays(const NavigationState *in_states, const int *indices, int *alive_indices,
                                int *alive_count, int active_count)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= active_count) return;

  int current_index = indices[idx];

  // Check if the ray is not outside
  if (!in_states[current_index].IsOutside()) {
    int out_idx            = atomicAdd(alive_count, 1);
    alive_indices[out_idx] = current_index; // Store the original index of the alive ray
  }
}

void PropagateRaysSurfBVH(int initThreads, int nrays, Vector3D<Precision> *points, Vector3D<Precision> *dirs,
                          NavigationState *in_states, Precision *length_over_crossings, bool debug = false,
                          int max_cross = vecgeom::kMaximumInt, bool accept_zeros = false, bool validate_results = true,
                          int verbosity = 0)
{
  // This function does traversal for nrays with the BVH in a single step mode, such that each step is done in a single
  // kernel, similar to what is done in AdePT in a real simulation
  int *indices;
  cudaMalloc(&indices, nrays * sizeof(int));
  std::vector<int> host_indices(nrays);
  std::iota(host_indices.begin(), host_indices.end(), 0);
  cudaMemcpy(indices, host_indices.data(), nrays * sizeof(int), cudaMemcpyHostToDevice);

  int *alive_indices;
  cudaMalloc(&alive_indices, nrays * sizeof(int));

  int *alive_count;
  cudaMalloc(&alive_count, sizeof(int));
  cudaMemset(alive_count, 0, sizeof(int));

  int host_alive_count = nrays;
  int num_cross        = 0;
  Stopwatch timer;
  double time_per_step = 0.;

  while (host_alive_count) {
    int initBlocks = (host_alive_count + initThreads - 1) / initThreads;

    timer.Start();
    PropagateRaysSurfBVHSingle<<<initBlocks, initThreads>>>(host_alive_count, indices, points, dirs, in_states,
                                                            length_over_crossings, false, max_cross, accept_zeros,
                                                            validate_results);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());

    time_per_step = timer.Stop();
    if (verbosity > 3) {
      std::cout << "istep: " << num_cross << " number of alive tracks: " << host_alive_count
                << " time : " << time_per_step << std::endl;
    }

    cudaMemset(alive_count, 0, sizeof(int));
    filterAliveRays<<<initBlocks, initThreads>>>(in_states, indices, alive_indices, alive_count, host_alive_count);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());
    cudaMemcpy(&host_alive_count, alive_count, sizeof(int), cudaMemcpyDeviceToHost);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());

    // Swap pointers so that indices now points to the new alive_indices for the next iteration
    std::swap(indices, alive_indices);

    num_cross++;
  }

  cudaFree(indices);
  cudaFree(alive_indices);
  cudaFree(alive_count);
}

void PropagateRaysSurfBVHSplit(int initThreads, int nrays, Vector3D<Precision> *points, Vector3D<Precision> *dirs,
                               NavigationState *in_states, Precision *length_over_crossings, bool debug = false,
                               int max_cross = vecgeom::kMaximumInt, bool accept_zeros = false,
                               bool validate_results = true, int verbosity = 0)
{
  // This function does traversal for nrays with the BVH in a single step mode and splits the traversal kernel into the
  // approach , such that each step is done in a single kernel, similar to what is done in AdePT in a real simulation
  int *indices;
  // Initialize initial indices
  cudaMalloc(&indices, nrays * sizeof(int));
  std::vector<int> host_indices(nrays);
  std::iota(host_indices.begin(), host_indices.end(), 0);
  cudaMemcpy(indices, host_indices.data(), nrays * sizeof(int), cudaMemcpyHostToDevice);
  int *alive_indices;
  cudaMalloc(&alive_indices, nrays * sizeof(int));
  int *alive_count;
  cudaMalloc(&alive_count, sizeof(int));
  cudaMemset(alive_count, 0, sizeof(int));

  long *hitcandidate_index;
  cudaMalloc(&hitcandidate_index, nrays * sizeof(long));
  cudaMemset(hitcandidate_index, 0, sizeof(long));

  Precision *distances;
  cudaMalloc(&distances, nrays * sizeof(Precision));
  cudaMemset(distances, 0, sizeof(Precision));

  int host_alive_count = nrays;
  int num_cross        = 0;
  Stopwatch timer;
  double time_per_step = 0.;

  int Blocks;

  while (host_alive_count) {
    Blocks = (host_alive_count + initThreads - 1) / initThreads;
    timer.Start();

    PropagateRaysFS<<<Blocks, initThreads>>>(host_alive_count, indices, points, dirs, in_states, hitcandidate_index,
                                             distances, length_over_crossings, false, max_cross, validate_results);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());
    RelocateRaysCS<<<Blocks, initThreads>>>(host_alive_count, indices, points, dirs, in_states, hitcandidate_index,
                                            distances, length_over_crossings, false, max_cross, validate_results);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());
    time_per_step = timer.Stop();
    if (verbosity > 3) {
      std::cout << "istep: " << num_cross << " number of alive tracks: " << host_alive_count
                << " time : " << time_per_step << std::endl;
    }

    cudaMemset(alive_count, 0, sizeof(int));
    filterAliveRays<<<Blocks, initThreads>>>(in_states, indices, alive_indices, alive_count, host_alive_count);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());
    cudaMemcpy(&host_alive_count, alive_count, sizeof(int), cudaMemcpyDeviceToHost);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());

    // Swap pointers so that indices now points to the new alive_indices for the next iteration
    std::swap(indices, alive_indices);

    num_cross++;
  }

  cudaFree(indices);
  cudaFree(alive_indices);
  cudaFree(alive_count);
  cudaFree(distances);
  cudaFree(hitcandidate_index);
}

//==================================================================================
int testRaytracingCUDA(int nrays, Vec3Dc const *pointsc, Vec3Dc const *dirsc, const SurfData &surfdata, bool debug,
                       bool accept_zeros = 0, int max_cross = vecgeom::kMaximumInt, bool test_bvh = false,
                       bool validate_results = true, bool only_surf = false, bool bvh_single_step = false,
                       bool bvh_split_step = false, int verbosity = 0)
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
  Vec3D *points_NC;
  BREP_CUDA_CHECK(cudaMalloc(&points_NC, nrays * sizeof(Vec3D)));
  BREP_CUDA_CHECK(cudaMemcpy(points_NC, pointsh, nrays * sizeof(Vec3D), cudaMemcpyHostToDevice));
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

  if (only_surf) debug = false;
  if (validate_results) {
    BREP_CUDA_CHECK(cudaMalloc(&refSafeties, nrays * sizeof(Precision)));
    BREP_CUDA_CHECK(cudaMalloc(&safeties, nrays * sizeof(Precision)));
    BREP_CUDA_CHECK(cudaMalloc(&bvhSafeties, nrays * sizeof(Precision)));
    BREP_CUDA_CHECK(cudaMalloc(&refLength_over_crossings, nrays * sizeof(Precision)));
    BREP_CUDA_CHECK(cudaMalloc(&length_over_crossings, nrays * sizeof(Precision)));
    BREP_CUDA_CHECK(cudaMalloc(&length_over_crossings_bvh, nrays * sizeof(Precision)));
  }

  constexpr int initThreads = 32;
  int initBlocks            = (nrays + initThreads - 1) / initThreads;

  Stopwatch timer;
  double time_locate_solids{0}, time_locate_solids_bvh{0}, time_locate_surf{0}, time_locate_surf_bvh{0},
      time_safety_solids{0}, time_safety_solids_bvh{0}, time_safety_surf{0}, time_safety_surf_bvh{0},
      time_traverse_solids{0}, time_traverse_solids_bvh{0}, time_traverse_surf{0}, time_traverse_surf_bvh{0};
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

  // Locating the global points with solid model
  if (!only_surf) {
    auto world_dev = vecgeom::cxx::CudaManager::Instance().world_gpu();
    timer.Start();
    LocateSolids<<<initBlocks, initThreads>>>(nrays, points, origStates, world_dev);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());
    time_locate_solids = timer.Stop();

    // Locating the global points with solid model + BVH
    timer.Start();
    LocateSolidsBVH<<<initBlocks, initThreads>>>(nrays, points, outputStates, world_dev);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());
    time_locate_solids_bvh = timer.Stop();

    ResetStates<<<initBlocks, initThreads>>>(nrays, outputStates);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Locating the global points with surface model
  timer.Start();
  if (only_surf)
    LocateSurf<<<initBlocks, initThreads>>>(nrays, points, origStates);
  else
    LocateSurf<<<initBlocks, initThreads>>>(nrays, points, outputStates);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  time_locate_surf = timer.Stop();

  // Locating the global points with surface model + BVH
  timer.Start();
  if (test_bvh) LocateSurfBVH<<<initBlocks, initThreads>>>(nrays, points, outputStatesBVH);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  time_locate_surf_bvh = timer.Stop();

  if (!only_surf) {
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
  }

  if (!debug) {
    if (only_surf)
      std::cout << "CUDA: locate_surf: " << time_locate_surf;
    else
      std::cout << "CUDA: locate_solids: " << time_locate_solids << "  locate_solids_BVH: " << time_locate_solids_bvh
                << "  locate_surf: " << time_locate_surf;
    if (test_bvh) {
      std::cout << " locate_surf_BVH: " << time_locate_surf_bvh << "\n";
    } else {
      std::cout << "\n";
    }
  }

  if (!only_surf) {
    // Safety for solids model (reference)
    timer.Start();
    ComputeSafetiesSolid<<<initBlocks, initThreads>>>(nrays, points, origStates, refSafeties, validate_results);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());
    time_safety_solids = timer.Stop();

    // Safety for solids model with BVH
    timer.Start();
    ComputeSafetiesSolidBVH<<<initBlocks, initThreads>>>(nrays, points, origStates, safeties, validate_results);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());
    time_safety_solids_bvh = timer.Stop();
  }

  // Safety for surface model
  timer.Start();
  ComputeSafetiesSurf<<<initBlocks, initThreads>>>(nrays, points, origStates, safeties, validate_results);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  time_safety_surf = timer.Stop();

  timer.Start();
  if (test_bvh)
    ComputeSafetiesSurfBVH<<<initBlocks, initThreads>>>(nrays, points, origStates, bvhSafeties, validate_results);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  time_safety_surf_bvh = timer.Stop();

  if (!only_surf) {
    if (validate_results) {
      ValidateSafety<<<initBlocks, initThreads>>>(nrays, safeties, refSafeties, num_better_safety_d,
                                                  num_worse_safety_d);
      BREP_CUDA_CHECK(cudaMemcpy(&num_better_safety, num_better_safety_d, sizeof(int), cudaMemcpyDeviceToHost));
      BREP_CUDA_CHECK(cudaMemcpy(&num_worse_safety, num_worse_safety_d, sizeof(int), cudaMemcpyDeviceToHost));
      BREP_CUDA_CHECK(cudaDeviceSynchronize());

      if (num_better_safety > 0) std::cout << "CUDA:    number of better safety values: " << num_better_safety << "\n";
      if (num_worse_safety > 0) std::cout << "CUDA:    number of worse safety values: " << num_worse_safety << "\n";

      if (test_bvh) {
        // Init counters
        num_better_safety = num_worse_safety = 0;
        BREP_CUDA_CHECK(cudaMemcpy(num_better_safety_d, &num_better_safety, sizeof(int), cudaMemcpyHostToDevice));
        BREP_CUDA_CHECK(cudaMemcpy(num_worse_safety_d, &num_worse_safety, sizeof(int), cudaMemcpyHostToDevice));

        ValidateSafety<<<initBlocks, initThreads>>>(nrays, bvhSafeties, refSafeties, num_better_safety_d,
                                                    num_worse_safety_d);
        BREP_CUDA_CHECK(cudaMemcpy(&num_better_safety, num_better_safety_d, sizeof(int), cudaMemcpyDeviceToHost));
        BREP_CUDA_CHECK(cudaMemcpy(&num_worse_safety, num_worse_safety_d, sizeof(int), cudaMemcpyDeviceToHost));
        BREP_CUDA_CHECK(cudaDeviceSynchronize());

        if (num_better_safety > 0)
          std::cout << "CUDA:    BVH number of better safety values: " << num_better_safety << "\n";
        if (num_worse_safety > 0)
          std::cout << "CUDA:    BVH number of worse safety values: " << num_worse_safety << "\n";
      }
    }
  }

  if (!debug) {
    if (only_surf)
      std::cout << "CUDA: safety_surf: " << time_safety_surf;
    else
      std::cout << "CUDA: safety_solids: " << time_safety_solids << " safety_solids_BVH: " << time_safety_solids_bvh
                << "  safety_surf: " << time_safety_surf;
    if (test_bvh) {
      std::cout << "  safety_surf_bvh: " << time_safety_surf_bvh << "\n";
    } else {
      std::cout << "\n";
    }
  }

  if (!only_surf) {
    // Traversal for solids model (reference)
    timer.Start();
    PropagateRaysSolid<LoopNavigator><<<initBlocks, initThreads>>>(
        nrays, points, dirs, origStates, refLength_over_crossings, max_cross, accept_zeros, validate_results);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());
    time_traverse_solids = timer.Stop();

    // Traversal for solids model with BVH
    timer.Start();
    PropagateRaysSolid<BVHNavigator><<<initBlocks, initThreads>>>(
        nrays, points, dirs, origStates, length_over_crossings, max_cross, accept_zeros, validate_results);
    BREP_CUDA_CHECK(cudaDeviceSynchronize());
    time_traverse_solids_bvh = timer.Stop();
  }

  // Traversal for the surface model
  timer.Start();
  PropagateRaysSurf<<<initBlocks, initThreads>>>(nrays, points, dirs, origStates, length_over_crossings, surfdata_D,
                                                 false, max_cross, accept_zeros, validate_results);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  time_traverse_surf = timer.Stop();

  // Traversal for the surface model with BVH
  timer.Start();
  if (test_bvh)
    PropagateRaysSurfBVH<<<initBlocks, initThreads>>>(nrays, points, dirs, origStates, length_over_crossings_bvh,
                                                      surfdata_D, false, max_cross, accept_zeros, validate_results);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());
  time_traverse_surf_bvh = timer.Stop();

  // Traversal for the surface model with BVH in single step mode
  NavigationState *BVH_single_states;
  BREP_CUDA_CHECK(cudaMalloc(&BVH_single_states, nrays * sizeof(NavigationState)));
  BREP_CUDA_CHECK(cudaMemcpy(BVH_single_states, origStates, nrays * sizeof(NavigationState), cudaMemcpyDeviceToDevice));

  // Traversal for the surface model with BVH in a single step mode, each step is done in a separate kernel launch.
  timer.Start();
  if (bvh_single_step)
    PropagateRaysSurfBVH(initThreads, nrays, points_NC, dirs, BVH_single_states, length_over_crossings_bvh, false,
                         max_cross, accept_zeros, validate_results, verbosity);
  auto time_traverse_surf_bvh_single = timer.Stop();

  // Traversal for the surface model with BVH in a single step mode, each step is done in a separate kernel launch.
  // The transport kernel is split, one for identifying the framed surface, one for the relocation on the common surface
  timer.Start();
  if (bvh_split_step) {
    // reset points and in_states, as they were changed by the single step kernel above
    BREP_CUDA_CHECK(
        cudaMemcpy(BVH_single_states, origStates, nrays * sizeof(NavigationState), cudaMemcpyDeviceToDevice));
    BREP_CUDA_CHECK(cudaMemcpy(points_NC, pointsh, nrays * sizeof(Vec3D), cudaMemcpyHostToDevice));
    BREP_CUDA_CHECK(cudaDeviceSynchronize());

    PropagateRaysSurfBVHSplit(initThreads, nrays, points_NC, dirs, BVH_single_states, length_over_crossings_bvh, false,
                              max_cross, accept_zeros, validate_results, verbosity);
  }
  auto time_traverse_surf_bvh_single_split = timer.Stop();

  if (!only_surf && validate_results) {
    auto world_dev = vecgeom::cxx::CudaManager::Instance().world_gpu();
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
    if (only_surf)
      std::cout << "CUDA: traverse_surf: " << time_traverse_surf;
    else
      std::cout << "CUDA: traverse_solids: " << time_traverse_solids
                << " traverse_solids_bvh: " << time_traverse_solids_bvh << "  traverse_surf: " << time_traverse_surf;
    if (test_bvh) std::cout << "  traverse_surf BVH: " << time_traverse_surf_bvh;
    if (bvh_single_step) std::cout << "  traverse_surf BVH single steps: " << time_traverse_surf_bvh_single;
    if (bvh_split_step) std::cout << "  traverse_surf BVH single split steps: " << time_traverse_surf_bvh_single_split;
    std::cout << "\n";
  }

  BrepCudaManager::Instance().Cleanup();
  delete[] pointsh;
  delete[] dirsh;
  if (!only_surf) {
    num_errors += num_errors_dist + num_errors_dist_bvh;
    if (num_errors > 0) printf("CUDA: num_erros = %d / %d\n", num_errors, nrays);
    BREP_CUDA_CHECK(cudaFree(num_errors_d));
    BREP_CUDA_CHECK(cudaFree(num_better_safety_d));
    BREP_CUDA_CHECK(cudaFree(num_worse_safety_d));
    BREP_CUDA_CHECK(cudaFree(num_errors_dist_d));
    BREP_CUDA_CHECK(cudaFree(num_errors_dist_bvh_d));
  }
  BREP_CUDA_CHECK(cudaFree(points));
  BREP_CUDA_CHECK(cudaFree(dirs));
  BREP_CUDA_CHECK(cudaFree(points_NC));
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
  if (only_surf) return 0;
  return num_errors;
}
