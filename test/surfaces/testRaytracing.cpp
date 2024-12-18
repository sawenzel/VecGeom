#include "testRaytracing.h"
#include <iostream>
#include <string>

#include "test/benchmark/ArgParser.h"
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/management/GeoManager.h>
#ifdef VECGEOM_CUDA_INTERFACE
#include <VecGeom/management/CudaManager.h>
#endif
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/base/RNG.h>
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/LoopNavigator.h>
#include <VecGeom/volumes/utilities/VolumeUtilities.h>
#include <VecGeom/base/Stopwatch.h>

#ifdef VECGEOM_GDML
#include <Frontend.h>
#endif

#define SURF_NAV_DEBUG 0

#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/BrepHelper.h>
#include <VecGeom/surfaces/Navigator.h>
#include <VecGeom/surfaces/BVHSurfNavigator.h>

using namespace vecgeom;
// surface model is based on Real_t precision defined in testRaytracing.h
using BrepHelper = vgbrep::BrepHelper<Real_t>;
using SurfData   = vgbrep::SurfData<Real_t>;
using vecCore::math::Abs;
// generation of rays etc is based on vecgeom::Precision as these are used also in the solid model
using Vec3D  = vecgeom::Vector3D<vecgeom::Precision>;
using Vec3Dc = Precision[3];

//==================================================================================
int LoadGDML(const char *gdml_name, bool ongpu, int min_per_scene, double mmunit = 1)
{
#ifndef VECGEOM_GDML
  std::cout << "### VecGeom must be compiled with GDML support to run this.\n";
  return 1;
#else
  GeoManager::Instance().SetMinPerScene(min_per_scene);
  auto load = vgdml::Frontend::Load(gdml_name, false, mmunit);
  if (!load) return 2;
#endif

  auto world = GeoManager::Instance().GetWorld();
  if (!world) return 3;
  vecgeom::cxx::BVHManager::Init();
  return 0;
}

//==================================================================================
int LoadOnGPU(bool only_surf)
{
#ifdef VECGEOM_CUDA_INTERFACE
  std::cout << "synchronizing VecGeom geometry to GPU ...\n";
  auto world = GeoManager::Instance().GetWorld();
  if (!world) return 3;
  // Set higher stack limit to allow depper CSG for the solids model
  CudaAssertError(CudaDeviceSetStackLimit(8192));
  // set higher heap limit to allow solid model to dynamically allocate on GPU during init for large geometries
  CudaAssertError(CudaDeviceSetHeapLimit(512 * 1024 * 1024));
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();
  if (only_surf) {
    cudaManager.SynchronizeNavigationTable();
  } else {
    cudaManager.LoadGeometry(world);
    if (!cudaManager.Synchronize()) return 4;
    vecgeom::cxx::BVHManager::DeviceInit();
  }
#endif
  return 0;
}

//==================================================================================
void LocateSolids(int nrays, Vector3D<Precision> const *points, NavigationState *in_states)
{
  for (auto i = 0; i < nrays; ++i) {
    LoopNavigator::LocatePointIn(GeoManager::Instance().GetWorld(), points[i], in_states[i], true);
  }
}
//==================================================================================
void LocateSolidsBVH(int nrays, Vector3D<Precision> const *points, NavigationState *in_states)
{
  for (auto i = 0; i < nrays; ++i) {
    BVHNavigator::LocatePointIn(GeoManager::Instance().GetWorld(), points[i], in_states[i], true);
  }
}
//==================================================================================
void LocateSurf(int nrays, Vector3D<Precision> const *points, NavigationState *out_states)
{
  for (auto i = 0; i < nrays; ++i) {
    auto const &pos = points[i];
    // Locate with surface-based model
    vgbrep::protonav::LocatePointIn<Precision, Real_t>(NavigationState::WorldId(), pos, out_states[i], true);
  }
}
//==================================================================================
void LocateSurfBVH(int nrays, Vector3D<Precision> const *points, NavigationState *out_states)
{
  for (auto i = 0; i < nrays; ++i) {
    auto const &pos = points[i];
    // Locate with surface-based model
    vgbrep::protonav::BVHSurfNavigator<Real_t>::LocatePointIn(NavigationState::WorldId(), pos, out_states[i], true);
  }
}
//==================================================================================
int ValidateLocate(int nrays, Vector3D<Precision> const *points, NavigationState const *in_states,
                   NavigationState *out_states, bool validate_bvh, bool debug)
{
  int num_errors = 0;
  for (auto i = 0; i < nrays; ++i) {
    if (out_states[i].GetNavIndex() != in_states[i].GetNavIndex()) {
      num_errors++;
      if (debug && num_errors == 1) {
        printf("%d: p{%16.12f, %16.12f, %16.12f}\n", i, points[i][0], points[i][1], points[i][2]);
        // This just replays the failing locate query for debugging
        NavigationState out_state;
        LoopNavigator::LocatePointIn(GeoManager::Instance().GetWorld(), points[i], out_state, true);
        printf("   solid model state:        ");
        out_state.Print();
        out_state.Clear();
        if (validate_bvh) {
          vgbrep::protonav::BVHSurfNavigator<Real_t>::LocatePointIn(NavigationState::WorldId(), points[i], out_state,
                                                                    true);
          printf("   surface model BVH state:  ");
          out_state.Print();
        } else {
          vgbrep::protonav::LocatePointIn<Precision, Real_t>(NavigationState::WorldId(), points[i], out_state, true);
          printf("   surface model state:      ");
          out_state.Print();
        }
      }
    }
  }
  return num_errors;
}
//==================================================================================
void ComputeSafetiesSolid(int nrays, Vector3D<Precision> const *points, NavigationState const *in_states,
                          Precision *ref_safeties, bool validate_results)
{
  if (validate_results) {
    for (auto i = 0; i < nrays; ++i) {
      // Compute safety using the solid-based model
      ref_safeties[i] = LoopNavigator::ComputeSafety(points[i], in_states[i]);
    }
  } else {
    for (auto i = 0; i < nrays; ++i) {
      LoopNavigator::ComputeSafety(points[i], in_states[i]);
    }
  }
}
//==================================================================================
void ComputeSafetiesSolidBVH(int nrays, Vector3D<Precision> const *points, NavigationState const *in_states,
                             Precision *safeties, bool validate_results)
{
  if (validate_results) {
    for (auto i = 0; i < nrays; ++i) {
      // Compute safety using the solid-based model with BVH
      safeties[i] = BVHNavigator::ComputeSafety(points[i], in_states[i]);
    }
  } else {
    for (auto i = 0; i < nrays; ++i) {
      BVHNavigator::ComputeSafety(points[i], in_states[i]);
    }
  }
}
//==================================================================================
void ComputeSafetiesSurf(int nrays, Vector3D<Precision> const *points, NavigationState const *in_states,
                         Precision *safeties, bool validate_results)
{
  if (validate_results) {
    for (auto i = 0; i < nrays; ++i) {
      int exit_surf;
      safeties[i] = vgbrep::protonav::ComputeSafety<Precision, Real_t>(points[i], in_states[i], exit_surf);
    }
  } else {
    for (auto i = 0; i < nrays; ++i) {
      int exit_surf;
      vgbrep::protonav::ComputeSafety<Precision, Real_t>(points[i], in_states[i], exit_surf);
    }
  }
}
//==================================================================================
void ComputeSafetiesSurfBVH(int nrays, Vector3D<Precision> const *points, NavigationState const *in_states,
                            Precision *safeties, bool validate_results)
{
  if (validate_results) {
    for (auto i = 0; i < nrays; ++i) {
      safeties[i] = vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeSafety(points[i], in_states[i]);
    }
  } else {
    for (auto i = 0; i < nrays; ++i) {
      vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeSafety(points[i], in_states[i]);
    }
  }
}
//==================================================================================
bool CheckSafety(Vector3D<Precision> const &point, NavigationState const &in_state, double safety, int nsamples)
{
  // Generate nsamples random points in a sphere with the safety radius and check if
  // all of them are located in in_state
  auto &rng         = RNG::Instance();
  auto const navind = in_state.GetNavIndex();
  NavigationState new_state;
  bool is_safe = true;
  for (int i = 0; i < nsamples; ++i) {
    new_state.Clear();
    Vector3D<Precision> safepoint(point);
    double phi = rng.uniform(0, kTwoPi);
    double the = std::acos(2 * rng.uniform() - 1);
    Vector3D<Precision> ranpoint(std::sin(the) * std::cos(phi), std::sin(the) * std::sin(phi), std::cos(the));
    safepoint += safety * ranpoint; // safety is rounded from float
    LoopNavigator::LocatePointIn(GeoManager::Instance().GetWorld(), safepoint, new_state, true);
    is_safe = new_state.GetNavIndex() == navind;
    if (!is_safe) break;
  }
  return is_safe;
}
//==================================================================================
int ValidateSafety(int nrays, Vector3D<Precision> const *points, NavigationState const *in_states,
                   Precision const *safeties, Precision const *refSafeties, bool debug, int &num_better_safety,
                   int &num_worse_safety, Precision const tolerance)
{
  int num_errors   = 0;
  int num_warnings = 0;
  for (auto i = 0; i < nrays; ++i) {
    num_better_safety += (safeties[i] > refSafeties[i] + kToleranceBVH);
    num_worse_safety += (safeties[i] < refSafeties[i] - kToleranceBVH);
    if (safeties[i] / refSafeties[i] < tolerance) {
      VECGEOM_LOG(critical) << std::setprecision(16)
                            << "Safety of surface model below critical tolerance for point index " << i
                            << " with point " << points[i] << " safety Solid = " << refSafeties[i]
                            << " safety surf = " << safeties[i]
                            << " and ratio surf/solid = " << (safeties[i] / refSafeties[i]) << std::endl;
    }
    if (debug && num_warnings < 10 && safeties[i] < refSafeties[i] - kToleranceBVH) {
      num_warnings++;
      printf("point %d: (%.10g, %.10g, %.10g) safety Solid = %g  safety surf = %g ratio surf/solid = %g\n", i,
             points[i][0], points[i][1], points[i][2], refSafeties[i], safeties[i], (safeties[i] / refSafeties[i]));
      if (num_warnings == 10) printf("=== only first 10 warnings are shown\n");
      // Replay before exiting for debugging
      int exit_surf = 0;
      vgbrep::protonav::ComputeSafety<Precision, Real_t>(points[i], in_states[i], exit_surf);
    }
    if (debug && safeties[i] > refSafeties[i] + kToleranceBVH) {
      bool safesafe = CheckSafety(points[i], in_states[i], refSafeties[i], 1000);
      if (!safesafe && num_errors < 10) {
        num_errors++;
        printf("point %d: (%.10g, %.10g, %.10g) safety Solid = %g  safety surf = %g NOT SAFE\n", i, points[i][0],
               points[i][1], points[i][2], refSafeties[i], safeties[i]);
        if (num_errors == 10) printf("=== only first 10 errors are shown\n");
        // Replay before exiting for debugging
        int exit_surf = 0;
        vgbrep::protonav::ComputeSafety<Precision, Real_t>(points[i], in_states[i], exit_surf);
      }
    }
  }
  return num_errors;
}
//==================================================================================
template <typename Navigator>
void PropagateRaysSolid(int nrays, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                        NavigationState const *in_states, CrossingSeq *crossings, int idebug = -1,
                        int max_cross = vecgeom::kMaximumInt, bool validate_results = true)
{
  constexpr double kPushDistance = 1000 * vecgeom::kToleranceDist<Precision>;
  int ilast                      = nrays;
  int istart                     = 0;
  if (idebug >= 0) {
    std::cout << std::setprecision(16) << "PropagateRaysSolid debug ray " << idebug << " : p{" << points[idebug]
              << "} d{" << dirs[idebug] << "}\n   start :";
    in_states[idebug].Print();
    istart = idebug;
    ilast  = istart + 1;
  }
  for (auto i = istart; i < ilast; ++i) {
    NavigationState start_state = in_states[i];
    NavigationState out_state;
    auto const &dir = dirs[i];
    auto pt         = points[i];
    if (validate_results) crossings[i].Init(pt[0], pt[1], pt[2], dir[0], dir[1], dir[2]);
    int num_cross = 0;
    do {
      auto distance =
          Navigator::ComputeStepAndPropagatedState(pt, dir, kInfLength, start_state, out_state, kPushDistance);
      if (validate_results)
        num_cross = crossings[i].SetNextCrossing(distance, out_state);
      else
        num_cross++;
      if (idebug >= 0) {
        std::cout << std::setprecision(16) << "     dist = " << distance << "\n   " << num_cross << " : ";
        out_state.Print();
      }
      pt += distance * dir;
      start_state = out_state;
    } while (!out_state.IsOutside() && num_cross < max_cross);
  }
}
//==================================================================================
void PropagateRaysSurf(int nrays, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                       NavigationState const *in_states, CrossingSeq *crossings, int idebug = -1, int idebug_step = -1,
                       bool detect_overlaps = false, int max_cross = vecgeom::kMaximumInt, bool use_bvh = false,
                       bool validate_results = true)
{
  int ilast        = nrays;
  int istart       = 0;
  int num_overlaps = 0;
  int num_cross    = 0;
  if (idebug >= 0) {
    std::cout << "PropagateRaysSurf debug ray " << idebug << " using BVH?  " << use_bvh << "\n   start : ";
    in_states[idebug].Print();
    istart = idebug;
    ilast  = istart + 1;
  }
  for (auto i = istart; i < ilast; ++i) {
    NavigationState start_state = in_states[i];
    NavigationState out_state;
    vgbrep::CrossedSurface
        crossed_surf; // contains highest exiting frame information and final exiting or entering frame information
    // For general handling the crossed_surf.hit_surface_data to be used, only for the relocation in the overlap
    // detection only the highest exiting infromation crossed_surf.exit_surface_data is used
    auto pt         = points[i];
    auto const &dir = dirs[i];
    if (validate_results) crossings[i].Init(pt[0], pt[1], pt[2], dir[0], dir[1], dir[2]);
    do {
      crossed_surf.Set(0, 0, 0); // need to reset because the same inner tube surface can be crossed twice in a row

      if (idebug >= 0 && num_cross == idebug_step) {
        std::cout << "Debugging step " << idebug_step << " starting from state:\n";
        start_state.Print();
      }
      Precision distance{0};
      if (!use_bvh) {
        distance =
            vgbrep::protonav::ComputeStepAndHit<Precision, Real_t>(pt, dir, start_state, out_state, crossed_surf);
      } else {
        distance = vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeStepAndHit(pt, dir, start_state, out_state,
                                                                                 crossed_surf);
      }

      // get framed surface data to check surface for overlapping
      auto const &surfdata = BrepHelper::Instance().GetSurfData();
      if (detect_overlaps && !surfdata.IsFSOverlapping(crossed_surf.hit_surf) && out_state.GetState() != 0) {
        NavigationState true_state;
        if (crossed_surf.hit_surf.GetFSindex() != -1) {
          true_state = out_state;
          vgbrep::protonav::ReLocatePointIn<Precision, Real_t>(start_state, pt + distance * dir, dir, true_state,
                                                               crossed_surf.exit_surf, distance);
        }

        // overlap if the output state and the true state disagree (and the outstate is not outside) or if extruding
        // overlap, which is handled separately
        if ((out_state.GetState() != true_state.GetState())) { // || crossed_surf.GetFSindex() == -1) {
          num_overlaps++;

          // fixme: if common_id = -1, then these accessors below are garbage. Recheck if this is needed or even
          // harmful.
          // cannot use getter since the framedsurf must not be const here
          auto const &surf        = surfdata.fCommonSurfaces[crossed_surf.hit_surf.GetCSindex()];
          auto const &exit_side   = crossed_surf.hit_surf.IsLeftSide() ? surf.fLeftSide : surf.fRightSide;
          auto &framedsurf        = exit_side.GetSurface(crossed_surf.hit_surf.GetFSindex(), surfdata);
          int surf_index          = framedsurf.fSurfIndex;
          framedsurf.fOverlapping = true;

          VECGEOM_LOG(warning) << std::setprecision(16) << num_overlaps << " overlap detected for ray " << i
                               << " at num_cross = " << num_cross << "\n   starting point " << points[i]
                               << " and direction " << dirs[i]
                               << "\n   Overlapping surface:  " << crossed_surf.hit_surf.GetCSindex() << " side "
                               << crossed_surf.hit_surf.IsLeftSide() << " frameid "
                               << crossed_surf.hit_surf.GetFSindex() << " local problem side: " << surf_index;
          std::cout << " out_state.Print() " << std::endl;
          out_state.Print();
          std::cout << " true_state.Print() " << std::endl;
          true_state.Print();
          // set out state to true state after printing
          out_state = true_state;
        }
      }
      // exiting framed surface marked as overlapping, need to relocate
      if (surfdata.IsFSOverlapping(crossed_surf.hit_surf) && crossed_surf.hit_surf.GetFSindex() != -1) {
        vgbrep::protonav::ReLocatePointIn<Precision, Real_t>(start_state, pt + distance * dir, dir, out_state,
                                                             crossed_surf.exit_surf, distance);
      }
      if (crossed_surf.hit_surf.GetFSindex() == -1) {
        // Most likely extruding overlap detected, relocating to correct state

        if (idebug >= 0) {
          VECGEOM_LOG(warning) << std::setprecision(16) << "No exiting surface for ray " << i
                               << " at num_cross = " << num_cross << "\n   starting point " << points[i]
                               << " and direction " << dirs[i] << "\n   state for failing step : ";
          start_state.Print();
        }

        // Find true location for the crossing point
        NavigationState true_state;
        // note that here we use the previous point pt and not pt + distance * dir because the distance is inf!
        vgbrep::protonav::ReLocatePointIn<Precision, Real_t>(start_state, pt, dir, true_state, crossed_surf.exit_surf,
                                                             /* distance=*/Precision(0.));
        if (idebug >= 0) {
          std::cout << "   crossing point : " << pt << " was located in : ";
          true_state.Print();
        }
        // Now replay to get correct distance
        if (!use_bvh) {
          distance =
              vgbrep::protonav::ComputeStepAndHit<Precision, Real_t>(pt, dir, true_state, out_state, crossed_surf);
        } else {
          distance = vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeStepAndHit(pt, dir, true_state, out_state,
                                                                                   crossed_surf);
        }
        // if corrected distance is still incorrect, abort
        if (distance == 0 || distance == vecgeom::InfinityLength<Precision>()) {
          VECGEOM_LOG(critical) << std::setprecision(16) << "After relocation, still no exiting surface for ray " << i
                                << " at num_cross = " << num_cross << "\n Terminating raytracing!";
          return;
        }
      }
      if (validate_results)
        num_cross = crossings[i].SetNextCrossing(distance, out_state);
      else
        num_cross++;
      if (idebug >= 0) {
        std::cout << std::setprecision(16) << "     dist = " << distance
                  << "  surf = " << crossed_surf.hit_surf.GetCSindex()
                  << "  frame = " << crossed_surf.hit_surf.GetFSindex()
                  << "  LeftSide: " << crossed_surf.hit_surf.IsLeftSide() << "\n   " << num_cross << " : ";
        out_state.Print();
      }
      pt += distance * dir;
      start_state = out_state;
    } while (!out_state.IsOutside() && num_cross < max_cross + 1);
  }
}
//==================================================================================
int ValidateCrossing(int nrays, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                     NavigationState const *in_states, CrossingSeq *ref_crossings, CrossingSeq *crossings, bool debug,
                     bool accept_zeros = false, int max_cross = vecgeom::kMaximumInt, bool use_bvh = false)
{
  int num_errors_dist = 0;
  int istep_err       = 0;
  int istep_err_solid = 0;
  for (auto i = 0; i < nrays; ++i) {
    bool error_dist = !crossings[i].IsEqual(ref_crossings[i], istep_err, istep_err_solid, accept_zeros);
    if (error_dist && !debug && num_errors_dist < 10) {
      std::cout << std::setprecision(16) << "=== error for ray " << i << " at step " << istep_err << "/"
                << ref_crossings[i].GetNsteps() << ": p{" << points[i] << "} d{" << dirs[i] << "}\n";
      if (num_errors_dist == 9) std::cout << "=== Only first 10 errors are shown\n";
    }
    num_errors_dist += error_dist;
    if (debug && error_dist && (num_errors_dist == 1)) {
      // replay first error
      printf("\033[1;31m=== ray %d has a propagation difference at step %d (correponding solid model step %d) dist_ref "
             "= %.10g :  dist = %.10g\033[0m\n",
             i, istep_err, istep_err_solid, ref_crossings[i].fSteps[istep_err_solid], crossings[i].fSteps[istep_err]);
      if (crossings[i].fStates[istep_err].GetState() != ref_crossings[i].fStates[istep_err_solid].GetState()) {
        printf("\033[1;32msolid model state after step:\033[0m\n");
        ref_crossings[i].fStates[istep_err_solid].Print();
        printf("\033[1;31msurface model using the %s state after step:\033[0m\n", use_bvh ? "BVH" : "looper");
        crossings[i].fStates[istep_err].Print();
      }
      printf("Replaying ray for debugging : \n\n");
      PropagateRaysSolid<LoopNavigator>(nrays, points, dirs, in_states, ref_crossings, i, max_cross);
      PropagateRaysSurf(nrays, points, dirs, in_states, crossings, i, istep_err, /*detect_overlaps =*/false, max_cross,
                        use_bvh);
    }
  }
  return num_errors_dist;
}
//==================================================================================
int testRaytracingHost(int nrays, Vector3D<Precision> *points, Vector3D<Precision> *dirs, bool debug,
                       Precision safety_tolerance, bool detect_overlaps = false, bool accept_zeros = false,
                       int max_cross = vecgeom::kMaximumInt, bool test_bvh = false, bool validate_results = true)
{
  // allocate storage
  NavigationState *origStates      = new NavigationState[nrays];
  NavigationState *outputStates    = new NavigationState[nrays];
  NavigationState *outputStatesBVH = new NavigationState[nrays];

  Precision *ref_safeties{nullptr}, *safeties{nullptr}, *bvh_safeties{nullptr};
  CrossingSeq *ref_crossings{nullptr}, *crossings{nullptr}, *bvh_crossings{nullptr};
  int num_errors          = 0;
  int num_errors_loc_bvh  = 0;
  int num_errors_safe     = 0;
  int num_errors_safe_bvh = 0;
  int num_errors_dist     = 0;
  int num_errors_dist_bvh = 0;
  int num_better_safety   = 0;
  int num_worse_safety    = 0;

  // We may want to disable all results validation for benchmarking
  if (validate_results) {
    ref_safeties = new Precision[nrays];
    memset(ref_safeties, 0, sizeof(Precision) * nrays);

    safeties = new Precision[nrays];
    memset(safeties, 0, sizeof(Precision) * nrays);

    bvh_safeties = new Precision[nrays];
    memset(bvh_safeties, 0, sizeof(Precision) * nrays);

    ref_crossings = new CrossingSeq[nrays];
    crossings     = new CrossingSeq[nrays];
    bvh_crossings = new CrossingSeq[nrays];
  }

  int idebug = (debug && nrays == 1) ? 0 : -1;

  Stopwatch timer;
  // Locating the global points
  timer.Start();
  LocateSolids(nrays, points, origStates);
  auto time_locate_solids = timer.Stop();

  timer.Start();
  LocateSolidsBVH(nrays, points, outputStates);
  auto time_locate_solids_bvh = timer.Stop();

  for (auto i = 0; i < nrays; ++i)
    outputStates[i].Clear();

  timer.Start();
  LocateSurf(nrays, points, outputStates);
  auto time_locate_surf = timer.Stop();

  for (auto i = 0; i < nrays; ++i)
    outputStatesBVH[i].Clear();

  timer.Start();
  if (test_bvh) LocateSurfBVH(nrays, points, outputStatesBVH);
  auto time_locate_surf_bvh = timer.Stop();

  // Correctness for locating points
  if (validate_results) {
    num_errors = ValidateLocate(nrays, points, origStates, outputStates, false, debug);
    if (num_errors > 0) {
      std::cout << "*** HOST: Point locate errors: " << num_errors << "\n";
      if (debug) return num_errors;
    }
  }

  if (test_bvh) {
    // Validate BVH Locate
    if (validate_results) {
      num_errors_loc_bvh = ValidateLocate(nrays, points, origStates, outputStatesBVH, true, debug);
      if (num_errors_loc_bvh > 0) {
        std::cout << "*** HOST: BVH point locate errors: " << num_errors_loc_bvh << "\n";
        if (debug) return num_errors_loc_bvh;
      }
    }
  }

  if (!debug) {
    std::cout << "HOST: locate_solids: " << time_locate_solids << "  locate_solids_BVH: " << time_locate_solids_bvh
              << "  locate_surf: " << time_locate_surf;
    if (test_bvh) {
      std::cout << " locate_surf_BVH: " << time_locate_surf_bvh << "\n";
    } else {
      std::cout << "\n";
    }
  }

  // Safety for solids model (reference)
  timer.Start();
  ComputeSafetiesSolid(nrays, points, origStates, ref_safeties, validate_results);
  auto time_safety_solids = timer.Stop();

  // Safety for solids model with BVH
  timer.Start();
  ComputeSafetiesSolidBVH(nrays, points, origStates, safeties, validate_results);
  auto time_safety_solids_bvh = timer.Stop();

  // Safety for surface model
  timer.Start();
  ComputeSafetiesSurf(nrays, points, origStates, safeties, validate_results);
  auto time_safety_surf = timer.Stop();

  // Safety for surface model with BVH
  timer.Start();
  if (test_bvh) ComputeSafetiesSurfBVH(nrays, points, origStates, bvh_safeties, validate_results);
  auto time_safety_surf_bvh = timer.Stop();

  // Correctness for safety
  if (validate_results) {
    num_errors_safe = ValidateSafety(nrays, points, origStates, safeties, ref_safeties, debug, num_better_safety,
                                     num_worse_safety, safety_tolerance);

    num_errors += num_errors_safe;

    if (num_errors_safe > 0) std::cout << "*** HOST: Safety errors: " << num_errors_safe << "\n";
    if (num_better_safety > 0) printf("HOST:    number of better safety values: %d\n", num_better_safety);
    if (num_worse_safety > 0) printf("HOST:    number of worse safety values: %d\n", num_worse_safety);

    if (test_bvh) {
      num_better_safety   = 0;
      num_worse_safety    = 0;
      num_errors_safe_bvh = ValidateSafety(nrays, points, origStates, bvh_safeties, ref_safeties, debug,
                                           num_better_safety, num_worse_safety, safety_tolerance);
      num_errors += num_errors_safe_bvh;
      if (num_errors_safe_bvh > 0) std::cout << "*** HOST: BVH safety errors: " << num_errors_safe_bvh << "\n";
      if (num_better_safety > 0) printf("HOST:    BVH number of better safety values: %d\n", num_better_safety);
      if (num_worse_safety > 0) printf("HOST:    BVH number of worse safety values: %d\n", num_worse_safety);
    }
  }

  // Report timing
  if (!debug) {
    std::cout << "HOST: safety_solids: " << time_safety_solids << "  safety_solids_BVH: " << time_safety_solids_bvh
              << "  safety_surf: " << time_safety_surf;
    if (test_bvh) {
      std::cout << "  safety_surf_bvh: " << time_safety_surf_bvh << "\n";
    } else {
      std::cout << "\n";
    }
  }

  // Distance computation + relocation for solid model
  timer.Start();
  PropagateRaysSolid<LoopNavigator>(nrays, points, dirs, origStates, ref_crossings, idebug, max_cross,
                                    validate_results);
  auto time_traverse_solids = timer.Stop();

  // Distance computation + relocation for solid model + BVH
  timer.Start();
  PropagateRaysSolid<BVHNavigator>(nrays, points, dirs, origStates, crossings, idebug, max_cross, validate_results);
  auto time_traverse_solids_bvh = timer.Stop();

  // Distance computation + relocation for surface model
  timer.Start();
  PropagateRaysSurf(nrays, points, dirs, origStates, crossings, idebug, /*idebug_step=*/-1, detect_overlaps, max_cross,
                    false, validate_results);
  auto time_traverse_surf = timer.Stop();

  // Distance computation + relocation for surface model + BVH
  timer.Start();
  if (test_bvh)
    PropagateRaysSurf(nrays, points, dirs, origStates, bvh_crossings, idebug, /*idebug_step=*/-1,
                      /*detect_overlaps=*/false, max_cross, true, validate_results);
  auto time_traverse_surf_bvh = timer.Stop();

  // Correctness for traversal
  if (validate_results) {
    num_errors_dist =
        ValidateCrossing(nrays, points, dirs, origStates, ref_crossings, crossings, debug, accept_zeros, max_cross);
    if (test_bvh)
      num_errors_dist_bvh = ValidateCrossing(nrays, points, dirs, origStates, ref_crossings, bvh_crossings, debug,
                                             accept_zeros, max_cross, /*use_bvh=*/true);
    num_errors += num_errors_dist;
    if (num_errors_dist > 0) std::cout << "*** HOST: traverse errors surf: " << num_errors_dist << "\n";
    if (test_bvh)
      if (num_errors_dist_bvh > 0) std::cout << "*** HOST: traverse errors surf BVH: " << num_errors_dist_bvh << "\n";
    if (num_errors > 0) printf("HOST: num_erros = %d / %d\n", num_errors, nrays);
  }

  if (!debug) {
    std::cout << "HOST: traverse_solids: " << time_traverse_solids
              << "  traverse_solids_BVH: " << time_traverse_solids_bvh << "  traverse_surf: " << time_traverse_surf;
    if (test_bvh) std::cout << "  traverse_surf BVH: " << time_traverse_surf_bvh;
    std::cout << std::endl;
  }

  if (validate_results) {
    delete[] ref_safeties;
    delete[] safeties;
    delete[] ref_crossings;
    delete[] crossings;
  }
  delete[] origStates;
  delete[] outputStates;

  if (validate_results) return num_errors;
  return 0;
}

// in testRaytracing.cu
int testRaytracingCUDA(int nrays, Vec3Dc const *points, Vec3Dc const *dirs, const SurfData &surfdata, bool debug,
                       bool accept_zeros = 0, int max_cross = vecgeom::kMaximumInt, bool test_bvh = false,
                       bool validate_results = true, bool only_surf = false, bool bvh_single_step = false,
                       bool bvh_split_step = false, int verbosity = 0);

//==================================================================================
int main(int argc, char *argv[])
{
  OPTION_STRING(gdml_name, "default.gdml");
  OPTION_INT(nrays, 10000);
  OPTION_INT(debug, 0);
  OPTION_INT(verbosity, 0);
  OPTION_INT(min_per_scene, 1000);
  OPTION_INT(ongpu, 1);
  OPTION_INT(max_cross, vecgeom::kMaximumInt);
  OPTION_BOOL(detect_overlaps, false);
  OPTION_BOOL(accept_zeros, false);
  OPTION_BOOL(test_bvh, false);
  OPTION_BOOL(bvh_single_step, false);
  OPTION_BOOL(bvh_split_step, false);
  OPTION_BOOL(validate_results, true);
  OPTION_BOOL(only_surf, true);
  OPTION_BOOL(use_TB_gun, false);
  OPTION_DOUBLE(mmunit, 1);
  OPTION_DOUBLE(safety_ratio, 0);
  std::vector<double> default_point = {vecgeom::InfinityLength<Precision>(), vecgeom::InfinityLength<Precision>(),
                                       vecgeom::InfinityLength<Precision>()};
  OPTION_VECTOR(point, default_point);
  std::vector<double> default_direction = {0., 0., 1.};
  OPTION_VECTOR(direction, default_direction);
  OPTION_VECTOR(max_world, default_point);
  std::vector<double> default_min_world = {-vecgeom::InfinityLength<Precision>(), -vecgeom::InfinityLength<Precision>(),
                                           -vecgeom::InfinityLength<Precision>()};
  OPTION_VECTOR(min_world, default_min_world);
  assert(point.size() == 3 && direction.size() == 3);
  assert(min_world.size() == 3 && default_min_world.size() == 3);

  // transform to Vec3D for further handling
  Vec3D point_3D     = {point[0], point[1], point[2]};
  Vec3D direction_3D = {direction[0], direction[1], direction[2]};
  Vec3D min_world_3d = {min_world[0], min_world[1], min_world[2]};
  Vec3D max_world_3d = {max_world[0], max_world[1], max_world[2]};
  if (direction_3D.Mag2() > 0) direction_3D.Normalize();

  bool use_provided_point = point_3D.Mag2() < vecgeom::InfinityLength<Precision>();
  if (use_provided_point) {
    // check if direction is normalized
    assert(direction_3D.IsNormalized());
    nrays = 1;
    if (debug)
      std::cout << "Tracking single ray with point " << point_3D << " and direction " << direction_3D << std::endl;
  }

  vecgeom::logger().level(vecgeom::LogLevel::info);
  Stopwatch timer;
  // Load the geometry
  timer.Start();
  bool load = LoadGDML(gdml_name.c_str(), ongpu, min_per_scene, mmunit);
  if (load > 0) return load;
  auto time_load = timer.Stop();
  std::cout << "Geometry loading: " << time_load << " [s]\n";

  BrepHelper::Instance().SetVerbosity(verbosity);

  timer.Start();
  // Conversion to the surface model
  if (!BrepHelper::Instance().Convert()) return 1;
  auto time_surf_dist = timer.Stop();
  BrepHelper::Instance().PrintSurfData();
  std::cout << "Conversion time to surface model: " << time_surf_dist << " [s]\n";

  // Generate random points and directions inside the setup
  Vec3D *points, *dirs;
  points = new Vec3D[nrays];
  dirs   = new Vec3D[nrays];
  Vec3D amin, amax;
  auto world = GeoManager::Instance().GetWorld();
  world->GetLogicalVolume()->GetUnplacedVolume()->Extent(amin, amax);
  for (int i = 0; i < 3; i++) {
    amin[i] = std::max(amin[i], min_world_3d[i]);
    amax[i] = std::min(amax[i], max_world_3d[i]);
  }

  if (use_TB_gun) {
    amin.Set(0., 0., -700.);
    amax.Set(0., 0., -700.);
  }

  Vec3D origin{0, 0, 0};
  if (!use_provided_point) {
    volumeUtilities::FillRandomPoints(amin, amax, points, nrays);
    if (!use_TB_gun) {
      volumeUtilities::FillRandomDirections(dirs, nrays);
    } else {
      // roughly the angles for the testbeam setup
      Precision aMaxPhi   = 0;
      Precision aMinPhi   = vecgeom::kTwoPi;
      Precision aMaxTheta = 0;
      Precision aMinTheta = atan2(20, 3200);
      for (int i = 0; i < nrays; i++) {
        Precision phi   = (aMaxPhi - aMinPhi) * RNG::Instance().uniform() + aMinPhi;
        Precision theta = acos((cos(aMaxTheta) - cos(aMinTheta)) * RNG::Instance().uniform() + cos(aMinTheta));
        dirs[i].Set(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
      }
    }
  } else {
    points[0] = point_3D;
    dirs[0]   = direction_3D;
  }

  // UGLY: Use a points struct to avoid passing Vector3D to the cuda namespace
  auto pointsc = new Vec3Dc[nrays];
  auto dirsc   = new Vec3Dc[nrays];
  for (auto i = 0; i < nrays; ++i) {
    auto &pt  = pointsc[i];
    auto &dir = dirsc[i];
    for (auto j = 0; j < 3; ++j) {
      pt[j]  = points[i][j]; // static_cast<Real_t>(points[i][j]);
      dir[j] = dirs[i][j];   // static_cast<Real_t>(dirs[i][j]);
    }
  }

  int errHost = testRaytracingHost(nrays, points, dirs, debug, safety_ratio, detect_overlaps, accept_zeros, max_cross,
                                   test_bvh, validate_results);
  if (debug && errHost > 0) return errHost;
  int errCUDA = 0;
#ifdef VECGEOM_CUDA_INTERFACE
  // Copy geometry to GPU
  auto const &surfdata = BrepHelper::Instance().GetSurfData();
  if (ongpu) {
    timer.Start();
    errCUDA            = LoadOnGPU(only_surf);
    auto time_transfer = timer.Stop();
    if (only_surf)
      std::cout << "Navigation table transferred to GPU : " << time_transfer << " [s]\n";
    else
      std::cout << "Solid model transferred to GPU : " << time_transfer << " [s]\n";
    if (!errCUDA)
      errCUDA = testRaytracingCUDA(nrays, pointsc, dirsc, surfdata, debug, accept_zeros, max_cross, test_bvh,
                                   validate_results, only_surf, bvh_single_step, bvh_split_step, verbosity);
  }
#endif

  // Clear surface data
  BrepHelper::Instance().ClearData();
  delete[] points;
  delete[] dirs;
  return errHost + errCUDA;
}
