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
  VECGEOM_DEVICE_API_CALL(DeviceSetLimit(VECGEOM_DEVICE_API_SYMBOL(LimitStackSize), 8192));
  // set higher heap limit to allow solid model to dynamically allocate on GPU during init for large geometries
  VECGEOM_DEVICE_API_CALL(DeviceSetLimit(VECGEOM_DEVICE_API_SYMBOL(LimitMallocHeapSize), 512 * 1024 * 1024));
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

/// @brief Function computing the normal on the global point traversing a surface
/// @tparam Real_t Precision type
/// @param pos Global position
/// @param prevState State before crossing
/// @param nextState State after crossing
/// @param valid Returned validity of the returned normal
/// @param exiting Returned exiting
/// @return Normalized normal on the given point
template <typename Real_t>
Vector3D<Real_t> ComputeNormal(Vector3D<Real_t> const &pos, NavigationState const &prevState,
                               NavigationState const &nextState, bool &valid, bool &exiting)
{
  Vector3D<Real_t> local_normal;
  Transformation3D trans;
  exiting           = prevState.IsDescendent(nextState.GetState());
  auto const &state = exiting ? prevState : nextState;
  state.TopMatrix(trans);
  valid              = state.Top()->GetUnplacedVolume()->Normal(/*localpoint=*/trans.Transform(pos), local_normal);
  auto global_normal = trans.InverseTransformDirection(local_normal);
  return global_normal;
}

//==================================================================================
void LocateSolids(Vector3D<Precision> const *points, Vector3D<Precision> const *directions, NavigationState *in_states,
                  TestConfig const &config)
{
  const char *matching[3] = {"does not match", "matches input", "matches next"};
  const char *svalid[2]   = {"invalid", "valid"};
  const char *sexiting[2] = {"entering", "exiting"};
  if (config.input_state > 0) {
    // User state defined
    in_states[0] = NavigationState(config.input_state);
    in_states[0].SetBoundaryState(config.on_boundary);
    NavigationState state_located = in_states[0];
    state_located.Pop();
    // Now in the parent of the user state
    Transformation3D trans;
    state_located.TopMatrix(trans);
    // Locate starting from the parent, validating the user state
    auto vol = LoopNavigator::LocatePointIn(in_states[0].Top(), trans.Transform(points[0]), state_located, true);
    if (!vol) VECGEOM_LOG(info) << "Provided input state top volume does NOT contain the point";
    int match = (vol != nullptr) && (state_located.GetNavIndex() == config.input_state);
    VECGEOM_LOG(info) << "Provided input state: " << in_states[0].GetNavIndex();
    VECGEOM_LOG(info) << "Located  input state: " << state_located.GetNavIndex() << " " << matching[match];
    in_states[0].Top()->GetUnplacedVolume()->Print();
    if (config.compute_normal && config.next_state > 0 && config.on_boundary) {
      NavigationState state_next(config.input_state);
      Vector3D<Precision> normal;
      bool valid_normal = true;
      bool exiting      = false;
      normal            = ComputeNormal(points[0], in_states[0], state_next, valid_normal, exiting);
      std::cout << "| " << sexiting[exiting] << "| " << svalid[valid_normal] << " normal ";
      if (valid_normal) std::cout << "| n.dot.dir = " << normal.Dot(directions[0]);
      printf("\n");
    }
    return;
  }

  for (auto i = 0; i < config.nrays; ++i) {
    LoopNavigator::LocatePointIn(GeoManager::Instance().GetWorld(), points[i], in_states[i], true);
  }
}
//==================================================================================
void LocateSolidsBVH(Vector3D<Precision> const *points, NavigationState *in_states, TestConfig const &config)
{
  for (auto i = 0; i < config.nrays; ++i) {
    BVHNavigator::LocatePointIn(GeoManager::Instance().GetWorld(), points[i], in_states[i], true);
  }
}
//==================================================================================
void LocateSurf(Vector3D<Precision> const *points, NavigationState *out_states, TestConfig const &config)
{
  for (auto i = 0; i < config.nrays; ++i) {
    auto const &pos = points[i];
    // Locate with surface-based model
    vgbrep::protonav::LocatePointIn<Precision, Real_t>(NavigationState::WorldId(), pos, out_states[i], true);
  }
}
//==================================================================================
void LocateSurfBVH(Vector3D<Precision> const *points, NavigationState *out_states, TestConfig const &config)
{
  for (auto i = 0; i < config.nrays; ++i) {
    auto const &pos = points[i];
    // Locate with surface-based model
    vgbrep::protonav::BVHSurfNavigator<Real_t>::LocatePointIn(NavigationState::WorldId(), pos, out_states[i], true);
  }
}
//==================================================================================
int ValidateLocate(Vector3D<Precision> const *points, NavigationState const *in_states, NavigationState *out_states,
                   bool validate_bvh, TestConfig const &config)
{
  int num_errors = 0;
  for (auto i = 0; i < config.nrays; ++i) {
    if (out_states[i].GetNavIndex() != in_states[i].GetNavIndex()) {
      num_errors++;
      if (config.debug && num_errors == 1) {
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
void ComputeSafetiesSolid(Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                          NavigationState const *in_states, Precision *ref_safeties, TestConfig const &config)
{
  const char *status[2] = {"entering", "exiting"};
  for (auto i = 0; i < config.nrays; ++i) {
    double safety = LoopNavigator::ComputeSafety(points[i], in_states[i]);
    if (config.validate_results) ref_safeties[i] = safety;
    if (config.input_state > 0) {
      VECGEOM_LOG(info) << "Safety = " << safety;
    }
  }
}
//==================================================================================
void ComputeSafetiesSolidBVH(Vector3D<Precision> const *points, NavigationState const *in_states, Precision *safeties,
                             TestConfig const &config)
{
  for (auto i = 0; i < config.nrays; ++i) {
    double safety = BVHNavigator::ComputeSafety(points[i], in_states[i]);
    if (config.validate_results) safeties[i] = safety;
  }
}
//==================================================================================
void ComputeSafetiesSurf(Vector3D<Precision> const *points, NavigationState const *in_states, Precision *safeties,
                         TestConfig const &config)
{
  int exit_surf;
  for (auto i = 0; i < config.nrays; ++i) {
    double safety = vgbrep::protonav::ComputeSafety<Precision, Real_t>(points[i], in_states[i], exit_surf);
    if (config.validate_results) safeties[i] = safety;
  }
}
//==================================================================================
void ComputeSafetiesSurfBVH(Vector3D<Precision> const *points, NavigationState const *in_states, Precision *safeties,
                            TestConfig const &config)
{
  for (auto i = 0; i < config.nrays; ++i) {
    double safety = vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeSafety(points[i], in_states[i]);
    if (config.validate_results) safeties[i] = safety;
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
int ValidateSafety(Vector3D<Precision> const *points, NavigationState const *in_states, Precision const *safeties,
                   Precision const *refSafeties, int &num_better_safety, int &num_worse_safety,
                   TestConfig const &config)
{
  int num_errors   = 0;
  int num_warnings = 0;
  for (auto i = 0; i < config.nrays; ++i) {
    num_better_safety += (safeties[i] > refSafeties[i] + kToleranceBVH);
    num_worse_safety += (safeties[i] < refSafeties[i] - kToleranceBVH);
    if (safeties[i] / refSafeties[i] < config.safety_ratio) {
      VECGEOM_LOG(critical) << std::setprecision(16)
                            << "Safety of surface model below critical tolerance for point index " << i
                            << " with point " << points[i] << " safety Solid = " << refSafeties[i]
                            << " safety surf = " << safeties[i]
                            << " and ratio surf/solid = " << (safeties[i] / refSafeties[i]) << std::endl;
    }
    if (config.debug && num_warnings < 10 && safeties[i] < refSafeties[i] - kToleranceBVH) {
      num_warnings++;
      printf("point %d: (%.10g, %.10g, %.10g) safety Solid = %g  safety surf = %g ratio surf/solid = %g\n", i,
             points[i][0], points[i][1], points[i][2], refSafeties[i], safeties[i], (safeties[i] / refSafeties[i]));
      if (num_warnings == 10) printf("=== only first 10 warnings are shown\n");
      // Replay before exiting for debugging
      int exit_surf = 0;
      vgbrep::protonav::ComputeSafety<Precision, Real_t>(points[i], in_states[i], exit_surf);
    }
    if (config.debug && safeties[i] > refSafeties[i] + kToleranceBVH) {
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
void PropagateRaysSolid(Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                        NavigationState const *in_states, CrossingSeq *crossings, int idebug, int idebug_step,
                        TestConfig const &config)
{
  constexpr double kPushDistance = 1000 * vecgeom::kToleranceDist<Precision>;
  const char *svalid[2]          = {"invalid", "valid"};
  const char *sexiting[2]        = {"entering", "exiting"};
  int ilast                      = config.nrays;
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
    if (config.validate_results) crossings[i].Init(pt[0], pt[1], pt[2], dir[0], dir[1], dir[2]);
    int num_cross = 0;
    do {
      if (idebug >= 0 && num_cross == idebug_step) {
        std::cout << "Debugging step " << idebug_step << " for solid model starting from state:\n";
        start_state.Print();
      }
      auto step_limit = kInfLength;
      if (config.input_state > 0 && num_cross == 0) step_limit = config.step_limit;
      // Use the ComputeStep + Relocate interface as in the MC
      auto distance = Navigator::ComputeStepAndNextVolume(pt, dir, step_limit, start_state, out_state, kPushDistance);
      bool crossed  = out_state.IsOnBoundary();
      bool same_vol = true;
      if (crossed)
        Navigator::RelocateToNextVolume(pt + distance * dir, dir, out_state);
      else {
        vecgeom::Transformation3D trans;
        start_state.TopMatrix(trans);
        same_vol = start_state.Top()->GetUnplacedVolume()->Inside(trans.Transform(pt)) != vecgeom::kOutside;
      }
      if (config.validate_results)
        num_cross = crossings[i].SetNextCrossing(distance, out_state);
      else
        num_cross++;
      if (idebug >= 0 && (idebug_step < 0 || num_cross == idebug_step + 1)) {
        std::cout << std::setprecision(16) << "     dist = " << distance;
        if (!crossed) {
          std::cout << " | limited step ";
          if (same_vol)
            std::cout << "| same volume ";
          else
            std::cout << "| different volume !!! ";
        } else {
          if (config.compute_normal) {
            Vector3D<Precision> normal;
            bool valid_normal = true;
            bool exiting      = false;
            normal            = ComputeNormal(pt + distance * dir, start_state, out_state, valid_normal, exiting);
            std::cout << "| " << sexiting[exiting] << " | " << svalid[valid_normal] << " normal ";
            if (valid_normal) std::cout << "| n.dot.dir = " << normal.Dot(dir);
          }
        }
        std::cout << "\n   " << num_cross << " : ";

        out_state.Print();
      }
      pt += distance * dir;
      start_state = out_state;
    } while (!out_state.IsOutside() && num_cross < config.max_cross);
  }
}
//==================================================================================
void PropagateRaysSurf(Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                       NavigationState const *in_states, CrossingSeq *crossings, int idebug, int idebug_step,
                       bool detect_overlaps, bool use_bvh, TestConfig const &config)
{
  int ilast        = config.nrays;
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
    if (config.validate_results) crossings[i].Init(pt[0], pt[1], pt[2], dir[0], dir[1], dir[2]);
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
      if (config.validate_results)
        num_cross = crossings[i].SetNextCrossing(distance, out_state);
      else
        num_cross++;
      if (idebug >= 0 && (idebug_step < 0 || num_cross == idebug_step + 1)) {
        std::cout << std::setprecision(16) << "     dist = " << distance
                  << "  surf = " << crossed_surf.hit_surf.GetCSindex()
                  << "  frame = " << crossed_surf.hit_surf.GetFSindex()
                  << "  LeftSide: " << crossed_surf.hit_surf.IsLeftSide() << "\n   " << num_cross << " : ";
        out_state.Print();
      }
      pt += distance * dir;
      start_state = out_state;
    } while (!out_state.IsOutside() && num_cross < config.max_cross + 1);
  }
}
//==================================================================================
int ValidateCrossing(Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                     NavigationState const *in_states, CrossingSeq *ref_crossings, CrossingSeq *crossings, bool use_bvh,
                     TestConfig const &config)
{
  int num_errors_dist = 0;
  int istep_err       = 0;
  int istep_err_solid = 0;
  for (auto i = 0; i < config.nrays; ++i) {
    bool error_dist = !crossings[i].IsEqual(ref_crossings[i], istep_err, istep_err_solid, config.accept_zeros);
    if (error_dist && !config.debug && num_errors_dist < 10) {
      std::cout << std::setprecision(16) << "=== error for ray " << i << " at step " << istep_err << "/"
                << ref_crossings[i].GetNsteps() << ": p{" << points[i] << "} d{" << dirs[i] << "}\n";
      if (num_errors_dist == 9) std::cout << "=== Only first 10 errors are shown\n";
    }
    num_errors_dist += error_dist;
    if (config.debug && error_dist && (num_errors_dist == 1)) {
      // replay first error
      printf("\033[1;31m=== ray %d has a propagation difference at step %d (correponding solid model step %d) dist_ref "
             "= %.10g :  dist = %.10g\033[0m\n",
             i, istep_err, istep_err_solid, ref_crossings[i].fSteps[istep_err_solid], crossings[i].fSteps[istep_err]);
      if ((istep_err < crossings[i].GetNsteps() - 1) && (istep_err_solid < ref_crossings[i].GetNsteps() - 1) &&
          crossings[i].fStates[istep_err + 1].GetState() != ref_crossings[i].fStates[istep_err_solid + 1].GetState()) {
        printf("\033[1;32msolid model state after step:\033[0m\n");
        ref_crossings[i].fStates[istep_err_solid + 1].Print();
        printf("\033[1;31msurface model using the %s state after step:\033[0m\n", use_bvh ? "BVH" : "looper");
        crossings[i].fStates[istep_err + 1].Print();
      }
      printf("Replaying ray for debugging : \n\n");
      PropagateRaysSolid<LoopNavigator>(points, dirs, in_states, ref_crossings, i, istep_err_solid, config);
      PropagateRaysSurf(points, dirs, in_states, crossings, i, istep_err, /*detect_overlaps =*/false, use_bvh, config);
    }
  }
  return num_errors_dist;
}
//==================================================================================
int testRaytracingHost(Vector3D<Precision> *points, Vector3D<Precision> *dirs, TestConfig const &config)
{
  // allocate storage
  NavigationState *origStates      = new NavigationState[config.nrays];
  NavigationState *outputStates    = new NavigationState[config.nrays];
  NavigationState *outputStatesBVH = new NavigationState[config.nrays];

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
  if (config.validate_results) {
    ref_safeties = new Precision[config.nrays];
    memset(ref_safeties, 0, sizeof(Precision) * config.nrays);

    safeties = new Precision[config.nrays];
    memset(safeties, 0, sizeof(Precision) * config.nrays);

    bvh_safeties = new Precision[config.nrays];
    memset(bvh_safeties, 0, sizeof(Precision) * config.nrays);

    ref_crossings = new CrossingSeq[config.nrays];
    crossings     = new CrossingSeq[config.nrays];
    bvh_crossings = new CrossingSeq[config.nrays];
  }

  int idebug = (config.debug && config.nrays == 1) ? 0 : -1;

  Stopwatch timer;
  // Locating the global points
  timer.Start();
  LocateSolids(points, dirs, origStates, config);
  auto time_locate_solids = timer.Stop();

  timer.Start();
  LocateSolidsBVH(points, outputStates, config);
  auto time_locate_solids_bvh = timer.Stop();

  for (auto i = 0; i < config.nrays; ++i)
    outputStates[i].Clear();

  double time_locate_surf{0.}, time_locate_surf_bvh{0.};
  if (config.use_surf) {
    timer.Start();
    LocateSurf(points, outputStates, config);
    time_locate_surf = timer.Stop();

    for (auto i = 0; i < config.nrays; ++i)
      outputStatesBVH[i].Clear();

    timer.Start();
    if (config.test_bvh) LocateSurfBVH(points, outputStatesBVH, config);
    time_locate_surf_bvh = timer.Stop();
  }

  // Correctness for locating points
  if (config.validate_results) {
    num_errors = ValidateLocate(points, origStates, outputStates, false, config);
    if (num_errors > 0) {
      std::cout << "*** HOST: Point locate errors: " << num_errors << "\n";
      if (config.debug) return num_errors;
    }
  }

  if (config.test_bvh) {
    // Validate BVH Locate
    if (config.validate_results) {
      num_errors_loc_bvh = ValidateLocate(points, origStates, outputStatesBVH, true, config);
      if (num_errors_loc_bvh > 0) {
        std::cout << "*** HOST: BVH point locate errors: " << num_errors_loc_bvh << "\n";
        if (config.debug) return num_errors_loc_bvh;
      }
    }
  }

  if (!config.debug) {
    std::cout << "HOST: locate_solids: " << time_locate_solids << "  locate_solids_BVH: " << time_locate_solids_bvh
              << "  locate_surf: " << time_locate_surf;
    if (config.test_bvh) {
      std::cout << " locate_surf_BVH: " << time_locate_surf_bvh << "\n";
    } else {
      std::cout << "\n";
    }
  }

  // Safety for solids model (reference)
  timer.Start();
  ComputeSafetiesSolid(points, dirs, origStates, ref_safeties, config);
  auto time_safety_solids = timer.Stop();

  // Safety for solids model with BVH
  timer.Start();
  ComputeSafetiesSolidBVH(points, origStates, safeties, config);
  auto time_safety_solids_bvh = timer.Stop();

  // Safety for surface model
  double time_safety_surf{0.}, time_safety_surf_bvh{0.};
  if (config.use_surf) {
    // Safety for surface model
    timer.Start();
    ComputeSafetiesSurf(points, origStates, safeties, config);
    time_safety_surf = timer.Stop();

    // Safety for surface model with BVH
    timer.Start();
    if (config.test_bvh) ComputeSafetiesSurfBVH(points, origStates, bvh_safeties, config);
    time_safety_surf_bvh = timer.Stop();
  }

  // Correctness for safety
  if (config.validate_results) {
    num_errors_safe =
        ValidateSafety(points, origStates, safeties, ref_safeties, num_better_safety, num_worse_safety, config);

    num_errors += num_errors_safe;

    if (num_errors_safe > 0) std::cout << "*** HOST: Safety errors: " << num_errors_safe << "\n";
    if (num_better_safety > 0) printf("HOST:    number of better safety values: %d\n", num_better_safety);
    if (num_worse_safety > 0) printf("HOST:    number of worse safety values: %d\n", num_worse_safety);

    if (config.test_bvh) {
      num_better_safety = 0;
      num_worse_safety  = 0;
      num_errors_safe_bvh =
          ValidateSafety(points, origStates, bvh_safeties, ref_safeties, num_better_safety, num_worse_safety, config);
      num_errors += num_errors_safe_bvh;
      if (num_errors_safe_bvh > 0) std::cout << "*** HOST: BVH safety errors: " << num_errors_safe_bvh << "\n";
      if (num_better_safety > 0) printf("HOST:    BVH number of better safety values: %d\n", num_better_safety);
      if (num_worse_safety > 0) printf("HOST:    BVH number of worse safety values: %d\n", num_worse_safety);
    }
  }

  // Report timing
  if (!config.debug) {
    std::cout << "HOST: safety_solids: " << time_safety_solids << "  safety_solids_BVH: " << time_safety_solids_bvh
              << "  safety_surf: " << time_safety_surf;
    if (config.test_bvh) {
      std::cout << "  safety_surf_bvh: " << time_safety_surf_bvh << "\n";
    } else {
      std::cout << "\n";
    }
  }

  // Distance computation + relocation for solid model
  timer.Start();
  PropagateRaysSolid<LoopNavigator>(points, dirs, origStates, ref_crossings, idebug, -1, config);
  auto time_traverse_solids = timer.Stop();

  // Distance computation + relocation for solid model + BVH
  double time_traverse_solids_bvh(0.);
  if (config.test_bvh) {
    timer.Start();
    PropagateRaysSolid<BVHNavigator>(points, dirs, origStates, crossings, idebug, -1, config);
    time_traverse_solids_bvh = timer.Stop();
  }

  // Distance computation + relocation for surface model
  double time_traverse_surf{0.}, time_traverse_surf_bvh{0.};
  if (config.use_surf) {
    // Distance computation + relocation for surface model
    timer.Start();
    PropagateRaysSurf(points, dirs, origStates, crossings, idebug, /*idebug_step=*/-1, config.detect_overlaps, false,
                      config);
    time_traverse_surf = timer.Stop();

    // Distance computation + relocation for surface model + BVH
    timer.Start();
    if (config.test_bvh)
      PropagateRaysSurf(points, dirs, origStates, bvh_crossings, idebug, /*idebug_step=*/-1,
                        /*detect_overlaps=*/false, true, config);
    time_traverse_surf_bvh = timer.Stop();
  }

  // Correctness for traversal
  if (config.validate_results) {
    num_errors_dist = ValidateCrossing(points, dirs, origStates, ref_crossings, crossings, false, config);
    if (config.test_bvh)
      num_errors_dist_bvh = ValidateCrossing(points, dirs, origStates, ref_crossings, bvh_crossings,
                                             /*use_bvh=*/true, config);
    num_errors += num_errors_dist;
    if (num_errors_dist > 0) std::cout << "*** HOST: traverse errors surf: " << num_errors_dist << "\n";
    if (config.test_bvh)
      if (num_errors_dist_bvh > 0) std::cout << "*** HOST: traverse errors surf BVH: " << num_errors_dist_bvh << "\n";
    if (num_errors > 0) printf("HOST: num_erros = %d / %d\n", num_errors, config.nrays);
  }

  if (!config.debug) {
    std::cout << "HOST: traverse_solids: " << time_traverse_solids
              << "  traverse_solids_BVH: " << time_traverse_solids_bvh << "  traverse_surf: " << time_traverse_surf;
    if (config.test_bvh) std::cout << "  traverse_surf BVH: " << time_traverse_surf_bvh;
    std::cout << std::endl;
  }

  if (config.validate_results) {
    delete[] ref_safeties;
    delete[] safeties;
    delete[] ref_crossings;
    delete[] crossings;
  }
  delete[] origStates;
  delete[] outputStates;

  if (config.validate_results) return num_errors;
  return 0;
}

// in testRaytracing.cu
int testRaytracingCUDA(Vec3Dc const *points, Vec3Dc const *dirs, const SurfData &surfdata, TestConfig &config);

//==================================================================================
int main(int argc, char *argv[])
{
  TestConfig config; // Check TestConfig struct for the meaning of parameters (TODO: add descriptions there)
  OPTION_STRING(gdml_name, "default.gdml");
  OPTION_INT(nrays, 10000);
  config.nrays = nrays;
  OPTION_INT(debug, 0);
  config.debug = debug;
  OPTION_INT(verbosity, 0);
  config.verbosity = verbosity;
  OPTION_INT(min_per_scene, 1000);
  config.min_per_scene = min_per_scene;
  OPTION_INT(ongpu, 1);
  config.ongpu = ongpu;
  OPTION_INT(max_cross, vecgeom::kMaximumInt);
  config.max_cross = max_cross;
  OPTION_BOOL(detect_overlaps, false);
  config.detect_overlaps = detect_overlaps;
  OPTION_BOOL(accept_zeros, true);
  config.accept_zeros = accept_zeros;
  OPTION_BOOL(test_bvh, false);
  config.test_bvh = test_bvh;
  OPTION_BOOL(bvh_single_step, false);
  config.bvh_single_step = bvh_single_step;
  OPTION_BOOL(bvh_split_step, false);
  config.bvh_split_step = bvh_split_step;
  OPTION_BOOL(validate_results, true);
  config.validate_results = validate_results;
  OPTION_BOOL(only_surf, true);
  config.only_surf = only_surf;
  OPTION_BOOL(use_surf, true);
  config.use_surf = use_surf;
  OPTION_BOOL(use_TB_gun, false);
  config.use_TB_gun = use_TB_gun;
  OPTION_DOUBLE(mmunit, 1);
  config.mmunit = mmunit;
  OPTION_DOUBLE(safety_ratio, 0);
  config.safety_ratio = safety_ratio;
  OPTION_ULONG(input_state, 0);
  config.input_state = input_state;
  OPTION_ULONG(next_state, 0);
  config.next_state = next_state;
  OPTION_BOOL(compute_normal, false);
  config.compute_normal = compute_normal;
  OPTION_DOUBLE(step_limit, vecgeom::kInfLength);
  config.step_limit = step_limit;
  OPTION_BOOL(on_boundary, false);
  config.on_boundary = on_boundary;

  std::vector<double> default_point = {vecgeom::InfinityLength<Precision>(), vecgeom::InfinityLength<Precision>(),
                                       vecgeom::InfinityLength<Precision>()};
  OPTION_VECTOR(point, default_point);
  std::vector<double> default_direction = {0., 0., 1.};
  OPTION_VECTOR(direction, default_direction);
  OPTION_VECTOR(max_world, default_point);
  std::vector<double> default_min_world = {-vecgeom::InfinityLength<Precision>(), -vecgeom::InfinityLength<Precision>(),
                                           -vecgeom::InfinityLength<Precision>()};
  OPTION_VECTOR(min_world, default_min_world);
  VECGEOM_ASSERT(point.size() == 3 && direction.size() == 3);
  VECGEOM_ASSERT(min_world.size() == 3 && default_min_world.size() == 3);

  // transform to Vec3D for further handling
  Vec3D point_3D     = {point[0], point[1], point[2]};
  Vec3D direction_3D = {direction[0], direction[1], direction[2]};
  Vec3D min_world_3d = {min_world[0], min_world[1], min_world[2]};
  Vec3D max_world_3d = {max_world[0], max_world[1], max_world[2]};
  if (direction_3D.Mag2() > 0) direction_3D.Normalize();

  vecgeom::logger().level(vecgeom::LogLevel::info);
  bool use_provided_point = point_3D.Mag2() < vecgeom::InfinityLength<Precision>();
#ifndef VECGEOM_USE_NAVINDEX
  if (config.input_state > 0) {
    VECGEOM_LOG(critical) << "You need  VECGEOM_NAV=index to force input state";
    return 1;
  }
#endif
  if (config.input_state > 0) {
    if (!use_provided_point) {
      VECGEOM_LOG(critical) << "Cannot force input state without providing an input point";
      return 1;
    }
    if (!config.debug) {
      config.debug = 1;
      VECGEOM_LOG(info) << "== Debugging enabled due to provided input state";
    }
  }
  if (!config.use_surf) {
    config.validate_results = false;
    config.debug            = 1;
  }

  if (use_provided_point) {
    // check if direction is normalized
    VECGEOM_ASSERT(direction_3D.IsNormalized());
    config.nrays = 1;
  }

  Stopwatch timer;
  // Load the geometry
  timer.Start();
  bool load = LoadGDML(gdml_name.c_str(), ongpu, min_per_scene, mmunit);
  if (load > 0) return load;
  auto time_load = timer.Stop();
  std::cout << "Geometry loading: " << time_load << " [s]\n";

#ifdef VECGEOM_USE_NAVINDEX
  if (config.input_state > 0 && !NavigationState::IsValid(config.input_state, 20)) {
    VECGEOM_LOG(critical) << "Forced input state is invalid";
    return 1;
  }
#endif

  if (config.use_surf) {
    BrepHelper::Instance().SetVerbosity(config.verbosity);
    timer.Start();
    // Conversion to the surface model
    if (!BrepHelper::Instance().Convert()) return 1;
    auto time_surf_dist = timer.Stop();
    BrepHelper::Instance().PrintSurfData();
    std::cout << "Conversion time to surface model: " << time_surf_dist << " [s]\n";
  }

  if (use_provided_point && config.debug)
    VECGEOM_LOG(info) << "Tracking single ray with point " << point_3D << " and direction " << direction_3D;

  // Generate random points and directions inside the setup
  Vec3D *points, *dirs;
  points = new Vec3D[config.nrays];
  dirs   = new Vec3D[config.nrays];
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
    volumeUtilities::FillRandomPoints(amin, amax, points, config.nrays);
    if (!config.use_TB_gun) {
      volumeUtilities::FillRandomDirections(dirs, config.nrays);
    } else {
      // roughly the angles for the testbeam setup
      Precision aMaxPhi   = 0;
      Precision aMinPhi   = vecgeom::kTwoPi;
      Precision aMaxTheta = 0;
      Precision aMinTheta = atan2(20, 3200);
      for (int i = 0; i < config.nrays; i++) {
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
  auto pointsc = new Vec3Dc[config.nrays];
  auto dirsc   = new Vec3Dc[config.nrays];
  for (auto i = 0; i < config.nrays; ++i) {
    auto &pt  = pointsc[i];
    auto &dir = dirsc[i];
    for (auto j = 0; j < 3; ++j) {
      pt[j]  = points[i][j]; // static_cast<Real_t>(points[i][j]);
      dir[j] = dirs[i][j];   // static_cast<Real_t>(dirs[i][j]);
    }
  }

  int errHost = testRaytracingHost(points, dirs, config);
  if ((debug && errHost > 0) || !config.use_surf) return errHost;
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
    if (!errCUDA) errCUDA = testRaytracingCUDA(pointsc, dirsc, surfdata, config);
  }
#endif

  // Clear surface data
  BrepHelper::Instance().ClearData();
  delete[] points;
  delete[] dirs;
  return errHost + errCUDA;
}
