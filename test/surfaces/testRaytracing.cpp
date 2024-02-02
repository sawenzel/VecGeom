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

using namespace vecgeom;
using BrepHelper = vgbrep::BrepHelper<Precision>;
using SurfData   = vgbrep::SurfData<Precision>;
using vecCore::math::Abs;
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
int LoadOnGPU()
{
#ifdef VECGEOM_CUDA_INTERFACE
  std::cout << "synchronizing VecGeom geometry to GPU ...\n";
  auto world = GeoManager::Instance().GetWorld();
  if (!world) return 3;
  // Set higher stack limit to allow depper CSG for the solids model
  CudaAssertError(CudaDeviceSetStackLimit(8192));
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();
  cudaManager.LoadGeometry(world);
  if (!cudaManager.Synchronize()) return 4;
  vecgeom::cxx::BVHManager::DeviceInit();
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
    vgbrep::protonav::LocatePointIn(GeoManager::Instance().GetWorld(), pos, out_states[i], true);
  }
}
//==================================================================================
int ValidateLocate(int nrays, Vector3D<Precision> const *points, NavigationState const *in_states,
                   NavigationState *out_states, bool debug)
{
  int num_errors = 0;
  for (auto i = 0; i < nrays; ++i) {
    if (out_states[i].GetNavIndex() != in_states[i].GetNavIndex()) {
      num_errors++;
      if (debug) {
        printf("%d: p{%16.12f, %16.12f, %16.12f} solid model state:  ", i, points[i][0], points[i][1], points[i][2]);
        in_states[i].Print();
        printf("   model state:  ");
        out_states[i].Print();
        out_states[i].Clear();
        // This just replays the failing locate query for debugging
        LoopNavigator::LocatePointIn(GeoManager::Instance().GetWorld(), points[i], out_states[i], true);
        out_states[i].Clear();
        vgbrep::protonav::LocatePointIn(GeoManager::Instance().GetWorld(), points[i], out_states[i], true);
      }
    }
  }
  return num_errors;
}
//==================================================================================
void ComputeSafetiesSolid(int nrays, Vector3D<Precision> const *points, NavigationState const *in_states,
                          Precision *ref_safeties)
{
  for (auto i = 0; i < nrays; ++i) {
    // Compute safety using the solid-based model
    ref_safeties[i] = LoopNavigator::ComputeSafety(points[i], in_states[i]);
  }
}
//==================================================================================
void ComputeSafetiesSolidBVH(int nrays, Vector3D<Precision> const *points, NavigationState const *in_states,
                             Precision *safeties)
{
  for (auto i = 0; i < nrays; ++i) {
    // Compute safety using the solid-based model with BVH
    safeties[i] = BVHNavigator::ComputeSafety(points[i], in_states[i]);
  }
}
//==================================================================================
void ComputeSafetiesSurf(int nrays, Vector3D<Precision> const *points, NavigationState const *in_states,
                         Precision *safeties)
{
  for (auto i = 0; i < nrays; ++i) {
    int exit_surf;
    safeties[i] = vgbrep::protonav::ComputeSafety(points[i], in_states[i], exit_surf);
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
    safepoint += safety * ranpoint;
    LoopNavigator::LocatePointIn(GeoManager::Instance().GetWorld(), point, new_state, true);

    is_safe = new_state.GetNavIndex() == navind;
    if (!is_safe) break;
  }
  return is_safe;
}
//==================================================================================
int ValidateSafety(int nrays, Vector3D<Precision> const *points, NavigationState const *in_states,
                   Precision const *safeties, Precision const *refSafeties, bool debug, int &num_better_safety,
                   int &num_worse_safety)
{
  int num_errors   = 0;
  int num_warnings = 0;
  for (auto i = 0; i < nrays; ++i) {
    num_better_safety += (safeties[i] > refSafeties[i] + kTolerance);
    num_worse_safety += (safeties[i] < refSafeties[i] - kTolerance);
    if (debug && num_warnings < 10 && safeties[i] < refSafeties[i] - kTolerance) {
      num_warnings++;
      printf("point %d: (%g, %g, %g) safety Solid = %g  safety surf = %g\n", i, points[i][0], points[i][1],
             points[i][2], refSafeties[i], safeties[i]);
      if (num_warnings == 10) printf("=== only fist 10 warnings are shown\n");
      // Replay before exiting for debugging
      int exit_surf = 0;
      vgbrep::protonav::ComputeSafety(points[i], in_states[i], exit_surf);
    }
    if (debug && safeties[i] > refSafeties[i] + kTolerance) {
      bool safesafe = CheckSafety(points[i], in_states[i], safeties[i], 1000);
      if (!safesafe && num_errors < 10) {
        num_errors++;
        printf("point %d: (%g, %g, %g) safety Solid = %g  safety surf = %g NOT SAFE\n", i, points[i][0], points[i][1],
               points[i][2], refSafeties[i], safeties[i]);
        if (num_errors == 10) printf("=== only fist 10 errors are shown\n");
        // Replay before exiting for debugging
        int exit_surf = 0;
        vgbrep::protonav::ComputeSafety(points[i], in_states[i], exit_surf);
      }
    }
  }
  return num_errors;
}
//==================================================================================
template <typename Navigator>
void PropagateRaysSolid(int nrays, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                        NavigationState const *in_states, Precision *length_over_crossings, int idebug = -1)
{
  constexpr double kPushDistance = 1000 * vecgeom::kToleranceDist<Precision>;
  int ilast                      = nrays;
  int istart                     = 0;
  if (idebug >= 0) {
    printf("PropagateRaysSolid debug ray %d: p{%16.12f, %16.12f, %16.12f} d{%16.12f, %16.12f, %16.12f}\n", idebug,
           points[idebug][0], points[idebug][1], points[idebug][2], dirs[idebug][0], dirs[idebug][1], dirs[idebug][2]);
    printf("   ");
    in_states[idebug].Print();
    istart = idebug;
    ilast  = istart + 1;
  }
  for (auto i = istart; i < ilast; ++i) {
    NavigationState start_state = in_states[i];
    NavigationState out_state;
    int num_cross   = 0;
    double dist_tot = 0;
    auto const &dir = dirs[i];
    auto pt         = points[i];
    do {
      auto distance =
          Navigator::ComputeStepAndPropagatedState(pt, dir, kInfLength, start_state, out_state, kPushDistance);
      if (idebug >= 0) {
        printf("     dist = %15.10f\n", distance);
        printf("   ");
        out_state.Print();
      }
      dist_tot += (num_cross + 1) * distance;
      pt += distance * dir;
      start_state = out_state;
      num_cross++;
    } while (!out_state.IsOutside());

    length_over_crossings[i] = num_cross ? dist_tot / (num_cross + 1) : 0;
  }
}
//==================================================================================
void PropagateRaysSurf(int nrays, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                       NavigationState const *in_states, Precision *length_over_crossings, int idebug = -1)
{
  int ilast  = nrays;
  int istart = 0;
  if (idebug >= 0) {
    printf("PropagateRaysSurf debug ray %d:\n", idebug);
    printf("   ");
    in_states[idebug].Print();
    istart = idebug;
    ilast  = istart + 1;
  }
  for (auto i = istart; i < ilast; ++i) {
    NavigationState start_state = in_states[i];
    NavigationState out_state;
    int num_cross   = 0;
    int exit_surf   = 0;
    double dist_tot = 0;
    auto pt         = points[i];
    auto const &dir = dirs[i];
    do {
      exit_surf     = 0; // need to reset because the same inner tube surface can be crossed twice in a row
      auto distance = vgbrep::protonav::ComputeStepAndHit(pt, dir, start_state, out_state, exit_surf);
      if (idebug >= 0) {
        printf("     dist = %15.10f  surf = %d\n", distance, exit_surf);
        printf("   ");
        out_state.Print();
      }
      dist_tot += (num_cross + 1) * distance;
      pt += distance * dir;
      start_state = out_state;
      num_cross++;
    } while (!out_state.IsOutside());
    length_over_crossings[i] = num_cross ? dist_tot / (num_cross + 1) : 0;
  }
}
//==================================================================================
int ValidateCrossing(int nrays, Vector3D<Precision> const *points, Vector3D<Precision> const *dirs,
                     NavigationState const *in_states, Precision *refLength_over_crossings,
                     Precision *length_over_crossings, bool debug)
{
  int num_errors_dist = 0;
  for (auto i = 0; i < nrays; ++i) {
    bool error_dist = Abs(length_over_crossings[i] - refLength_over_crossings[i]) >
                      vgbrep::RoundingError(refLength_over_crossings[i], 100 * kTolerance);
    num_errors_dist += error_dist;
    if (debug && error_dist && (num_errors_dist == 1)) {
      // replay first error
      printf("point %d: dist_ref = %g  dist = %g\n", i, refLength_over_crossings[i], length_over_crossings[i]);
      PropagateRaysSolid<LoopNavigator>(nrays, points, dirs, in_states, refLength_over_crossings, i);
      PropagateRaysSurf(nrays, points, dirs, in_states, length_over_crossings, i);
    }
  }
  return num_errors_dist;
}
//==================================================================================
int testRaytracingHost(int nrays, Vector3D<Precision> *points, Vector3D<Precision> *dirs, bool debug)
{
  // allocate storage
  NavigationState *origStates   = new NavigationState[nrays];
  NavigationState *outputStates = new NavigationState[nrays];

  Precision *refSafeties = new Precision[nrays];
  memset(refSafeties, 0, sizeof(Precision) * nrays);

  Precision *safeties = new Precision[nrays];
  memset(safeties, 0, sizeof(Precision) * nrays);

  Precision *refLength_over_crossings = new Precision[nrays];
  memset(refLength_over_crossings, 0, sizeof(Precision) * nrays);

  Precision *length_over_crossings = new Precision[nrays];
  memset(length_over_crossings, 0, sizeof(Precision) * nrays);

  int num_errors        = 0;
  int num_errors_safe   = 0;
  int num_errors_dist   = 0;
  int num_better_safety = 0;
  int num_worse_safety  = 0;

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

  // Corectness for locating points
  num_errors = ValidateLocate(nrays, points, origStates, outputStates, debug);
  if (num_errors > 0) std::cout << "HOST: Point locate errors: " << num_errors << "\n";
  if (!debug) {
    std::cout << "HOST: locate_solids: " << time_locate_solids << "  locate_solids_BVH: " << time_locate_solids_bvh
              << "  locate_surf: " << time_locate_surf << "\n";
  }

  // Safety for solids model (reference)
  timer.Start();
  ComputeSafetiesSolid(nrays, points, origStates, refSafeties);
  auto time_safety_solids = timer.Stop();

  // Safety for solids model with BVH
  timer.Start();
  ComputeSafetiesSolidBVH(nrays, points, origStates, safeties);
  auto time_safety_solids_bvh = timer.Stop();

  // Safety for surface model
  timer.Start();
  ComputeSafetiesSurf(nrays, points, origStates, safeties);
  auto time_safety_surf = timer.Stop();

  // Corectness for safety
  num_errors_safe =
      ValidateSafety(nrays, points, origStates, safeties, refSafeties, debug, num_better_safety, num_worse_safety);
  num_errors += num_errors_safe;
  // Report timing
  if (num_errors_safe > 0) std::cout << "HOST: Safety errors: " << num_errors_safe << "\n";
  if (!debug) {
    std::cout << "HOST: safety_solids: " << time_safety_solids << "  safety_solids_BVH: " << time_safety_solids_bvh
              << "  safety_surf: " << time_safety_surf << "\n";
  }
  if (num_better_safety > 0) printf("HOST:    number of better safety values: %d\n", num_better_safety);
  if (num_worse_safety > 0) printf("HOST:    number of worse safety values: %d\n", num_worse_safety);

  // Distance computation + relocation for solid model
  timer.Start();
  PropagateRaysSolid<LoopNavigator>(nrays, points, dirs, origStates, refLength_over_crossings);
  auto time_traverse_solids = timer.Stop();

  // Distance computation + relocation for solid model + BVH
  timer.Start();
  PropagateRaysSolid<BVHNavigator>(nrays, points, dirs, origStates, length_over_crossings);
  auto time_traverse_solids_bvh = timer.Stop();

  // Distance computation + relocation for surface model
  timer.Start();
  PropagateRaysSurf(nrays, points, dirs, origStates, length_over_crossings);
  auto time_traverse_surf = timer.Stop();

  // Corectness for traversal
  num_errors_dist =
      ValidateCrossing(nrays, points, dirs, origStates, refLength_over_crossings, length_over_crossings, debug);

  num_errors += num_errors_dist;
  if (num_errors_dist > 0) std::cout << "HOST: traverse errors surf: " << num_errors_dist << "\n";
  if (!debug) {
    std::cout << "HOST: traverse_solids: " << time_traverse_solids
              << "  traverse_solids_BVH: " << time_traverse_solids_bvh << "  traverse_surf: " << time_traverse_surf
              << "  num_errors = " << num_errors_dist << "\n";
  }

  if (num_errors > 0) printf("HOST: num_erros = %d / %d\n", num_errors, nrays);

  delete[] origStates;
  delete[] outputStates;
  delete[] refSafeties;
  delete[] safeties;
  delete[] refLength_over_crossings;
  delete[] length_over_crossings;
  return num_errors;
}

// in testRaytracing.cu
int testRaytracingCUDA(int nrays, Vec3Dc const *points, Vec3Dc const *dirs, const SurfData &surfdata, bool debug);

//==================================================================================
int main(int argc, char *argv[])
{
  OPTION_STRING(gdml_name, "default.gdml");
  OPTION_INT(nrays, 10000);
  OPTION_INT(debug, 0);
  OPTION_INT(verbosity, 0);
  OPTION_INT(min_per_scene, 1000);
  OPTION_INT(ongpu, 1);
  OPTION_DOUBLE(mmunit, 1);
  std::vector<double> default_point = {vecgeom::InfinityLength<Precision>(), vecgeom::InfinityLength<Precision>(),
                                       vecgeom::InfinityLength<Precision>()};
  OPTION_VECTOR(point, default_point);
  std::vector<double> default_direction = {
      0.,
      0.,
      0.,
  };
  OPTION_VECTOR(direction, default_direction);
  assert(point.size() == 3 && direction.size() == 3);
  // transform to Vec3D for further handling
  Vec3D point_3D     = {point[0], point[1], point[2]};
  Vec3D direction_3D = {direction[0], direction[1], direction[2]};

  bool use_provided_point = (direction_3D.Mag2() != 0) || (point_3D.Mag2() < vecgeom::InfinityLength<Precision>());
  if (use_provided_point) {
    // check if direction is normalized
    assert(direction_3D.IsNormalized());
    nrays = 1;
    if (debug)
      std::cout << "Tracking single ray with point " << point_3D << " and direction " << direction_3D << std::endl;
  }

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

  Vec3D origin{0, 0, 0};
  if (!use_provided_point) {
    volumeUtilities::FillRandomPoints(amin, amax, points, nrays);
    volumeUtilities::FillRandomDirections(dirs, nrays);
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
      pt[j]  = points[i][j];
      dir[j] = dirs[i][j];
    }
  }

  int errHost = testRaytracingHost(nrays, points, dirs, debug);
  int errCUDA = 0;
#ifdef VECGEOM_CUDA_INTERFACE
  // Copy geometry to GPU
  auto const &surfdata = BrepHelper::Instance().GetSurfData();
  if (ongpu) {
    timer.Start();
    errCUDA            = LoadOnGPU();
    auto time_transfer = timer.Stop();
    std::cout << "Solid model GPU transfer time: " << time_transfer << " [s]\n";
    if (!errCUDA) errCUDA = testRaytracingCUDA(nrays, pointsc, dirsc, surfdata, debug);
  }
#endif

  // Clear surface data
  BrepHelper::Instance().ClearData();
  delete[] points;
  delete[] dirs;
  return errHost + errCUDA;
}
