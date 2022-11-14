#include <iostream>
#include <string>

#include "test/benchmark/ArgParser.h"
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/base/RNG.h>
#ifdef VECGEOM_GDML
#include <Frontend.h>
#endif

#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/BrepHelper.h>
#include <VecGeom/surfaces/Navigator.h>
#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include <VecGeom/navigation/SimpleSafetyEstimator.h>
#include <VecGeom/volumes/utilities/VolumeUtilities.h>
#include <VecGeom/navigation/NavStatePool.h>
#include <VecGeom/base/Stopwatch.h>

using namespace vecgeom;
// Forwards
void LoadGDML(const char *name);
bool CheckSafety(Vector3D<Precision> const &, NavStateIndex const &, double, int);
double PropagateRay(Vector3D<Precision> const &, Vector3D<Precision> const &,
  NavStateIndex const &);
double PropagateRay(Vector3D<Precision> const &, Vector3D<Precision> const &,
  NavStateIndex const &, vgbrep::SurfData<Precision> const &);

int main(int argc, char *argv[])
{
  using BrepHelper = vgbrep::BrepHelper<Precision>;
  using vecCore::math::Abs;

  OPTION_STRING(gdml_name, "default.gdml");
  OPTION_INT(nrays, 10000);
  OPTION_INT(debug, 0);

  // Load the geometry
#ifndef VECGEOM_GDML
  std::cout << "### VecGeom must be compiled with GDML support to run this.\n";
  return 1;
#else
  auto load = vgdml::Frontend::Load(gdml_name.c_str(), false, 1);
  if (!load) return 2;
#endif

  Stopwatch timer;
  timer.Start();
  // Prepare the model: create the surfaces corresponding to the loaded geometry
  BrepHelper::Instance().SetVerbosity(false);
  if (!BrepHelper::Instance().CreateLocalSurfaces()) return 1;
  if (!BrepHelper::Instance().CreateCommonSurfacesFlatTop()) return 2;
  Precision time_surf_dist = timer.Stop();
  // if (debug)
    std::cout << "Conversion time: " << time_surf_dist << "[s]\n";
  
  auto const &surfdata = BrepHelper::Instance().GetSurfData();

  // Generate random points and directions inside the setup
  SOA3D<Precision> points(nrays);
  SOA3D<Precision> dirs(nrays);

  auto world = GeoManager::Instance().GetWorld();
  assert(world);
  Vector3D<Precision> amin, amax;
  world->GetLogicalVolume()->GetUnplacedVolume()->Extent(amin, amax);

  Vector3D<Precision> samplingVolume = 0.5 * (amax - amin);
  volumeUtilities::FillRandomPoints(samplingVolume, points);
  volumeUtilities::FillRandomDirections(dirs);

  // now setup all the navigation states
  int ndeep = GeoManager::Instance().getMaxDepth();
  NavStatePool origStates(nrays, ndeep);
  NavStatePool outputStates(nrays, ndeep);

  Precision *refSteps = new Precision[nrays];
  memset(refSteps, 0, sizeof(Precision) * nrays);

  Precision *refSafeties = new Precision[nrays];
  memset(refSafeties, 0, sizeof(Precision) * nrays);

  int num_errors        = 0;
  int num_better_safety = 0;
  int num_worse_safety  = 0;

  for (auto i = 0; i < nrays; ++i) {
    Vector3D<Precision> const &pos = points[i];
    Vector3D<Precision> const &dir = dirs[i];

    // Locate with primitive model
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), pos, *origStates[i], true);
    // Validate with surface model
    NavStateIndex locate_state;
    vgbrep::protonav::LocatePointIn(GeoManager::Instance().GetWorld(), pos, locate_state, surfdata, true);
    if (locate_state.GetNavIndex() != origStates[i]->GetNavIndex()) {
      num_errors++;
      if (debug) {
        printf("%d: input state:  ", i);
        origStates[i]->Print();
        printf("   model input state:  ");
        locate_state.Print();
        locate_state.Clear();
        // This just replays the failing locate query for debugging
        vgbrep::protonav::LocatePointIn(GeoManager::Instance().GetWorld(), pos, locate_state, surfdata, true);
        return 3;
      }
    }

    // Compute safety for initial point
    refSafeties[i] = SimpleSafetyEstimator::Instance()->ComputeSafety(pos, *origStates[i]);
    // Validate with surface model
    int exit_surf;
    auto safety = vgbrep::protonav::ComputeSafety(pos, *origStates[i], surfdata, exit_surf);
    if (debug && safety > refSafeties[i] + kTolerance) {
      bool safesafe = CheckSafety(pos, *origStates[i], safety, 1000);
      if (!safesafe) {
        num_errors++;
        // Replay before exiting for debugging
        safety = vgbrep::protonav::ComputeSafety(pos, *origStates[i], surfdata, exit_surf);
        return 4;
      }
    }
    num_better_safety += (safety > refSafeties[i] + kTolerance);
    num_worse_safety += (safety < refSafeties[i] - kTolerance);

    // Traverse geometry till exit
    auto length_over_crossings_ref = PropagateRay(pos, dir, *origStates[i]);
    auto length_over_crossings = PropagateRay(pos, dir, *origStates[i], surfdata);
    bool error_dist = Abs(length_over_crossings - length_over_crossings_ref) > kTolerance;
    if (error_dist) num_errors++;
  }

  printf("=== testRaytracing: num_erros = %d / %d\n", num_errors, nrays);
  if (num_better_safety > 0) printf("    Number of better safety values: %d\n", num_better_safety);

  if (num_worse_safety > 0) printf("    Number of worse safety values: %d\n", num_worse_safety);

  // Test clearing surface data
  BrepHelper::Instance().ClearData();
  delete[] refSteps;
  delete[] refSafeties;
  return num_errors;
}

bool CheckSafety(Vector3D<Precision> const &point, NavStateIndex const &in_state, double safety, int nsamples)
{
  // Generate nsamples random points in a sphere with the safety radius and check if
  // all of them are located in in_state
  auto &rng         = RNG::Instance();
  auto const navind = in_state.GetNavIndex();
  NavStateIndex new_state;
  bool is_safe = true;
  for (int i = 0; i < nsamples; ++i) {
    new_state.Clear();
    Vector3D<Precision> safepoint(point);
    double phi = rng.uniform(0, kTwoPi);
    double the = std::acos(2 * rng.uniform() - 1);
    Vector3D<Precision> ranpoint(std::sin(the) * std::cos(phi), std::sin(the) * std::sin(phi), std::cos(the));
    safepoint += safety * ranpoint;
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), point, new_state, true);

    is_safe = new_state.GetNavIndex() == navind;
    if (!is_safe) break;
  }
  return is_safe;
}

double PropagateRay(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                    NavStateIndex const &in_state, vgbrep::SurfData<Precision> const &surfdata)
{
  // Locate the start point. This is not yet implemented in the surface model
  NavStateIndex start_state = in_state;
  NavStateIndex out_state;
  int num_cross   = 0;
  int exit_surf   = 0;
  double dist_tot = 0;
  auto pt = point;
  do {
    auto distance = vgbrep::protonav::ComputeStepAndHit(pt, direction, start_state, out_state, surfdata, exit_surf);
    dist_tot += distance;
    pt += distance * direction;
    start_state = out_state;
    num_cross++;
  } while (!out_state.IsOutside());

  return num_cross ? dist_tot / num_cross : 0;
}

double PropagateRay(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                    NavStateIndex const &in_state)
{
  // Locate the start point. This is not yet implemented in the surface model
  auto *nav = NewSimpleNavigator<>::Instance();
  NavStateIndex start_state = in_state;
  NavStateIndex out_state;
  int num_cross   = 0;
  double dist_tot = 0;
  auto pt = point + kTolerance * direction;
  do {
    double distance;
    nav->FindNextBoundaryAndStep(pt, direction, start_state, out_state, kInfLength, distance);
    dist_tot += distance + kTolerance;
    pt += (distance + kTolerance) * direction;
    start_state = out_state;
    num_cross++;
  } while (!out_state.IsOutside());

  return num_cross ? dist_tot / num_cross : 0;
}
