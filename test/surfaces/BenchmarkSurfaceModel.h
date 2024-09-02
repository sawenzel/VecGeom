#include <iostream>
#include <fstream>
#include <string>

#include "test/benchmark/ArgParser.h"
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/base/RNG.h>

#include <VecGeom/base/Math.h>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/BrepHelper.h>
#include <VecGeom/surfaces/Navigator.h>
#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include <VecGeom/volumes/utilities/VolumeUtilities.h>
#include <VecGeom/navigation/NavStatePool.h>
#include "VecGeom/base/Stopwatch.h"

using BrepHelper = vgbrep::BrepHelper<vecgeom::Precision>;
using namespace vecgeom;

bool CheckSafety(Vector3D<Precision> const &point, NavigationState const &in_state, Precision safety, int nsamples)
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
    Precision phi = rng.uniform(0, kTwoPi);
    Precision the = std::acos(2 * rng.uniform() - 1);
    Vector3D<Precision> ranpoint(std::sin(the) * std::cos(phi), std::sin(the) * std::sin(phi), std::cos(the));
    safepoint += safety * ranpoint;
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), point, new_state, true);

    is_safe = new_state.GetNavIndex() == navind;
    if (!is_safe) break;
  }
  return is_safe;
}

Precision PropagateRay(vecgeom::Vector3D<vecgeom::Precision> const &point,
                       vecgeom::Vector3D<vecgeom::Precision> const &direction,
                       vgbrep::SurfData<vecgeom::Precision> const &surfdata)
{
  // Locate the start point. This is not yet implemented in the surface model
  NavigationState in_state, out_state;
  vgbrep::CrossedSurface crossed_surf;
  Precision dist_tot = 0;
  GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), point, in_state, true);
  auto pt = point;
  printf("start: ");
  in_state.Print();
  do {
    auto distance =
        vgbrep::protonav::ComputeStepAndHit<Precision, Precision>(pt, direction, in_state, out_state, crossed_surf);
    if (crossed_surf.hit_surf.GetCSindex() != 0) {
      dist_tot += distance;
      pt += distance * direction;
    }
    printf("surface %d at dist = %g: ", crossed_surf.hit_surf.GetCSindex(), distance);
    out_state.Print();
    in_state = out_state;
  } while (!out_state.IsOutside());

  return dist_tot;
}

bool ValidateNavigation(int npoints, Precision worldX, Precision worldY, Precision worldZ, Precision scale)
{
  constexpr Precision tolerance = 10 * vecgeom::kTolerance;

  int num_errors        = 0;
  int num_better_safety = 0;
  int num_worse_safety  = 0;
  SOA3D<Precision> points(npoints);
  SOA3D<Precision> dirs(npoints);

  Vector3D<Precision> samplingVolume(worldX * scale, worldY * scale, worldZ * scale);
  volumeUtilities::FillRandomPoints(samplingVolume, points);
  volumeUtilities::FillRandomDirections(dirs);

  // now setup all the navigation states
  int ndeep = GeoManager::Instance().getMaxDepth();
  NavStatePool origStates(npoints, ndeep);
  NavStatePool outputStates(npoints, ndeep);

  Precision *refSteps = new Precision[npoints];
  memset(refSteps, 0, sizeof(Precision) * npoints);

  Precision *refSafeties = new Precision[npoints];
  memset(refSafeties, 0, sizeof(Precision) * npoints);

  auto *nav = vecgeom::NewSimpleNavigator<>::Instance();
  for (int i = 0; i < npoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    Vector3D<Precision> const &dir = dirs[i];
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), pos, *origStates[i], true);
    nav->FindNextBoundaryAndStep(pos, dir, *origStates[i], *outputStates[i], vecgeom::kInfLength, refSteps[i]);
    refSafeties[i] = SimpleSafetyEstimator::Instance()->ComputeSafety(pos, *origStates[i]);

    // shoot the same ray in the surface model
    vgbrep::CrossedSurface crossed_surf;
    bool safesafe = true;
    NavigationState in_state, out_state, surflocate_state;
    vgbrep::protonav::LocatePointIn<Precision, Precision>(NavigationState::WorldId(), pos, surflocate_state, true);
    auto distance =
        vgbrep::protonav::ComputeStepAndHit<Precision, Precision>(pos, dir, *origStates[i], out_state, crossed_surf);
    int common_id = crossed_surf.hit_surf.GetCSindex();
    auto safety   = vgbrep::protonav::ComputeSafety<Precision, Precision>(pos, *origStates[i], common_id);
    if (safety > refSafeties[i] + kTolerance) safesafe = CheckSafety(pos, *origStates[i], safety, 1000);
    num_better_safety += safesafe && (safety > refSafeties[i] + kTolerance);
    num_worse_safety += safesafe && (safety < refSafeties[i] - kTolerance);
    if (safety < refSafeties[i] - kTolerance)
      printf("safe_ref = %g   safe = %g  pos = (%g, %g, %g)\n", refSafeties[i], safety, pos[0], pos[1], pos[2]);

    bool errlocate = surflocate_state.GetNavIndex() != origStates[i]->GetNavIndex();
    bool errpath   = out_state.GetNavIndex() != outputStates[i]->GetNavIndex();
    bool errdist   = std::abs(distance - refSteps[i]) > tolerance;
    bool errsafe   = !safesafe;
    bool err       = errlocate || errpath || errdist || errsafe;
    num_errors += int(err);
    if (err) {
      printf("%d: input state:  ", i);
      origStates[i]->Print();
      printf("%d: model locate: ", i);
      surflocate_state.Print();
      printf("ref output state: ");
      outputStates[i]->Print();
      printf("model output state: ");
      out_state.Print();
    }

    if (errdist) printf("ref dist: %g   model dist: %g\n", refSteps[i], distance);

    if (errsafe) printf("ref safe: %g   model safe: %g\n", refSafeties[i], safety);
  }

  printf("=== Validation: num_erros = %d / %d\n", num_errors, npoints);
  delete[] refSteps;
  delete[] refSafeties;

  return num_errors == 0;
}

bool ShootOneParticle(Precision px, Precision py, Precision pz, Precision dx, Precision dy, Precision dz)
{
  // Very hacky, as I don't know how all the backend stuff works. -DC

  constexpr Precision tolerance = 10 * vecgeom::kTolerance;

  int num_errors = 0;
  SOA3D<Precision> points(1);
  SOA3D<Precision> dirs(1);

  Vector3D<Precision> point{px, py, pz};
  Vector3D<Precision> direction{dx, dy, dz};
  direction.Normalize();

  std::cout << "Shooting single particle." << std::endl
            << "POS: " << point << std::endl
            << "DIR: " << direction << std::endl;

  // now setup all the navigation states
  int ndeep = GeoManager::Instance().getMaxDepth();
  NavStatePool origStates(1, ndeep);
  NavStatePool outputStates(1, ndeep);

  vecgeom::Precision *refSteps = new Precision[1];
  memset(refSteps, 0, sizeof(Precision));

  auto *nav = vecgeom::NewSimpleNavigator<>::Instance();

  GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), point, *origStates[0], true);
  nav->FindNextBoundaryAndStep(point, direction, *origStates[0], *outputStates[0], vecgeom::kInfLength, refSteps[0]);

  // shoot the same ray in the surface model
  vgbrep::CrossedSurface crossed_surf;
  NavigationState out_state;
  auto distance = vgbrep::protonav::ComputeStepAndHit<Precision, Precision>(point, direction, *origStates[0], out_state,
                                                                            crossed_surf);
  if (out_state.GetNavIndex() != outputStates[0]->GetNavIndex() || std::abs(distance - refSteps[0]) > tolerance) {
    num_errors++;
    std::cout << "ERROR." << std::endl;
    std::cout << " IN VOLUME MODEL: (navind) " << outputStates[0]->GetNavIndex() << "; (distance) " << refSteps[0]
              << std::endl;
    std::cout << "IN SURFACE MODEL: (navind) " << out_state.GetNavIndex() << "; (distance) " << distance << std::endl;
    return 0;
  }

  std::cout << "VALIDATION OK." << std::endl;
  delete[] refSteps;

  return num_errors == 0;
}

void TestPerformance(Precision worldX, Precision worldY, Precision worldZ, Precision scale, int npoints, int nbLayers)
{
  SOA3D<Precision> points(npoints);
  SOA3D<Precision> dirs(npoints);

  Vector3D<Precision> samplingVolume(scale * worldX, scale * worldY, scale * worldZ);
  vecgeom::volumeUtilities::FillRandomPoints(samplingVolume, points);
  vecgeom::volumeUtilities::FillRandomDirections(dirs);

  // now setup all the navigation states
  int ndeep = GeoManager::Instance().getMaxDepth();
  NavStatePool origStates(npoints, ndeep);
  NavigationState out_state;
  vecgeom::Precision distance = 0;
  auto *nav                   = vecgeom::NewSimpleNavigator<>::Instance();

  // Locate all input points, without timing this operation
  for (int i = 0; i < npoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), pos, *origStates[i], true);
  }

  // Benchamrk primitive-based NewSimpleNavigator
  Stopwatch timer;
  timer.Start();
  for (int i = 0; i < npoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    Vector3D<Precision> const &dir = dirs[i];
    nav->FindNextBoundaryAndStep(pos, dir, *origStates[i], out_state, vecgeom::kInfLength, distance);
  }
  Precision time_prim = timer.Stop();

  Stopwatch timer1;
  timer1.Start();
  for (int i = 0; i < npoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    Vector3D<Precision> const &dir = dirs[i];
    vgbrep::CrossedSurface crossed_surf;
    distance =
        vgbrep::protonav::ComputeStepAndHit<Precision, Precision>(pos, dir, *origStates[i], out_state, crossed_surf);
  }
  Precision time_surf = timer1.Stop();

  printf("Time for %d points: NewSimpleNavigator = %f [s]  vgbrep::protonav = %f\n", npoints, time_prim, time_surf);
}

// Not updated
void TestAndSavePerformance(Precision worldRadius, int npoints, int nbLayers)
{
  const Precision CalorSizeR        = worldRadius;
  const Precision GapThickness      = 2.3;
  const Precision AbsorberThickness = 5.7;

  const Precision LayerThickness = GapThickness + AbsorberThickness;
  const Precision CalorThickness = nbLayers * LayerThickness;

  SOA3D<Precision> points(npoints);
  SOA3D<Precision> dirs(npoints);

  Vector3D<Precision> samplingVolume(0.5 * CalorSizeR, 0.5 * CalorSizeR, 0.5 * CalorThickness);
  vecgeom::volumeUtilities::FillRandomPoints(samplingVolume, points);
  vecgeom::volumeUtilities::FillRandomDirections(dirs);

  // now setup all the navigation states
  int ndeep = GeoManager::Instance().getMaxDepth();
  NavStatePool origStates(npoints, ndeep);
  NavigationState out_state;
  vecgeom::Precision distance = 0;
  auto *nav                   = vecgeom::NewSimpleNavigator<>::Instance();

  // Locate all input points, without timing this operation
  for (int i = 0; i < npoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), pos, *origStates[i], true);
  }

  // Benchamrk primitive-based NewSimpleNavigator
  Stopwatch timer;
  timer.Start();
  for (int i = 0; i < npoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    Vector3D<Precision> const &dir = dirs[i];
    nav->FindNextBoundaryAndStep(pos, dir, *origStates[i], out_state, vecgeom::kInfLength, distance);
  }
  Precision time_prim = timer.Stop();

  Stopwatch timer1;
  timer1.Start();
  for (int i = 0; i < npoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    Vector3D<Precision> const &dir = dirs[i];
    vgbrep::CrossedSurface crossed_surf;
    distance =
        vgbrep::protonav::ComputeStepAndHit<Precision, Precision>(pos, dir, *origStates[i], out_state, crossed_surf);
  }
  Precision time_surf = timer1.Stop();

  printf("Time for %d points: NewSimpleNavigator = %f [s]  vgbrep::protonav = %f\n", npoints, time_prim, time_surf);

  std::ofstream file_out;
  file_out.open("performance_measuring.txt", std::ios_base::app);
  file_out << nbLayers << " " << npoints << " " << time_prim << " " << time_surf << std::endl;
  file_out.close();
  std::cout << "PRINTED TO FILE." << std::endl;
}