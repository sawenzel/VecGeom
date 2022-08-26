#include <iostream>
#include <string>

#include "test/benchmark/ArgParser.h"
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/Box.h>
#include <VecGeom/volumes/Tube.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/base/RNG.h>

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
void CreateVecGeomWorld(int, int);
void CreateTubeWorld(int, int);
bool CheckSafety(Vector3D<Precision> const &, NavStateIndex const &, double, int);
bool ValidateNavigation(int, int, int, int, vgbrep::SurfData<Precision> const &);
bool ValidateTubeNavigation(int, int, vgbrep::SurfData<Precision> const &);
void TestPerformance(int, int, int, int, vgbrep::SurfData<Precision> const &);
double PropagateRay(Vector3D<Precision> const &, Vector3D<Precision> const &, vgbrep::SurfData<Precision> const &);

int main(int argc, char *argv[])
{
  using BrepHelper = vgbrep::BrepHelper<Precision>;

  OPTION_INT(layers, 50);
  OPTION_INT(absorbers, 2.);
  OPTION_INT(nvalidate, 10000);
  OPTION_INT(nbench, 1000000);
  OPTION_INT(verbose, 0);
  OPTION_INT(distcheck, 1);
  OPTION_INT(safecheck, 1);

  CreateVecGeomWorld(layers, absorbers);

  BrepHelper::Instance().SetVerbosity(verbose);

  if (!BrepHelper::Instance().CreateLocalSurfaces()) return 1;
  if (!BrepHelper::Instance().CreateCommonSurfacesFlatTop()) return 2;

  ValidateNavigation(nvalidate, layers, distcheck, safecheck, BrepHelper::Instance().GetSurfData());

  TestPerformance(nbench, layers, distcheck, safecheck, BrepHelper::Instance().GetSurfData());

  // Test clearing surface data
  BrepHelper::Instance().ClearData();
}

void CreateVecGeomWorld(int NbOfLayers, int NbOfAbsorbers)
{
  const double CalorSizeY        = 40;
  const double CalorSizeZ        = 40;
  const double GapThickness      = 2.3;
  const double AbsorberThickness = 5.7;

  const double LayerThickness = GapThickness + AbsorberThickness;
  const double CalorThickness = NbOfLayers * LayerThickness;

  const double WorldSizeX = 1.2 * CalorThickness + 1000;
  const double WorldSizeY = 1.2 * CalorSizeY + 1000;
  const double WorldSizeZ = 1.2 * CalorSizeZ + 1000;

  auto worldSolid            = new UnplacedBox(0.5 * WorldSizeX, 0.5 * WorldSizeY, 0.5 * WorldSizeZ);
  auto worldLogic            = new LogicalVolume("World", worldSolid);
  VPlacedVolume *worldPlaced = worldLogic->Place();

  //
  // Calorimeter
  //
  auto calorSolid = new UnplacedBox(0.5 * CalorThickness, 0.5 * CalorSizeY, 0.5 * CalorSizeZ);
  auto calorLogic = new LogicalVolume("Calorimeter", calorSolid);
  Transformation3D origin;
  worldLogic->PlaceDaughter(calorLogic, &origin);

  //
  // Layers
  //
  auto layerSolid = new UnplacedBox(0.5 * LayerThickness, 0.5 * CalorSizeY, 0.5 * CalorSizeZ);

  //
  // Absorbers
  //
  auto gapSolid = new UnplacedBox(0.5 * GapThickness, 0.5 * CalorSizeY, 0.5 * CalorSizeZ);
  auto gapLogic = new LogicalVolume("Gap", gapSolid);
  Transformation3D gapPlacement(-0.5 * LayerThickness + 0.5 * GapThickness, 0, 0);

  auto absorberSolid = new UnplacedBox(0.5 * AbsorberThickness, 0.5 * CalorSizeY, 0.5 * CalorSizeZ);
  auto absorberLogic = new LogicalVolume("Absorber", absorberSolid);
  Transformation3D absorberPlacement(0.5 * LayerThickness - 0.5 * AbsorberThickness, 0, 0);

  // Create a new LogicalVolume per layer, we need unique IDs for scoring.
  double xCenter = -0.5 * CalorThickness + 0.5 * LayerThickness;
  for (int i = 0; i < NbOfLayers; i++) {
    std::string name("Layer_");
    name += std::to_string(i);
    auto layerLogic = new LogicalVolume("Layer", layerSolid);
    Transformation3D placement(xCenter, 0, 0);
    calorLogic->PlaceDaughter(name.c_str(), layerLogic, &placement);

    layerLogic->PlaceDaughter(gapLogic, &gapPlacement);
    layerLogic->PlaceDaughter(absorberLogic, &absorberPlacement);

    xCenter += LayerThickness;
  }

  GeoManager::Instance().SetWorldAndClose(worldPlaced);
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
                    vgbrep::SurfData<Precision> const &surfdata)
{
  // Locate the start point. This is not yet implemented in the surface model
  NavStateIndex in_state, out_state;
  int exit_surf   = 0;
  double dist_tot = 0;
  GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), point, in_state, true);
  auto pt = point;
  printf("start: ");
  in_state.Print();
  do {
    auto distance = vgbrep::protonav::ComputeStepAndHit(pt, direction, in_state, out_state, surfdata, exit_surf);
    if (exit_surf != 0) {
      dist_tot += distance;
      pt += distance * direction;
    }
    printf("surface %d at dist = %g: ", exit_surf, distance);
    out_state.Print();
    in_state = out_state;
  } while (!out_state.IsOutside());

  return dist_tot;
}

bool ValidateNavigation(int npoints, int nbLayers, int distcheck, int safecheck,
                        vgbrep::SurfData<Precision> const &surfdata)
{
  // prepare tracks to be used for benchmarking
  constexpr double tolerance     = 10 * kTolerance;
  const double CalorSizeY        = 40;
  const double CalorSizeZ        = 40;
  const double GapThickness      = 2.3;
  const double AbsorberThickness = 5.7;

  const double LayerThickness = GapThickness + AbsorberThickness;
  const double CalorThickness = nbLayers * LayerThickness;

  int num_errors        = 0;
  int num_better_safety = 0;
  int num_worse_safety  = 0;
  SOA3D<Precision> points(npoints);
  SOA3D<Precision> dirs(npoints);

  Vector3D<Precision> samplingVolume(0.5 * CalorThickness + 30, 0.5 * CalorSizeY + 30, 0.5 * CalorSizeZ + 30);
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

  auto *nav = NewSimpleNavigator<>::Instance();
  for (int i = 0; i < npoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    Vector3D<Precision> const &dir = dirs[i];
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), pos, *origStates[i], true);
    if (distcheck) nav->FindNextBoundaryAndStep(pos, dir, *origStates[i], *outputStates[i], kInfLength, refSteps[i]);
    if (safecheck) refSafeties[i] = SimpleSafetyEstimator::Instance()->ComputeSafety(pos, *origStates[i]);

    // shoot the same ray in the surface model
    int exit_surf = 0;
    NavStateIndex out_state;
    double distance = 0, safety = 0;
    bool safesafe = true;
    if (distcheck)
      distance = vgbrep::protonav::ComputeStepAndHit(pos, dir, *origStates[i], out_state, surfdata, exit_surf);
    if (safecheck) {
      safety = vgbrep::protonav::ComputeSafety(pos, *origStates[i], surfdata, exit_surf);
      if (safety > refSafeties[i] + kTolerance) safesafe = CheckSafety(pos, *origStates[i], safety, 1000);
      num_better_safety += safesafe && (safety > refSafeties[i] + kTolerance);
      num_worse_safety += safesafe && (safety < refSafeties[i] - kTolerance);
    }
    bool errpath = out_state.GetNavIndex() != outputStates[i]->GetNavIndex();
    bool errdist = distcheck ? std::abs(distance - refSteps[i]) > tolerance : false;
    bool errsafe = !safesafe;
    bool err     = errpath || errdist || errsafe;
    if (err) num_errors++;

    if (err) {
      printf("%d: input state:  ", i);
      origStates[i]->Print();
      printf("ref output state: ");
      outputStates[i]->Print();
      printf("model output state: ");
      out_state.Print();
    }

    if (errdist) printf("ref dist: %g   model dist: %g\n", refSteps[i], distance);

    if (errsafe) printf("ref safe: %g   model safe: %g\n", refSafeties[i], safety);
  }

  printf("=== Validation: num_erros = %d / %d\n", num_errors, npoints);
  if (num_better_safety > 0) printf("    Number of better safety values: %d\n", num_better_safety);

  if (num_worse_safety > 0) printf("    Number of worse safety values: %d\n", num_worse_safety);

  delete[] refSteps;
  delete[] refSafeties;

  return num_errors == 0;
}

void TestPerformance(int npoints, int nbLayers, int distcheck, int safecheck,
                     vgbrep::SurfData<Precision> const &surfdata)
{
  const double CalorSizeYZ       = 40;
  const double GapThickness      = 2.3;
  const double AbsorberThickness = 5.7;

  const double LayerThickness = GapThickness + AbsorberThickness;
  const double CalorThickness = nbLayers * LayerThickness;

  SOA3D<Precision> points(npoints);
  SOA3D<Precision> dirs(npoints);

  //  Vector3D<Precision> samplingVolume(0.2, 0.2, 0.2);
  Vector3D<Precision> samplingVolume(0.5 * CalorThickness, 0.5 * CalorSizeYZ, 0.5 * CalorSizeYZ);
  volumeUtilities::FillRandomPoints(samplingVolume, points);
  volumeUtilities::FillRandomDirections(dirs);

  Precision xfirst  = -0.5 * CalorThickness + 0.5 * LayerThickness;
  Precision xlast   = xfirst + (nbLayers - 1) * LayerThickness;
  Precision xmiddle = xfirst + 0.5 * (nbLayers - 1) * LayerThickness;

  Vector3D<Precision> pointInFirstLayer(xfirst, 0, 0);
  Vector3D<Precision> pointInLastLayer(xlast, 0, 0);
  Vector3D<Precision> pointInMiddleLayer(xmiddle, 0, 0);
  Vector3D<Precision> pointBottomFirstLayer(xfirst, -0.6 * CalorSizeYZ, 0);
  Vector3D<Precision> pointBottomLastLayer(xlast, -0.6 * CalorSizeYZ, 0);
  Vector3D<Precision> pointBottomMiddleLayer(xmiddle, -0.6 * CalorSizeYZ, 0);

  Vector3D<Precision> dirXplus(1, 0, 0);
  Vector3D<Precision> dirXminus(-1, 0, 0);
  Vector3D<Precision> dirYplus(0, 1, 0);
  Vector3D<Precision> dirYminus(0, -1, 0);
  Vector3D<Precision> dirXY(1, 0, 0);
  dirXY.Normalize();

  // Vector3D<Precision> const &pt  = pointBottomLastLayer;
  // Vector3D<Precision> const &dir = dirYplus;

  // now setup all the navigation states
  int ndeep = GeoManager::Instance().getMaxDepth();
  NavStatePool origStates(npoints, ndeep);
  NavStateIndex out_state;
  Precision distance = 0;
  auto *nav          = NewSimpleNavigator<>::Instance();

  // Locate all input points, without timing this operation
  for (int i = 0; i < npoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    // Vector3D<Precision> pos(points[i] + pt);
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), pos, *origStates[i], true);
  }

  // Benchamrk primitive-based NewSimpleNavigator
  Stopwatch timer;
  if (distcheck) {
    timer.Start();
    for (int i = 0; i < npoints; ++i) {
      // Vector3D<Precision> pos(points[i] + pt);
      Vector3D<Precision> const &pos = points[i];
      Vector3D<Precision> const &dir = dirs[i];
      nav->FindNextBoundaryAndStep(pos, dir, *origStates[i], out_state, kInfLength, distance);
      // out_state.Print();
    }
    //printf("\n");
    Precision time_prim_dist = timer.Stop();

    timer.Start();
    for (int i = 0; i < npoints; ++i) {
      // Vector3D<Precision> pos(points[i] + pt);
      Vector3D<Precision> const &pos = points[i];
      Vector3D<Precision> const &dir = dirs[i];
      int exit_surf                  = 0;
      distance = vgbrep::protonav::ComputeStepAndHit(pos, dir, *origStates[i], out_state, surfdata, exit_surf);
      // out_state.Print();
    }
    Precision time_surf_dist = timer.Stop();

    printf("Distance for %d points: NewSimpleNavigator = %f [s]  vgbrep::protonav = %f\n", npoints, time_prim_dist,
           time_surf_dist);
  }

  if (safecheck) {
    timer.Start();
    for (int i = 0; i < npoints; ++i) {
      Vector3D<Precision> const &pos = points[i];
      SimpleSafetyEstimator::Instance()->ComputeSafety(pos, *origStates[i]);
    }
    Precision time_prim_safe = timer.Stop();

    timer.Start();
    for (int i = 0; i < npoints; ++i) {
      int exit_surf                  = 0;
      Vector3D<Precision> const &pos = points[i];
      vgbrep::protonav::ComputeSafety(pos, *origStates[i], surfdata, exit_surf);
    }
    Precision time_surf_safe = timer.Stop();

    printf("Safety for %d points: SimpleSafetyEstimator = %f [s]  vgbrep::protonav = %f\n", npoints, time_prim_safe,
           time_surf_safe);
  }
}
