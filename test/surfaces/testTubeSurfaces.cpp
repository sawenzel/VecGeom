

#include <iostream>
#include <fstream>
#include <string>

#include "test/benchmark/ArgParser.h"
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/Tube.h>
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

// Forward definitions of geometry-creating functions.
void CreateSimpleTube(double, double, double, double, double, double, double);
void CreateTwoNestedTubes(double, double, double, double, double, double, double, double, double, double, double,
                          double);
void CreateIdenticalTubes(double, double, double, double, double, double, double);
void CreateConcatenatedTubes(double, double, double, double, double, double, double);
void CreateLayeredGeometry(double, double, int, int, double, double, double);

// Forward definitions of testing functions.
bool ValidateNavigation(int, int, vgbrep::SurfData<vecgeom::Precision> const &, double, double, double);
bool ShootOneParticle(double, double, double, double, double, double, double, double,
                      vgbrep::SurfData<vecgeom::Precision> const &);
void TestPerformance(double, int, int, vgbrep::SurfData<vecgeom::Precision> const &);
void TestAndSavePerformance(double, int, int, vgbrep::SurfData<vecgeom::Precision> const &);

int main(int argc, char *argv[])
{
  //
  // Testing options
  //
  OPTION_INT(generate, 0);
  OPTION_INT(test, 0);
  OPTION_INT(verbose, 0);
  // Determines the scale of generating volume relative to world:
  OPTION_DOUBLE(scale, 0.5);

  //
  // World and validation options
  //
  OPTION_DOUBLE(worldRadius, 40);
  OPTION_DOUBLE(worldZ, 80);
  OPTION_INT(nvalidate, 10000);

  //
  // Complex geometry options
  //
  OPTION_INT(layers, 50);
  OPTION_INT(absorbers, 2);
  OPTION_DOUBLE(rmin, 0);
  OPTION_DOUBLE(sphi, 0);
  OPTION_DOUBLE(dphi, 360);

  // protection for sphi and dphi, copied from TubeStruct (why not enabled there?)
  sphi *= vecgeom::kDegToRad;
  dphi *= vecgeom::kDegToRad;
  if (dphi < 0) {
    std::cout << "Cannot have negative dphi" << std::endl;
    return 1;
  }

  if (dphi >= vecgeom::kTwoPi - 0.5 * vecgeom::kAngTolerance) {
    dphi = vecgeom::kTwoPi;
    sphi = 0;
  }

  if (sphi < 0) {
    sphi = vecgeom::kTwoPi - std::fmod(std::fabs(sphi), vecgeom::kTwoPi);
  } else {
    sphi = std::fmod(sphi, vecgeom::kTwoPi);
  }

  printf("sphi=%g dphi=%g\n", sphi * vecgeom::kRadToDeg, dphi * vecgeom::kRadToDeg);

  //
  // Benchmarking options:
  //
  OPTION_INT(nbench, 1000000);

  switch (generate) {
  case 0:
    std::cout << "Creating layered geometry..." << std::endl;
    CreateLayeredGeometry(worldRadius, worldZ, layers, absorbers, rmin, sphi, dphi);
    break;
  case 1:
    std::cout << "Creating simple tube..." << std::endl;
    CreateSimpleTube(worldRadius, worldZ, rmin, worldRadius * 0.7, worldZ * 0.5, sphi, dphi);
    break;
  case 2:
    std::cout << "Creating nested tube..." << std::endl;
    CreateTwoNestedTubes(worldRadius, worldZ, 1, worldRadius / 2, worldZ / 2, 0, vecgeom::kTwoPi, 1, worldRadius / 4,
                         worldZ / 4, sphi, dphi);
    break;
  case 3:
    std::cout << "Creating identical tubes..." << std::endl;
    CreateIdenticalTubes(worldRadius, worldZ, rmin, worldRadius / 2, worldZ / 2, 1, 3);
    break;
  case 4:
    std::cout << "Creating concatenated tubes..." << std::endl;
    CreateConcatenatedTubes(worldRadius, worldZ, rmin, 10, 10, sphi, dphi);
    break;
  default:
    std::cout << "Generation option " << generate << " does not exist." << std::endl;
    return -1;
  }

  BrepHelper::Instance().SetVerbosity(verbose);

  if (!BrepHelper::Instance().CreateLocalSurfaces()) return 1;
  if (!BrepHelper::Instance().CreateCommonSurfacesFlatTop()) return 2;

  switch (test) {
  case 0:
    ValidateNavigation(nvalidate, 10, BrepHelper::Instance().GetSurfData(), worldRadius, worldZ, scale);
    break;
  case 1:
    ShootOneParticle(worldRadius, worldZ,
                     1, -10, 0,
                     0, 1, 0,
                     BrepHelper::Instance().GetSurfData());
    break;
  case 2:
    ValidateNavigation(nvalidate, 10, BrepHelper::Instance().GetSurfData(), worldRadius, worldZ, scale);
    TestPerformance(worldRadius, nbench, layers, BrepHelper::Instance().GetSurfData());
    break;
  case 3:
    TestAndSavePerformance(worldRadius, nbench, layers, BrepHelper::Instance().GetSurfData());
    break;
  default:
    std::cout << "Test " << test << " does not exist." << std::endl;
  }
}

void CreateSimpleTube(double worldR, double worldZ, double rmin, double rmax, double z, double sphi, double dphi)
{
  auto worldSolid                     = new vecgeom::UnplacedBox(worldR, worldR, worldZ);
  auto worldLogic                     = new vecgeom::LogicalVolume("World", worldSolid);
  vecgeom::VPlacedVolume *worldPlaced = worldLogic->Place();

  auto const tubeSolid = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTube>(rmin, rmax, z, sphi, dphi);
  auto tubeLogic       = new vecgeom::LogicalVolume("Tube", tubeSolid);
  vecgeom::Transformation3D origin;
  worldLogic->PlaceDaughter(tubeLogic, &origin);

  vecgeom::GeoManager::Instance().SetWorldAndClose(worldPlaced);
}

void CreateTwoNestedTubes(double worldR, double worldZ, double rmin, double rmax, double z, double sphi, double dphi,
                          double rmin2, double rmax2, double z2, double sphi2, double dphi2)
{
  auto worldSolid                     = new vecgeom::UnplacedBox(worldR, worldR, worldZ);
  auto worldLogic                     = new vecgeom::LogicalVolume("World", worldSolid);
  vecgeom::VPlacedVolume *worldPlaced = worldLogic->Place();

  vecgeom::Transformation3D origin;

  auto const outerTube = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTube>(rmin, rmax, z, sphi, dphi);
  auto outerTubeLogic  = new vecgeom::LogicalVolume("OuterTube", outerTube);
  worldLogic->PlaceDaughter(outerTubeLogic, &origin);

  auto const innerTube = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTube>(rmin2, rmax2, z2, sphi2, dphi2);
  auto innerTubeLogic  = new vecgeom::LogicalVolume("InnerTube", innerTube);
  outerTubeLogic->PlaceDaughter(innerTubeLogic, &origin);

  vecgeom::GeoManager::Instance().SetWorldAndClose(worldPlaced);
}

void CreateIdenticalTubes(double worldR, double worldZ, double rmin, double rmax, double z, double sphi, double dphi)
{
  auto worldSolid                     = new vecgeom::UnplacedBox(worldR, worldR, worldZ);
  auto worldLogic                     = new vecgeom::LogicalVolume("World", worldSolid);
  vecgeom::VPlacedVolume *worldPlaced = worldLogic->Place();

  vecgeom::Transformation3D origin;

  auto const outerTube = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTube>(rmin, rmax, z, sphi, dphi);
  auto outerTubeLogic  = new vecgeom::LogicalVolume("OuterTube", outerTube);
  worldLogic->PlaceDaughter(outerTubeLogic, &origin);

  auto const innerTube = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTube>(rmin, rmax, z, sphi, dphi);
  auto innerTubeLogic  = new vecgeom::LogicalVolume("InnerTube", innerTube);
  outerTubeLogic->PlaceDaughter(innerTubeLogic, &origin);

  vecgeom::GeoManager::Instance().SetWorldAndClose(worldPlaced);
}

void CreateConcatenatedTubes(double worldR, double worldZ, double rmin, double rmax, double z, double sphi, double dphi)
{
  auto worldSolid                     = new vecgeom::UnplacedBox(worldR, worldR, worldZ);
  auto worldLogic                     = new vecgeom::LogicalVolume("World", worldSolid);
  vecgeom::VPlacedVolume *worldPlaced = worldLogic->Place();

  vecgeom::Transformation3D lower{0, 0, -z};
  auto const lowerTube = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTube>(rmin, rmax, z, sphi, dphi);
  auto upperTubeLogic  = new vecgeom::LogicalVolume("LowerTube", lowerTube);
  worldLogic->PlaceDaughter(upperTubeLogic, &lower);

  vecgeom::Transformation3D upper{0, 0, z};
  auto const upperTube = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTube>(rmin, rmax, z, sphi, dphi);
  auto lowerTubeLogic  = new vecgeom::LogicalVolume("UpperTube", upperTube);
  worldLogic->PlaceDaughter(lowerTubeLogic, &upper);

  vecgeom::GeoManager::Instance().SetWorldAndClose(worldPlaced);
}

void CreateLayeredGeometry(double worldRadius, double worldZ, int NbOfLayers, int NbOfAbsorbers, double rmin,
                           double sphi, double dphi)
{
  const double CalorSizeR        = worldRadius;
  const double GapThickness      = 2.3;
  const double AbsorberThickness = 5.7;

  const double LayerThickness = GapThickness + AbsorberThickness;
  const double CalorThickness = NbOfLayers * LayerThickness;

  const double WorldSizeZ  = 1.2 * CalorThickness + 1000;
  const double WorldSizeXY = 1.2 * 2 * CalorSizeR + 1000;

  //
  // World
  //
  auto worldSolid = new vecgeom::UnplacedBox(0.5 * WorldSizeXY, 0.5 * WorldSizeXY, 0.5 * WorldSizeZ);
  auto worldLogic = new vecgeom::LogicalVolume("World", worldSolid);
  vecgeom::VPlacedVolume *worldPlaced = worldLogic->Place();

  //
  // Calorimeter
  //
  auto calorSolid =
      vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTube>(0, CalorSizeR, 0.5 * CalorThickness, 0, vecgeom::kTwoPi);
  auto calorLogic = new vecgeom::LogicalVolume("Calorimeter", calorSolid);
  vecgeom::Transformation3D origin;
  worldLogic->PlaceDaughter(calorLogic, &origin);

  //
  // Layers
  //
  auto layerSolid =
      vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTube>(rmin, CalorSizeR, 0.5 * LayerThickness, sphi, dphi);

  //
  // Absorbers
  //
  auto gapSolid =
      vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTube>(rmin, CalorSizeR, 0.5 * GapThickness, sphi, dphi);
  auto gapLogic = new vecgeom::LogicalVolume("Gap", gapSolid);
  vecgeom::Transformation3D gapPlacement(0, 0, -0.5 * LayerThickness + 0.5 * GapThickness); // Possible errors

  auto absorberSolid =
      vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTube>(rmin, CalorSizeR, 0.5 * AbsorberThickness, sphi, dphi);
  auto absorberLogic = new vecgeom::LogicalVolume("Absorber", absorberSolid);
  vecgeom::Transformation3D absorberPlacement(0, 0, 0.5 * LayerThickness - 0.5 * AbsorberThickness);

  // Create a new LogicalVolume per layer, we need unique IDs for scoring.
  double zCenter = -0.5 * CalorThickness + 0.5 * LayerThickness;
  for (int i = 0; i < NbOfLayers; i++) {
    std::string name("Layer_");
    name += std::to_string(i);
    auto layerLogic = new vecgeom::LogicalVolume("Layer", layerSolid);
    vecgeom::Transformation3D placement(0, 0, zCenter);
    calorLogic->PlaceDaughter(name.c_str(), layerLogic, &placement);

    layerLogic->PlaceDaughter(gapLogic, &gapPlacement);
    layerLogic->PlaceDaughter(absorberLogic, &absorberPlacement);

    zCenter += LayerThickness;
  }

  vecgeom::GeoManager::Instance().SetWorldAndClose(worldPlaced);
}

double PropagateRay(vecgeom::Vector3D<vecgeom::Precision> const &point,
                    vecgeom::Vector3D<vecgeom::Precision> const &direction,
                    vgbrep::SurfData<vecgeom::Precision> const &surfdata)
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

bool ValidateNavigation(int npoints, int nbLayers, vgbrep::SurfData<vecgeom::Precision> const &surfdata, double worldR,
                        double worldZ, double scale)
{
  constexpr double tolerance = 10 * vecgeom::kTolerance;

  int num_errors = 0;
  SOA3D<Precision> points(npoints);
  SOA3D<Precision> dirs(npoints);

  Vector3D<Precision> samplingVolume(worldR * scale, worldR * scale, worldZ * scale);
  vecgeom::volumeUtilities::FillRandomPoints(samplingVolume, points);
  vecgeom::volumeUtilities::FillRandomDirections(dirs);

  // now setup all the navigation states
  int ndeep = GeoManager::Instance().getMaxDepth();
  NavStatePool origStates(npoints, ndeep);
  NavStatePool outputStates(npoints, ndeep);

  vecgeom::Precision *refSteps = new Precision[npoints];
  memset(refSteps, 0, sizeof(Precision) * npoints);

  auto *nav = vecgeom::NewSimpleNavigator<>::Instance();
  for (int i = 0; i < npoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    Vector3D<Precision> const &dir = dirs[i];
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), pos, *origStates[i], true);
    nav->FindNextBoundaryAndStep(pos, dir, *origStates[i], *outputStates[i], vecgeom::kInfLength, refSteps[i]);

    // shoot the same ray in the surface model
    int exit_surf = 0;
    NavStateIndex out_state;
    auto distance = vgbrep::protonav::ComputeStepAndHit(pos, dir, *origStates[i], out_state, surfdata, exit_surf);
    if (out_state.GetNavIndex() != outputStates[i]->GetNavIndex() || std::abs(distance - refSteps[i]) > tolerance) {
      num_errors++;
      if (num_errors % 10 != 0) continue;
      std::cout << "ERROR " << i << std::endl << "POS:" << pos << std::endl << "DIR:" << dir << std::endl;
      if (out_state.GetNavIndex() != outputStates[i]->GetNavIndex())
        std::cout << "NAVINDEX MISMATCH: " << out_state.GetNavIndex() << " (new)" << std::endl
                  << "                   " << outputStates[i]->GetNavIndex() << " (old)" << std::endl;
      if (std::abs(distance - refSteps[i]) > tolerance)
        std::cout << "DIST MISMATCH: " << distance << " (new)" << std::endl
                  << "               " << refSteps[i] << " (old)" << std::endl;
    }
  }

  printf("=== Validation: num_erros = %d / %d\n", num_errors, npoints);
  delete[] refSteps;

  return num_errors == 0;
}

bool ShootOneParticle(double worldR, double worldZ, double px, double py, double pz, double dx, double dy, double dz,
                      vgbrep::SurfData<vecgeom::Precision> const &surfdata)
{
  // Very hacky, as I don't know how all the backend stuff works. -DC

  constexpr double tolerance = 10 * vecgeom::kTolerance;

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
  int exit_surf = 0;
  NavStateIndex out_state;
  auto distance = vgbrep::protonav::ComputeStepAndHit(point, direction, *origStates[0], out_state, surfdata, exit_surf);
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

void TestPerformance(double worldRadius, int npoints, int nbLayers,
                     vgbrep::SurfData<vecgeom::Precision> const &surfdata)
{
  const double CalorSizeR        = worldRadius;
  const double GapThickness      = 2.3;
  const double AbsorberThickness = 5.7;

  const double LayerThickness = GapThickness + AbsorberThickness;
  const double CalorThickness = nbLayers * LayerThickness;

  SOA3D<Precision> points(npoints);
  SOA3D<Precision> dirs(npoints);

  Vector3D<Precision> samplingVolume(0.5 * CalorSizeR, 0.5 * CalorSizeR, 0.5 * CalorThickness);
  vecgeom::volumeUtilities::FillRandomPoints(samplingVolume, points);
  vecgeom::volumeUtilities::FillRandomDirections(dirs);

  // now setup all the navigation states
  int ndeep = GeoManager::Instance().getMaxDepth();
  NavStatePool origStates(npoints, ndeep);
  NavStateIndex out_state;
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
    int exit_surf                  = 0;
    distance = vgbrep::protonav::ComputeStepAndHit(pos, dir, *origStates[i], out_state, surfdata, exit_surf);
  }
  Precision time_surf = timer1.Stop();

  printf("Time for %d points: NewSimpleNavigator = %f [s]  vgbrep::protonav = %f\n", npoints, time_prim, time_surf);
}



void TestAndSavePerformance(double worldRadius, int npoints, int nbLayers,
                     vgbrep::SurfData<vecgeom::Precision> const &surfdata)
{
  const double CalorSizeR        = worldRadius;
  const double GapThickness      = 2.3;
  const double AbsorberThickness = 5.7;

  const double LayerThickness = GapThickness + AbsorberThickness;
  const double CalorThickness = nbLayers * LayerThickness;

  SOA3D<Precision> points(npoints);
  SOA3D<Precision> dirs(npoints);

  Vector3D<Precision> samplingVolume(0.5 * CalorSizeR, 0.5 * CalorSizeR, 0.5 * CalorThickness);
  vecgeom::volumeUtilities::FillRandomPoints(samplingVolume, points);
  vecgeom::volumeUtilities::FillRandomDirections(dirs);

  // now setup all the navigation states
  int ndeep = GeoManager::Instance().getMaxDepth();
  NavStatePool origStates(npoints, ndeep);
  NavStateIndex out_state;
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
    int exit_surf                  = 0;
    distance = vgbrep::protonav::ComputeStepAndHit(pos, dir, *origStates[i], out_state, surfdata, exit_surf);
  }
  Precision time_surf = timer1.Stop();

  printf("Time for %d points: NewSimpleNavigator = %f [s]  vgbrep::protonav = %f\n", npoints, time_prim, time_surf);

  std::ofstream file_out;
  file_out.open("performance_measuring.txt", std::ios_base::app);
  file_out << nbLayers << " " << npoints << " " << time_prim <<" " << time_surf << std::endl;
  file_out.close();
  std::cout << "PRINTED TO FILE." << std::endl;
}