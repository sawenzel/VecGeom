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

bool ValidateNavigation(int npoints, vgbrep::SurfData<vecgeom::Precision> const &surfdata, double worldX, double worldY,
                        double worldZ, double scale)
{
  constexpr double tolerance = 10 * vecgeom::kTolerance;

  int num_errors = 0;
  SOA3D<Precision> points(npoints);
  SOA3D<Precision> dirs(npoints);

  Vector3D<Precision> samplingVolume(worldX * scale, worldY * scale, worldZ * scale);
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

bool ShootOneParticle(double px, double py, double pz, double dx, double dy, double dz,
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

void TestPerformance(double worldX, double worldY, double worldZ, double scale,
                     int npoints, int nbLayers,
                     vgbrep::SurfData<vecgeom::Precision> const &surfdata)
{
  SOA3D<Precision> points(npoints);
  SOA3D<Precision> dirs(npoints);

  Vector3D<Precision> samplingVolume(scale * worldX, scale*worldY, scale*worldZ);
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


// Not updated
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