/*
 * SafetyKernelBenchmarker
 *
 *  Created on: 18.11.2015
 *      Author: swenzel
 */

// benchmarking various different safety kernels per logical volume
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/RNG.h"
#include "VecGeom/navigation/GlobalLocator.h"
#include "VecGeom/navigation/NavStatePool.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/volumes/PlacedVolume.h"

#include "VecGeomTest/RootGeoManager.h"

#include "VecGeom/management/GeoManager.h"
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/navigation/SimpleSafetyEstimator.h"
#include "VecGeom/navigation/SimpleABBoxSafetyEstimator.h"
#include "VecGeom/navigation/HybridSafetyEstimator.h"
#include "VecGeom/management/HybridManager2.h"
#include "VecGeom/navigation/VoxelSafetyEstimator.h"
#include "VecGeom/management/FlatVoxelManager.h"

// in case someone has written a special safety estimator for the CAHL logical volume
// #include "VecGeom/navigation/CAHLSafetyEstimator.h"
// #define SPECIALESTIMATOR CAHLSafetyEstimator

#ifdef VECGEOM_ROOT
#include "TGeoNavigator.h"
#include "TGeoNode.h"
#include "TGeoManager.h"
#include "TGeoBranchArray.h"
#include "TGeoBBox.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Navigator.hh"
#include "G4VPhysicalVolume.hh"
#include "G4ThreeVector.hh"
#include "VecGeomTest/G4GeoManager.h"
#include "G4VoxelNavigation.hh"
#endif

#include <iostream>
#include <vector>

// #define CALLGRIND_ENABLED
#ifdef CALLGRIND_ENABLED
#include <valgrind/callgrind.h>
#endif

#ifdef CALLGRIND_ENABLED
#define RUNBENCH(NAME)             \
  CALLGRIND_START_INSTRUMENTATION; \
  NAME;                            \
  CALLGRIND_STOP_INSTRUMENTATION;  \
  CALLGRIND_DUMP_STATS
#else
#define RUNBENCH(NAME) NAME
#endif

using namespace vecgeom;

template <typename T>
__attribute__((noinline)) void benchSafety(SOA3D<Precision> const &points, NavStatePool &pool)
{
  // bench safety
  Precision *safety = new Precision[points.size()];
  Stopwatch timer;
  VSafetyEstimator *se = T::Instance();
  timer.Start();
  for (size_t i = 0; i < points.size(); ++i) {
    safety[i] = se->ComputeSafety(points[i], *(pool[i]));
  }
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  double accum(0.);
  for (size_t i = 0; i < points.size(); ++i) {
    accum += safety[i];
  }
  delete[] safety;
  std::cerr << "accum  " << T::GetClassName() << " " << accum << "\n";
}

template <typename T>
__attribute__((noinline)) void benchLocalSafety(SOA3D<Precision> const &localpoints, NavStatePool &pool)
{
  // bench safety
  Precision *safety = new Precision[localpoints.size()];
  Stopwatch timer;
  VSafetyEstimator *se = T::Instance();
  const auto topvolume = pool[0]->Top();
  timer.Start();
  for (size_t i = 0; i < localpoints.size(); ++i) {
    safety[i] = se->ComputeSafetyForLocalPoint(localpoints[i], topvolume);
  }
  timer.Stop();
  std::cerr << timer.Elapsed() << "\n";
  double accum(0.);
  for (size_t i = 0; i < localpoints.size(); ++i) {
    accum += safety[i];
  }
  delete[] safety;
  std::cerr << "accum  " << T::GetClassName() << " " << accum << "\n";
}

#ifdef VECGEOM_GEANT4
__attribute__((noinline)) void benchmarkLocalG4Safety(SOA3D<Precision> const &points,
                                                      SOA3D<Precision> const &localpoints)
{
  Stopwatch timer;
  G4VPhysicalVolume *world(vecgeom::G4GeoManager::Instance().GetG4GeometryFromROOT());
  if (world != nullptr) G4GeoManager::Instance().LoadG4Geometry(world);

  // Note: Vector3D's are expressed in cm, while G4ThreeVectors are expressed in mm
  const Precision cm = 10.; // cm --> mm conversion
  G4Navigator &g4nav = *(G4GeoManager::Instance().GetNavigator());

  Precision *safety = new Precision[points.size()];
  G4VoxelNavigation g4voxelnavigator;

  // we need one navigation history object for the points; taking the first point is enough
  // as all of them are in the same toplevel volume
  G4ThreeVector g4pos(points[0].x() * cm, points[0].y() * cm, points[0].z() * cm);
  g4nav.LocateGlobalPointAndSetup(g4pos, nullptr, false);
  auto history = g4nav.CreateTouchableHistory()->GetHistory();
  std::cerr << history << "\n";
  std::cerr << history->GetDepth() << "\n";

#ifdef CALLGRIND_ENABLED
  CALLGRIND_START_INSTRUMENTATION;
#endif
  timer.Start();
  for (decltype(points.size()) i = 0; i < points.size(); ++i) {
    const auto &vgpoint = points[i];
    const G4ThreeVector g4lpos(vgpoint.x() * cm, vgpoint.y() * cm, vgpoint.z() * cm);
    safety[i] = g4nav.ComputeSafety(g4lpos); // *history, 1E20);
  }
  timer.Stop();
  std::cerr << (Precision)timer.Elapsed() << "\n";
#ifdef CALLGRIND_ENABLED
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;
#endif
  double accum(0.);
  for (size_t i = 0; i < localpoints.size(); ++i) {
    accum += safety[i];
  }
  // cleanup
  delete[] safety;
  std::cerr << "accum  G4 " << accum / cm << "\n";
}
#endif // end if G4

// benchmarks the ROOT safety interface
#ifdef VECGEOM_ROOT
__attribute__((noinline)) void benchmarkROOTSafety(int nPoints, SOA3D<Precision> const &points)
{
  TGeoNavigator *rootnav = ::gGeoManager->GetCurrentNavigator();
  std::vector<TGeoBranchArray *> brancharrays(nPoints);
  Precision *safety = new Precision[nPoints];

  for (int i = 0; i < nPoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    rootnav->ResetState();
    rootnav->FindNode(pos.x(), pos.y(), pos.z());
    brancharrays[i] = TGeoBranchArray::MakeInstance(GeoManager::Instance().getMaxDepth());
    brancharrays[i]->InitFromNavigator(rootnav);
  }

#ifdef CALLGRIND_ENABLED
  CALLGRIND_START_INSTRUMENTATION;
#endif
  Stopwatch timer;
  timer.Start();
  for (int i = 0; i < nPoints; ++i) {
    Vector3D<Precision> const &pos = points[i];
    brancharrays[i]->UpdateNavigator(rootnav);
    rootnav->SetCurrentPoint(pos.x(), pos.y(), pos.z());
    safety[i] = rootnav->Safety();
  }
  timer.Stop();
  std::cerr << "ROOT time" << timer.Elapsed() << "\n";
#ifdef CALLGRIND_ENABLED
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;
#endif
  double accum(0.);
  for (int i = 0; i < nPoints; ++i) {
    accum += safety[i];
  }
  std::cerr << "ROOT s " << accum << "\n";
  delete[] safety;
  return;
}

__attribute__((noinline)) void benchmarkLocalROOTSafety(int nPoints, SOA3D<Precision> const &points,
                                                        SOA3D<Precision> const &localpoints)
{
  // we init the ROOT navigator to a specific branch state
  auto nav = gGeoManager->GetCurrentNavigator();
  nav->FindNode(points[0].x(), points[0].y(), points[0].z());
  Precision *safety = new Precision[nPoints];

#ifdef CALLGRIND_ENABLED
  CALLGRIND_START_INSTRUMENTATION;
#endif
  Stopwatch timer;
  timer.Start();
  for (int i = 0; i < nPoints; ++i) {
    // There is no direct way of calling the local safety function; points are always transformed so I have to
    // give the global point
    Vector3D<Precision> const &pos = points[i];
    nav->SetCurrentPoint(pos.x(), pos.y(), pos.z());
    safety[i] = nav->Safety();
  }
  timer.Stop();
  std::cerr << "ROOT time" << timer.Elapsed() << "\n";
#ifdef CALLGRIND_ENABLED
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;
#endif
  double accum(0.);
  for (int i = 0; i < nPoints; ++i) {
    accum += safety[i];
  }
  std::cerr << "ROOT s " << accum << "\n";
  delete[] safety;
  return;
}
#endif

// main routine starting up the individual benchmarks
void benchDifferentSafeties(SOA3D<Precision> const &points, SOA3D<Precision> const &localpoints, NavStatePool &pool)
{
  std::cerr << "## - GLOBAL POINTS - \n";
  std::cerr << "##\n";
  RUNBENCH(benchSafety<SimpleSafetyEstimator>(points, pool));
  std::cerr << "##\n";
  RUNBENCH(benchSafety<SimpleABBoxSafetyEstimator>(points, pool));
  std::cerr << "##\n";
  RUNBENCH(benchSafety<HybridSafetyEstimator>(points, pool));
  std::cerr << "##\n";
  RUNBENCH(benchSafety<VoxelSafetyEstimator>(points, pool));
  std::cerr << "##\n";
  RUNBENCH(benchmarkROOTSafety(points.size(), points));
  // std::cerr << "##\n";
  // benchVectorSafety<SPECIALESTIMATOR>(points, pool);
  std::cerr << "## - LOCAL POINTS - \n";
  std::cerr << "##\n";
  RUNBENCH(benchLocalSafety<SimpleSafetyEstimator>(localpoints, pool));
  std::cerr << "##\n";
  RUNBENCH(benchLocalSafety<SimpleABBoxSafetyEstimator>(localpoints, pool));
  std::cerr << "##\n";
  RUNBENCH(benchLocalSafety<HybridSafetyEstimator>(localpoints, pool));
  std::cerr << "##\n";
  RUNBENCH(benchLocalSafety<VoxelSafetyEstimator>(localpoints, pool));
  std::cerr << "##\n";
  RUNBENCH(benchmarkLocalROOTSafety(points.size(), points, localpoints));
#ifdef VECGEOM_GEANT4
  std::cerr << "##\n";
  RUNBENCH(benchmarkLocalG4Safety(points, localpoints));
#endif

  std::cerr << "##\n";
}

// main program
int main(int argc, char *argv[])
{
  // read in detector passed as argument
  if (argc > 1) {
    RootGeoManager::Instance().set_verbose(3);
    RootGeoManager::Instance().LoadRootGeometry(std::string(argv[1]));
  } else {
    std::cerr << "please give a ROOT geometry file\n";
    return 1;
  }

  // setup data structures
  int npoints = 100000;
  SOA3D<Precision> points(npoints);
  SOA3D<Precision> localpoints(npoints);
  SOA3D<Precision> directions(npoints);
  NavStatePool statepool(npoints, GeoManager::Instance().getMaxDepth());

  // setup test points
  TGeoBBox const *rootbbox = dynamic_cast<TGeoBBox const *>(gGeoManager->GetTopVolume()->GetShape());
  Vector3D<Precision> bbox(rootbbox->GetDX(), rootbbox->GetDY(), rootbbox->GetDZ());

  std::string volname(argv[2]);
  volumeUtilities::FillGlobalPointsForLogicalVolume<SOA3D<Precision>>(
      GeoManager::Instance().FindLogicalVolume(volname.c_str()), localpoints, points, npoints);
  std::cerr << "points filled\n";
  for (size_t i = 0; i < points.size(); ++i) {
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), points[i], *(statepool[i]), true);
    if (statepool[i]->Top()->GetLogicalVolume() != GeoManager::Instance().FindLogicalVolume(volname.c_str())) {
      //
      std::cerr << "problem : point " << i << " probably in overlapping region \n";
      points.set(i, points[i - 1]);
      statepool[i - 1]->CopyTo(statepool[i]);
    }
  }

  HybridManager2::Instance().InitStructure(GeoManager::Instance().FindLogicalVolume(volname.c_str()));
  FlatVoxelManager::Instance().InitStructure(GeoManager::Instance().FindLogicalVolume(volname.c_str()));

  std::cerr << "located ...\n";
  benchDifferentSafeties(points, localpoints, statepool);
  return 0;
}
