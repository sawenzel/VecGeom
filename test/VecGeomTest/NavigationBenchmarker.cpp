/// \file NavigationBenchmarker.h
/// \author Guilherme Lima (lima at fnal dot gov)
//
// 2014-11-26 G.Lima - created, by adapting Johannes' Benchmarker for navigation

#include "NavigationBenchmarker.h"

#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/navigation/NewSimpleNavigator.h"
#include "VecGeom/navigation/SimpleABBoxNavigator.h"

#ifdef VECGEOM_ROOT
#include "RootGeoManager.h"

#include "TGeoNavigator.h"
#include "TGeoManager.h"
#include "TGeoBranchArray.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4GeoManager.h"
#include "G4Navigator.hh"
#endif

#ifdef VECGEOM_CUDA_INTERFACE
#include "VecGeom/backend/cuda/Backend.h"
#include "VecGeom/backend/cuda/Interface.h"
#include "VecGeom/management/CudaManager.h"
#endif

#ifdef CALLGRIND_ENABLED
#include "valgrind/callgrind.h"
#endif

#include <vector>

namespace vecgeom {

// #ifdef VECGEOM_CUDA_INTERFACE
// void GetVolumePointers( std::list<DevicePtr<cuda::VPlacedVolume>> &volumesGpu ) {
//   using cxx::CudaManager;
//   CudaManager::Instance().LoadGeometry(CudaManager::Instance().world());
//   CudaManager::Instance().Synchronize();
//   for (std::list<VolumePointers>::const_iterator v = fVolumes.begin();
//        v != fVolumes.end(); ++v) {
//     volumesGpu.push_back(CudaManager::Instance().LookupPlaced(v->Specialized()));
//   }
// }
// #endif

//==================================

Precision benchmarkLocatePoint(int nPoints, int nReps, SOA3D<Precision> const &points)
{
  std::vector<NavigationState> states(nPoints);

  Stopwatch timer;
  timer.Start();
  for (int n = 0; n < nReps; ++n) {
    for (int i = 0; i < nPoints; ++i) {
      GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), points[i], states[i], true);
    }
  }
  Precision elapsed = timer.Stop();

  return (Precision)elapsed;
}

template <typename Navigator>
Precision benchmarkSerialSafety(int nPoints, int nReps, SOA3D<Precision> const &points)
{
  std::vector<NavigationState> curStates(nPoints);
  Precision *safety = new Precision[nPoints];

  for (int i = 0; i < nPoints; ++i) {
    curStates[i].Clear();
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), points[i], curStates[i], true);
  }

  const auto *safetyCalculator = Navigator::Instance()->GetSafetyEstimator();
  Stopwatch timer;
  timer.Start();
#ifdef CALLGRIND_ENABLED
  CALLGRIND_START_INSTRUMENTATION;
#endif
  for (int n = 0; n < nReps; ++n) {
    for (int i = 0; i < nPoints; ++i) {
      safety[i] = safetyCalculator->ComputeSafety(points[i], curStates[i]);
    }
  }
#ifdef CALLGRIND_ENABLED
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;
#endif
  Precision elapsed = timer.Stop();

  // cleanup
  delete[] safety;

  return (Precision)elapsed;
}

#ifdef VECGEOM_ROOT
Precision benchmarkROOTSafety(int nPoints, int nReps, SOA3D<Precision> const &points)
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
  for (int n = 0; n < nReps; ++n) {
    for (int i = 0; i < nPoints; ++i) {
      Vector3D<Precision> const &pos = points[i];
      brancharrays[i]->UpdateNavigator(rootnav);
      rootnav->SetCurrentPoint(pos.x(), pos.y(), pos.z());
      safety[i] = rootnav->Safety();
    }
  }
  timer.Stop();
#ifdef CALLGRIND_ENABLED
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;
#endif

  // cleanup
  delete[] safety;
  for (int i = 0; i < nPoints; ++i)
    TGeoBranchArray::ReleaseInstance(brancharrays[i]);

  return (Precision)timer.Elapsed();
}
#endif

//==================================

Precision benchmarkSerialNavigation(int nPoints, int nReps, SOA3D<Precision> const &points,
                                    SOA3D<Precision> const &dirs, Precision const *maxSteps)
{
  // setup all the navigation states
  std::vector<NavigationState> curStates(nPoints);
  std::vector<NavigationState> newStates(nPoints);

  for (int i = 0; i < nPoints; ++i) {
    curStates[i].Clear();
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), points[i], curStates[i], true);
  }

  auto *nav      = vecgeom::NewSimpleNavigator<>::Instance();
  Precision step = 0.0;
  Stopwatch timer;
  timer.Start();
  for (int n = 0; n < nReps; ++n) {
    for (int i = 0; i < nPoints; ++i) {
      nav->FindNextBoundaryAndStep(points[i], dirs[i], curStates[i], newStates[i], maxSteps[i], step);
    }
  }
  Precision elapsed = timer.Stop();

  return (Precision)elapsed;
}

//==================================
#ifdef VECGEOM_ROOT
Precision benchmarkROOTNavigation(int nPoints, int nReps, SOA3D<Precision> const &points, SOA3D<Precision> const &dirs,
                                  Precision const *maxSteps)
{

  TGeoNavigator *rootnav = ::gGeoManager->GetCurrentNavigator();
  TGeoNode **rootNodes   = new TGeoNode *[nPoints];

  Stopwatch timer;
  timer.Start();
  for (int n = 0; n < nReps; ++n) {
    for (int i = 0; i < nPoints; ++i) {
      Vector3D<Precision> const &pos = points[i];
      Vector3D<Precision> const &dir = dirs[i];

      rootnav->ResetState();
      rootNodes[i] = rootnav->FindNode(pos.x(), pos.y(), pos.z());

      rootnav->SetCurrentPoint(pos.x(), pos.y(), pos.z());
      rootnav->SetCurrentDirection(dir.x(), dir.y(), dir.z());
      rootnav->FindNextBoundaryAndStep(maxSteps[i]);
    }
  }

  // cleanup
  delete[] rootNodes;

  return (Precision)timer.Stop();
}
#endif

//==================================
#ifdef VECGEOM_GEANT4
Precision benchmarkGeant4Navigation(int nPoints, int nReps, SOA3D<Precision> const &points,
                                    SOA3D<Precision> const &dirs, Precision const *maxSteps)
{
  Stopwatch timer;
  timer.Start();

  // Note: Vector3D's are expressed in cm, while G4ThreeVectors are expressed in mm
  const Precision cm             = 10.; // cm --> mm conversion
  G4Navigator &g4nav             = *(G4GeoManager::Instance().GetNavigator());
  G4TouchableHistory **g4history = new G4TouchableHistory *[nPoints];

  for (int n = 0; n < nReps; ++n) {
    for (int i = 0; i < nPoints; ++i) {
      G4ThreeVector g4pos(points[i].x() * cm, points[i].y() * cm, points[i].z() * cm);
      G4ThreeVector g4dir(dirs[i].x(), dirs[i].y(), dirs[i].z());
      G4double maxStep = maxSteps[i];

      // false --> locate from top
      G4VPhysicalVolume const *vol = g4nav.LocateGlobalPointAndSetup(g4pos, &g4dir, false);
      if (!vol) std::cout << "benchG4Navit: pos=" << g4pos << " and dir=" << g4dir << " --> vol=" << vol << "\n";

      G4double safety = 0.0;
      G4double step   = g4nav.ComputeStep(g4pos, g4dir, maxStep, safety);
      if (step > 9.0e+98) step = maxStep;

      G4ThreeVector nextPos = g4pos + (step + 1.0e-6) * g4dir;

      // TODO: save touchable history array - returnable?  symmetrize with root benchmark

      // Technically, vecgeom returns vol at nextPos, but we might remove it here so Geant4 gets a performance boost
      // g4nav.SetGeometricallyLimitedStep();
      // vol = g4nav.LocateGlobalPointAndSetup( nextPos );
    }
  }

  // cleanup
  delete[] g4history;
  // vecCore::AlignedFree(maxSteps);

  return (Precision)timer.Stop();
}
#endif

//=======================================
/// Function to run navigation benchmarks
void runNavigationBenchmarks(LogicalVolume const *startVol, int np, int nreps, Precision const *maxStep, Precision bias)
{

  SOA3D<Precision> points(np);
  SOA3D<Precision> locpts(np);
  SOA3D<Precision> dirs(np);
  vecgeom::volumeUtilities::FillGlobalPointsAndDirectionsForLogicalVolume(startVol, locpts, points, dirs, bias, np);

  Precision cputime;

  cputime = benchmarkLocatePoint(np, nreps, points);
  printf("CPU elapsed time: %10.6f ms for locating and setting steps\n", 1000. * cputime);

  //*** safety benchmarking

  // scalar
  cputime = benchmarkSerialSafety<vecgeom::NewSimpleNavigator<>>(np, nreps, points);
  printf("CPU elapsed time: %10.6f ms for serialSafety<NewSimpleNavigator<>>\n", 1000. * cputime);

  cputime = benchmarkSerialSafety<vecgeom::SimpleABBoxNavigator<>>(np, nreps, points);
  printf("CPU elapsed time: %10.6f ms for serialSafety<SimpleABBoxNavigator<>>\n", 1000. * cputime);

#ifdef VECGEOM_ROOT
  // ROOT safety
  cputime = benchmarkROOTSafety(np, nreps, points);
  printf("CPU elapsed time: %10.6f ms for ROOT\n", 1000. * cputime);
#endif

  //**** navigation benchmarking

  cputime = benchmarkSerialNavigation(np, nreps, points, dirs, maxStep);
  printf("\nCPU elapsed time: %10.6f ms for serialized navigation\n", 1000. * cputime);

#ifdef VECGEOM_ROOT
  cputime = benchmarkROOTNavigation(np, nreps, points, dirs, maxStep);
  printf("CPU elapsed time: %10.6f ms for ROOT navigation\n", 1000. * cputime);
#endif

#ifdef VECGEOM_GEANT4
  cputime = benchmarkGeant4Navigation(np, nreps, points, dirs, maxStep);
  printf("CPU elapsed time: %10.6f ms for Geant4 navigation\n", 1000. * cputime);
#endif

  return;
}

//======================================
// Use ROOT as reference to validate VecGeom navigation.
// The procedure is appropriate for one track at a time (serial
// interface), no need to store ROOT results.  Takes as input one
// track (position+direction) and VecGeom navigation output (step and
// navState).
#ifdef VECGEOM_ROOT
bool validateNavigationStepAgainstRoot(Vector3D<Precision> const &pos, Vector3D<Precision> const &dir,
                                       Precision maxStep, Precision testStep, NavigationState const &testState)
{
  bool result = true;

  TGeoNavigator *rootnav = ::gGeoManager->GetCurrentNavigator();
  rootnav->ResetState();
  rootnav->FindNode(pos.x(), pos.y(), pos.z());
  rootnav->SetCurrentPoint(pos.x(), pos.y(), pos.z());
  rootnav->SetCurrentDirection(dir.x(), dir.y(), dir.z());
  rootnav->FindNextBoundaryAndStep(maxStep);

  const char *vgname   = testState.Top() ? testState.Top()->GetName() : "NULL";
  const char *rtname   = rootnav->GetCurrentNode()->GetName();
  static int maxReport = 0;
  if (maxReport < 10) {
    if (strcmp(vgname, rtname)) {
      std::cerr << "validateAgainstROOT: pos=" << pos << " -> vol name mismatch: VGname=<" << vgname << ">, ROOTname=<"
                << rtname << ">\n";
      maxReport++;
      if (maxReport == 10) std::cerr << "validateAgainstROOT: more mismatches detected, but further reports dropped!\n";
    }
  }
  if (testState.Top() == NULL) {
    if (!rootnav->IsOutside()) {
      result = false;
      std::cerr << " OUTSIDEERROR \n";
    }
  }

  else if (Abs(testStep - rootnav->GetStep()) > 5. * kTolerance ||
           rootnav->GetCurrentNode() != RootGeoManager::Instance().tgeonode(testState.Top())) {
    result = false;
    std::cerr << "\n*** ERROR on validateAgainstROOT: " << " ROOT node=" << rootnav->GetCurrentNode()->GetName()
              << " outside=" << rootnav->IsOutside() << " step=" << rootnav->GetStep()
              << " <==> VecGeom node=" << (testState.Top() ? testState.Top()->GetLabel() : "NULL")
              << " step=" << testStep << " /// Step ratio=" << testStep / rootnav->GetStep()
              << " / step diff=" << Abs(testStep - rootnav->GetStep())
              << " / rel.error=" << Abs(testStep - rootnav->GetStep()) / testStep << " / tolerance=" << 5. * kTolerance
              << "\n";

    std::cerr << rootnav->GetCurrentNode() << ' ' << RootGeoManager::Instance().tgeonode(testState.Top()) << "\n";
  }

  return result;
}
#endif // VECGEOM_ROOT

//=======================================
// Use Geant4 as reference to validate VecGeom navigation.
//
// The procedure is appropriate for one track at a time (serial interface),
// no need to store Geant4 results.
//
// Takes as input one track (position+direction)
// and VecGeom navigation output (step and navState).
#ifdef VECGEOM_GEANT4
bool validateNavigationStepAgainstGeant4(Vector3D<Precision> const &pos, Vector3D<Precision> const &dir,
                                         Precision maxStep, Precision testStep, NavigationState const &testState,
                                         Precision &step, G4VPhysicalVolume const *&nextVol)
{
  // Note: Vector3D's are expressed in cm, while G4ThreeVectors are expressed in mm
  const Precision cm = 10.; // cm --> mm conversion
  bool result        = true;
  G4Navigator &g4nav = *(G4GeoManager::Instance().GetNavigator());

  G4ThreeVector g4pos(pos.x() * cm, pos.y() * cm, pos.z() * cm);
  G4ThreeVector g4dir(dir.x(), dir.y(), dir.z());
  // false == locate from top
  nextVol = g4nav.LocateGlobalPointAndSetup(g4pos, &g4dir, false, false);

  G4double safety = 0.0;
  step            = g4nav.ComputeStep(g4pos, g4dir, maxStep * cm, safety);
  // note that if maxStep limitation is taken, step actually returns 9+e98 (Geant4 kInfinity)
  if (step > 9e+97)
    step = maxStep * cm;
  else
    g4nav.SetGeometricallyLimitedStep();

  G4ThreeVector nextPos = g4pos + (step + 1.0e-6) * g4dir;
  nextVol               = g4nav.LocateGlobalPointAndSetup(nextPos, &g4dir, true, false);

  std::string vgLogName(testState.Top()->GetLogicalVolume()->GetLabel());

  if (testState.Top() == NULL) {
    if (!g4nav.ExitedMotherVolume()) {
      result = false;
      std::cerr << " OUTSIDEERROR \n";
    }
  } else if (Abs(testStep - step / cm) > 5. * kTolerance || vgLogName.compare(nextVol->GetName())) {
    result = false;
    std::cerr << "\n*** ERROR on validateAgainstGeant4: " << " Geant4 node="
              << (nextVol ? nextVol->GetName() : "Null")
              //             <<" outside="<< g4nav->IsOutside()
              << " step=" << step / cm // printouts are in cm units
              << " <==> VecGeom node=" << (testState.Top() ? testState.Top()->GetLabel() : "NULL")
              << " step=" << testStep << "\n";

    // std::cerr<< vol <<' '<< RootGeoManager::Instance().tgeonode(testState.Top()) << "\n";
  }

  return result;
}
#endif // VECGEOM_GEANT4

//=======================================

bool validateVecGeomNavigation(int np, SOA3D<Precision> const &points, SOA3D<Precision> const &dirs,
                               Precision const *maxSteps)
{
  bool result = true;

  // now setup all the navigation states
  std::vector<NavigationState> origStates(np);
  std::vector<NavigationState> vgSerialStates(np);

  Precision *refSteps = (Precision *)vecCore::AlignedAlloc(32, sizeof(Precision) * np);
  memset(refSteps, 0, sizeof(Precision) * np);

  // navigation using the serial interface
#ifdef VECGEOM_ROOT
  int rootMismatches = 0;
#endif
#ifdef VECGEOM_GEANT4
  int g4Mismatches = 0;
#endif
  auto *nav = vecgeom::NewSimpleNavigator<>::Instance();
  for (int i = 0; i < np; ++i) {
    Vector3D<Precision> const &pos = points[i];
    Vector3D<Precision> const &dir = dirs[i];
    GlobalLocator::LocateGlobalPoint(GeoManager::Instance().GetWorld(), pos, origStates[i], true);
    nav->FindNextBoundaryAndStep(pos, dir, origStates[i], vgSerialStates[i], maxSteps[i], refSteps[i]);

    // validate serial interface against ROOT and/or Geant4
    bool ok = true;

#ifdef VECGEOM_ROOT
    ok = validateNavigationStepAgainstRoot(pos, dir, maxSteps[i], refSteps[i], vgSerialStates[i]);
    result &= ok;
    if (!ok) {
      ++rootMismatches;
    }
#endif

#ifdef VECGEOM_GEANT4
    G4double g4step                 = ::kInfinity;
    G4VPhysicalVolume const *nextPV = NULL;
    ok = validateNavigationStepAgainstGeant4(pos, dir, maxSteps[i], refSteps[i], vgSerialStates[i], g4step, nextPV);
    result &= ok;
    if (!ok) ++g4Mismatches;
#endif

    result &= ok;
    if (!ok) {
      std::cout << "\n=======> Summary: ITERATION " << i << " - pos = " << pos << " dir = " << dir << " / Steps (";
#ifdef VECGEOM_ROOT
      std::cout << "Root/";
#endif
#ifdef VECGEOM_GEANT4
      std::cout << "Geant4/";
#endif
      std::cout << "VecGeom): ";

#ifdef VECGEOM_ROOT
      TGeoNavigator *rootnav = ::gGeoManager->GetCurrentNavigator();
      std::cout << rootnav->GetStep() << " / ";
#endif
#ifdef VECGEOM_GEANT4
      std::cout << 0.1 * g4step << " / ";
#endif
      std::cout << refSteps[i] << "\n";

      //=== compare navigation states
      std::cout << "Next volumes: ";
#ifdef VECGEOM_ROOT
      std::cout << rootnav->GetCurrentNode()->GetName() << " / ";
#endif
#ifdef VECGEOM_GEANT4
      std::cout << (nextPV ? nextPV->GetName() : "NULL") << " / ";
#endif
      std::cout << (vgSerialStates[i].Top() ? vgSerialStates[i].Top()->GetLabel() : "NULL") << "\n";

      // nav.InspectEnvironmentForPointAndDirection( pos, dir, *origState );
    }
  }
#ifdef VECGEOM_ROOT
  std::cout << "VecGeom navigation - serial interface: # ROOT mismatches (step lengths) = " << rootMismatches << " / "
            << np << "\n";
#endif
#ifdef VECGEOM_GEANT4
  std::cout << "VecGeom navigation - serial interface: # Geant4 mismatches = " << g4Mismatches << " / " << np << "\n";
#endif

#ifdef VECGEOM_ENABLE_CUDA
  Precision *gpuSteps = (Precision *)vecCore::AlignedAlloc(32, np * sizeof(Precision));
  std::vector<NavigationState> gpuStates(np);

  // load GPU geometry
  CudaManager::Instance().set_verbose(0);
  CudaManager::Instance().LoadGeometry(GeoManager::Instance().GetWorld());
  CudaManager::Instance().Synchronize();

  auto *origStatesGpu = cxx::AllocateOnGpu<NavigationState>(np * sizeof(NavigationState));
  auto *gpuStatesGpu  = cxx::AllocateOnGpu<NavigationState>(np * sizeof(NavigationState));
  cxx::CopyToGpu(origStates.data(), origStatesGpu, np * sizeof(NavigationState));
  cxx::CopyToGpu(gpuStates.data(), gpuStatesGpu, np * sizeof(NavigationState));

  printf("Start validating GPU navigation...\n");
  runNavigationCuda(origStatesGpu, gpuStatesGpu, GeoManager::Instance().GetWorld(), np, points.x(), points.y(),
                    points.z(), dirs.x(), dirs.y(), dirs.z(), maxSteps, gpuSteps);

  cxx::CopyFromGpu(gpuStatesGpu, gpuStates.data(), np * sizeof(NavigationState));

  //*** Comparing results from GPU against serialized navigation
  // TODO: move checks into a separate function, like e.g.:
  //  ok = compareNavigationResults(refSteps, vgSerialStates, gpuSteps, gpuStates);
  int errorCountGpu = 0;
  for (int i = 0; i < np; ++i) {
    bool mismatch = false;
    if (Abs(gpuSteps[i] - refSteps[i]) > 5. * kTolerance) mismatch = true;
    if (gpuStates[i].Top() != vgSerialStates[i].Top()) mismatch = true;
    if (gpuStates[i].IsOnBoundary() != vgSerialStates[i].IsOnBoundary()) mismatch = true;
    // if( safeties[i] != nav.GetSafety( points[i], *origStates[i] ))         mismatch = true;
    if (mismatch) {
      result = false;
      ++errorCountGpu;
      std::cout << "GPU navigation mismatches: track[" << i << "]=(" << points[i].x() << "; " << points[i].y() << "; "
                << points[i].z() << ") " << " steps: " << refSteps[i] << " / " << gpuSteps[i]
                << " navStates: " << vgSerialStates[i].Top()->GetLabel()
                << (vgSerialStates[i].IsOnBoundary() ? "*" : "") << " / " << gpuStates[i].Top()->GetLabel()
                << (gpuStates[i].IsOnBoundary() ? "*" : "") << "\n";
    }
  }

  std::cout << "VecGeom navigation on the GPUs: #mismatches = " << errorCountGpu << " / " << np << "\n";
  cxx::FreeFromGpu(origStatesGpu);
  cxx::FreeFromGpu(gpuStatesGpu);
#endif // VECGEOM_ENABLE_CUDA

  // if(mismatches>0) std::cout << "Navigation test failed with "<< mismatches <<" mismatches\n";
  // else std::cout<<"Navigation test passed.\n";

  //=== cleanup
  if (refSteps) vecCore::AlignedFree(refSteps);
  // if (vecSteps) vecCore::AlignedFree(vecSteps);
  // if (safeties) vecCore::AlignedFree(safeties);
#ifdef VECCORE_CUDA
  if (gpuSteps) vecCore::AlignedFree(gpuSteps);
#endif
  // delete[] vgVectorStates;

  return result;
}

} // End namespace vecgeom
