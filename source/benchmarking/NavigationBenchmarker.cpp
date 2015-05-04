/// \file NavigationBenchmarker.h
/// \author Guilherme Lima (lima at fnal dot gov)
//
// 2014-11-26 G.Lima - created, by adapting Johannes' Benchmarker for navigation

#include "benchmarking/NavigationBenchmarker.h"

#include "base/SOA3D.h"
#include "base/Stopwatch.h"
#include "volumes/utilities/VolumeUtilities.h"

#include "volumes/PlacedVolume.h"
#include "navigation/SimpleNavigator.h"
#include "navigation/NavStatePool.h"

#ifdef VECGEOM_ROOT
  #include "TGeoNavigator.h"
  #include "TGeoManager.h"
#endif

#ifdef VECGEOM_GEANT4
  #include "management/G4GeoManager.h"
  #include "G4Navigator.hh"
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  #include "backend/cuda/Backend.h"
  #include "management/CudaManager.h"
#endif

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

Precision benchmarkLocatePoint( int nPoints, int nReps, SOA3D<Precision> const& points) {

  NavigationState * state = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );
  vecgeom::SimpleNavigator nav;

  Stopwatch timer;
  timer.Start();
  for(int n=0; n<nReps; ++n) {
    for( int i=0; i<nPoints; ++i ) {
      state->Clear();
      nav.LocatePoint( GeoManager::Instance().GetWorld(), points[i], *state, true);
    }
  }
  Precision elapsed = timer.Stop();

  NavigationState::ReleaseInstance( state );
  return (Precision)elapsed;
}

//==================================

Precision benchmarkSerialNavigation( int nPoints, int nReps,
                                     SOA3D<Precision> const& points,
                                     SOA3D<Precision> const& dirs ) {

  NavigationState ** curStates = new NavigationState*[nPoints];
  for( int i=0; i<nPoints; ++i) curStates[i] = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );

  NavigationState ** newStates = new NavigationState*[nPoints];
  for( int i=0; i<nPoints; ++i) newStates[i] = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );

  vecgeom::SimpleNavigator nav;

  for( int i=0; i<nPoints; ++i ) {
    curStates[i]->Clear();
    nav.LocatePoint( GeoManager::Instance().GetWorld(), points[i], *curStates[i], true);
  }

  Precision maxStep = kInfinity, step=0.0;
  Stopwatch timer;
  timer.Start();
  for(int n=0; n<nReps; ++n) {
    for( int i=0; i<nPoints; ++i ) {
         nav.FindNextBoundaryAndStep( points[i], dirs[i], *(curStates[i]),
                                      *(newStates[i]), maxStep, step );
    }
  }
  Precision elapsed = timer.Stop();

  // cleanup
  for( int i=0; i<nPoints; ++i) {
    NavigationState::ReleaseInstance( curStates[i] );
    NavigationState::ReleaseInstance( newStates[i] );
  }
  delete[] curStates;
  delete[] newStates;

  return (Precision)elapsed;
}

//==================================

Precision benchmarkVectorNavigation( int nPoints, int nReps,
                                     SOA3D<Precision> const& points,
                                     SOA3D<Precision> const& dirs ) {

  SOA3D<Precision> workspace1(nPoints);
  SOA3D<Precision> workspace2(nPoints);

  int * intworkspace = (int *) _mm_malloc(sizeof(int)*nPoints,32);
  Precision * maxSteps = (Precision *) _mm_malloc(sizeof(Precision)*nPoints,32);
  Precision * vecSteps = (Precision *) _mm_malloc(sizeof(Precision)*nPoints,32);
  Precision * safeties = (Precision *) _mm_malloc(sizeof(Precision)*nPoints,32);

  for (int i=0;i<nPoints;++i) maxSteps[i] = kInfinity;
  memset(vecSteps, 0, sizeof(Precision)*nPoints);
  memset(safeties, 0, sizeof(Precision)*nPoints);

  vecgeom::SimpleNavigator nav;

  // setup all the navigation states
  NavigationState ** curStates = new NavigationState*[nPoints];
  NavigationState ** newStates = new NavigationState*[nPoints];

  for( int i=0; i<nPoints; ++i ) {
    curStates[i] = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );
    newStates[i] = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );
    nav.LocatePoint( GeoManager::Instance().GetWorld(), points[i], *curStates[i], true);
  }

  Stopwatch timer;
  timer.Start();
  for(int n=0; n<nReps; ++n) {
    nav.FindNextBoundaryAndStep( points, dirs, workspace1, workspace2, curStates, newStates,
                                 maxSteps, safeties, vecSteps, intworkspace );
  }
  Precision elapsed = timer.Stop();

  // cleanup
  for( int i=0; i<nPoints; ++i) {
    NavigationState::ReleaseInstance( curStates[i] );
    NavigationState::ReleaseInstance( newStates[i] );
  }
  delete[] curStates;
  delete[] newStates;

  _mm_free(intworkspace);
  _mm_free(maxSteps);
  _mm_free(vecSteps);
  _mm_free(safeties);

  return (Precision)elapsed;
}

//==================================
#ifdef VECGEOM_ROOT
Precision benchmarkROOTNavigation( int nPoints, int nReps,
                                   SOA3D<Precision> const& points,
                                   SOA3D<Precision> const& dirs ) {

  TGeoNavigator * rootnav = ::gGeoManager->GetCurrentNavigator();
  TGeoNode ** rootNodes = new TGeoNode*[nPoints];

  Precision * maxSteps = (Precision *) _mm_malloc(sizeof(Precision)*nPoints,32);
  for (int i=0;i<nPoints;++i) maxSteps[i] = kInfinity;

  Stopwatch timer;
  timer.Start();
  for(int n=0; n<nReps; ++n) {
    for (int i=0;i<nPoints;++i) {
      Vector3D<Precision> const& pos = points[i];
      Vector3D<Precision> const& dir = dirs[i];

      rootnav->ResetState();
      rootNodes[i] = rootnav->FindNode( pos.x(), pos.y(), pos.z() );

      rootnav->SetCurrentPoint( pos.x(), pos.y(), pos.z() );
      rootnav->SetCurrentDirection( dir.x(), dir.y(), dir.z() );
      rootnav->FindNextBoundaryAndStep( maxSteps[i] );
    }
  }

  // cleanup
  delete[] rootNodes;
  _mm_free(maxSteps);

  return (Precision)timer.Stop();
}
#endif

//==================================
#ifdef VECGEOM_GEANT4
Precision benchmarkGeant4Navigation( int nPoints, int nReps,
                                     SOA3D<Precision> const& points,
                                     SOA3D<Precision> const& dirs )
{
  Stopwatch timer;
  timer.Start();

  // Note: Vector3D's are expressed in cm, while G4ThreeVectors are expressed in mm
  const Precision cm = 10.;  // cm --> mm conversion
  G4Navigator& g4nav = *(G4GeoManager::Instance().GetNavigator());
  G4TouchableHistory** g4history = new G4TouchableHistory*[nPoints];

  // define max steps needed?
  //Precision* maxSteps = (Precisoin*)_mm_malloc(sizeof(Precision)*nPoints,32);

  for(int n=0; n<nReps; ++n) {
    for (int i=0; i<nPoints; ++i) {
      G4ThreeVector g4pos( points[i].x()*cm, points[i].y()*cm, points[i].z()*cm );
      G4ThreeVector g4dir( dirs[i].x(), dirs[i].y(), dirs[i].z() );

      // false == locate from top
      G4VPhysicalVolume const* vol = g4nav.LocateGlobalPointAndSetup( g4pos, &g4dir, false );
      if(!vol) std::cout<<"benchG4Navit: pos="<< g4pos <<" and dir="<< g4dir <<" --> vol="<< vol <<"\n";

      G4double safety = 0.0;
      G4double step  = g4nav.ComputeStep( g4pos, g4dir, vecgeom::kInfinity, safety );

      G4ThreeVector nextPos = g4pos + (step + 1.0e-6) * g4dir;

      g4nav.SetGeometricallyLimitedStep();
      vol = g4nav.LocateGlobalPointAndSetup( nextPos );

      // TODO: save touchable history array - returnable?  symmetrize with root benchmark
    }
  }

  // cleanup
  delete[] g4history;
//  _mm_free(maxSteps);

  return (Precision)timer.Stop();
}
#endif

//=======================================
/// Function to run navigation benchmarks
void runNavigationBenchmarks( LogicalVolume const* startVol, int np, int nreps, Precision bias) {

  SOA3D<Precision> points(np);
  SOA3D<Precision> locpts(np);
  SOA3D<Precision> dirs(np);
  vecgeom::volumeUtilities::FillGlobalPointsAndDirectionsForLogicalVolume( startVol, locpts, points, dirs, bias, np);

  Precision cputime;

  cputime = benchmarkLocatePoint(np,nreps,points);
  printf("CPU elapsed time (locating and setting steps) %f ms\n", 1000.*cputime);

  cputime = benchmarkSerialNavigation(np,nreps,points, dirs);
  printf("CPU elapsed time (serialized navigation) %f ms\n", 1000.*cputime);

  cputime = benchmarkVectorNavigation(np,nreps,points, dirs);
  printf("CPU elapsed time (vectorized navigation) %f ms\n", 1000.*cputime);

#ifdef VECGEOM_ROOT
  cputime = benchmarkROOTNavigation(np,nreps,points, dirs);
  printf("CPU elapsed time (ROOT navigation) %f ms\n", 1000.*cputime);
#endif

#ifdef VECGEOM_GEANT4
  cputime = benchmarkGeant4Navigation(np,nreps,points, dirs);
  printf("CPU elapsed time (Geant4 navigation) %f ms\n", 1000.*cputime);
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
bool validateNavigationStepAgainstRoot(
  Vector3D<Precision> const& pos,
  Vector3D<Precision> const& dir,
  Precision maxStep,
  Precision testStep,
  NavigationState const& testState)
{
  bool result = true;

  TGeoNavigator* rootnav = ::gGeoManager->GetCurrentNavigator();
  rootnav->ResetState();
  rootnav->FindNode( pos.x(), pos.y(), pos.z() );
  rootnav->SetCurrentPoint( pos.x(), pos.y(), pos.z() );
  rootnav->SetCurrentDirection( dir.x(), dir.y(), dir.z() );
  rootnav->FindNextBoundaryAndStep( maxStep );

  if( testState.Top() == NULL ) {
    if (! rootnav->IsOutside() ) {
      result = false;
      std::cerr << " OUTSIDEERROR \n";
    }
  }
  else if( Abs(testStep - rootnav->GetStep()) > 100.*kTolerance
           || rootnav->GetCurrentNode() != RootGeoManager::Instance().tgeonode(testState.Top()) )
  {
    result = false;
    std::cerr << "\n*** ERROR on validateAgainstROOT: "
              <<" ROOT node="<< rootnav->GetCurrentNode()->GetName()
              <<" outside="<< rootnav->IsOutside()
              <<" step="<< rootnav->GetStep()
              <<" <==> VecGeom node="<< (testState.Top() ? testState.Top()->GetLabel() : "NULL")
              <<" step="<< testStep
              <<" /// Step ratio="<< testStep/rootnav->GetStep()
              <<" / step diff="<< Abs(testStep-rootnav->GetStep())
              <<" / rel.error="<< Abs(testStep-rootnav->GetStep())/testStep
              <<"\n";

    std::cerr<< rootnav->GetCurrentNode() <<' '<< RootGeoManager::Instance().tgeonode(testState.Top()) << "\n";
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
bool validateNavigationStepAgainstGeant4(
  Vector3D<Precision> const& pos,
  Vector3D<Precision> const& dir,
  Precision maxStep,
  Precision testStep,
  NavigationState const& testState,
  Precision& step,
  G4VPhysicalVolume const* nextVol)
{
  // Note: Vector3D's are expressed in cm, while G4ThreeVectors are expressed in mm
  const Precision cm = 10.;  // cm --> mm conversion
  bool result = true;
  G4Navigator& g4nav = *(G4GeoManager::Instance().GetNavigator());

  G4ThreeVector g4pos( pos.x()*cm, pos.y()*cm, pos.z()*cm );
  G4ThreeVector g4dir( dir.x(), dir.y(), dir.z() );
  // false == locate from top
  G4VPhysicalVolume const * vol = g4nav.LocateGlobalPointAndSetup( g4pos, &g4dir, false );

  G4double safety = 0.0;
  step  = g4nav.ComputeStep( g4pos, g4dir, vecgeom::kInfinity, safety );

  G4ThreeVector nextPos = g4pos + (step + 1.0e-6) * g4dir;

  g4nav.SetGeometricallyLimitedStep();
  vol = g4nav.LocateGlobalPointAndSetup( nextPos );

  //--------------

  if( testState.Top() == NULL ) {
     if (! g4nav.ExitedMotherVolume() ) {
       result = false;
       std::cerr << " OUTSIDEERROR \n";
     }
  }
  else if( Abs(testStep - step/cm) > 100.*kTolerance
  //          || g4nav->GetCurrentNode() != RootGeoManager::Instance().tgeonode(testState.Top())
    ) {
    result = false;
    std::cerr << "\n*** ERROR on validateAgainstGeant4: "
              <<" Geant4 node="<< vol->GetName()
  //             <<" outside="<< g4nav->IsOutside()
              <<" step="<< step/cm    // printouts are in cm units
              <<" <==> VecGeom node="<< (testState.Top() ? testState.Top()->GetLabel() : "NULL")
              <<" step="<< testStep
              <<"\n";

    //std::cerr<< vol <<' '<< RootGeoManager::Instance().tgeonode(testState.Top()) << "\n";
  }

  return result;
}
#endif // VECGEOM_ROOT

//=======================================

bool validateVecGeomNavigation( int np, SOA3D<Precision> const& points, SOA3D<Precision> const& dirs) {

  bool result = true;

  // now setup all the navigation states - one loop at a time for better data locality
  NavigationState** origStates     = new NavigationState*[np];
  for( int i=0; i<np; ++i) origStates[i] = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );

  NavigationState** vgSerialStates = new NavigationState*[np];
  for( int i=0; i<np; ++i) vgSerialStates[i] = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );

  vecgeom::SimpleNavigator nav;
  Precision * maxSteps = (Precision *) _mm_malloc(sizeof(Precision)*np,32);
  Precision * refSteps = (Precision *) _mm_malloc(sizeof(Precision)*np,32);

  for (int i=0;i<np;++i) maxSteps[i] = kInfinity;
  memset(refSteps, 0, sizeof(Precision)*np);

  // navigation using the serial interface
#ifdef VECGEOM_ROOT
  int rootMismatches = 0;
#endif
#ifdef VECGEOM_GEANT4
  int g4Mismatches = 0;
#endif
  for (int i=0; i<np; ++i) {
    Vector3D<Precision> const& pos = points[i];
    Vector3D<Precision> const& dir = dirs[i];
    nav.LocatePoint( GeoManager::Instance().GetWorld(), pos, *origStates[i], true);
    nav.FindNextBoundaryAndStep( pos, dir, *origStates[i], *vgSerialStates[i], maxSteps[i], refSteps[i]);

    // validate serial interface against ROOT and/or Geant4
    bool ok = true;

#ifdef VECGEOM_ROOT
    ok = validateNavigationStepAgainstRoot(pos, dir, maxSteps[i], refSteps[i], *vgSerialStates[i] );
    result &= ok;
    if(!ok) ++rootMismatches;
#endif

#ifdef VECGEOM_GEANT4
    G4double g4step = kInfinity;
    G4VPhysicalVolume const* nextPV = NULL;
    ok = validateNavigationStepAgainstGeant4(pos, dir, maxSteps[i], refSteps[i], *vgSerialStates[i], g4step, nextPV );
    result &= ok;
    if(!ok) ++g4Mismatches;
#endif

    result &= ok;
    if( !ok ) {
      std::cout << "\n=======> Summary: ITERATION " << i
                <<" - pos = " << pos <<" dir = " << dir
                <<" / Steps (";
#ifdef VECGEOM_ROOT
      std::cout<<"Root/";
#endif
#ifdef VECGEOM_GEANT4
      std::cout<<"Geant4/";
#endif
      std::cout<<"VecGeom): ";

#ifdef VECGEOM_ROOT
      TGeoNavigator* rootnav = ::gGeoManager->GetCurrentNavigator();
      std::cout<< rootnav->GetStep() << " / ";
#endif
#ifdef VECGEOM_GEANT4
      std::cout<< 0.1*g4step <<" / ";
#endif
      std::cout<< refSteps[i] <<"\n";

      //=== compare navigation states
      std::cout <<"Next volumes: ";
#ifdef VECGEOM_ROOT
      std::cout << rootnav->GetCurrentNode()->GetName() <<" / ";
#endif
#ifdef VECGEOM_GEANT4
      std::cout << (nextPV ? nextPV->GetName() : "NULL") <<" / ";
#endif
      std::cout << (vgSerialStates[i]->Top()? vgSerialStates[i]->Top()->GetLabel() : "NULL") << "\n";
      // nav.InspectEnvironmentForPointAndDirection( pos, dir, *origState );
    }
  }
#ifdef VECGEOM_ROOT
  std::cout<<"VecGeom navigation - serial interface: # ROOT mismatches = "<< rootMismatches <<" / "<< np <<"\n";
#endif
#ifdef VECGEOM_GEANT4
  std::cout<<"VecGeom navigation - serial interface: # Geant4 mismatches = "<< g4Mismatches <<" / "<< np <<"\n";
#endif

  //=== N-particle navigation interface

  //--- Creating vgVectorStates
  NavigationState** vgVectorStates = new NavigationState*[np];
  for( int i=0; i<np; ++i) vgVectorStates[i] = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );

  SOA3D<Precision> workspace1(np);
  SOA3D<Precision> workspace2(np);
  int* intworkspace = (int *) _mm_malloc(sizeof(int)*np,32);

  Precision * vecSteps = (Precision *) _mm_malloc(sizeof(Precision)*np,32);
  Precision * safeties = (Precision *) _mm_malloc(sizeof(Precision)*np,32);
  memset(vecSteps, 0, sizeof(Precision)*np);
  memset(safeties, 0, sizeof(Precision)*np);

  nav.FindNextBoundaryAndStep( points, dirs, workspace1, workspace2, origStates, vgVectorStates,
                               maxSteps, safeties, vecSteps, intworkspace );

  //*** compare N-particle agains 1-particle interfaces
  // TODO: move checks into a separate function, like e.g.:
  //  ok = compareNavigationResults(refSteps, vgSerialStates, vecSteps, vgVectorStates);
  int errorCount = 0;
  for(int i=0; i<np; ++i) {
    bool mismatch = false;
    void* void1 = (void*)vgSerialStates[i]->Top();
    void* void2 = (void*)vgVectorStates[i]->Top();
    if( Abs( vecSteps[i] - refSteps[i] ) > kTolerance )                   mismatch = true;
    if( void1 && void2 && void1 != void2 )                                mismatch = true;
    if( vgSerialStates[i]->IsOnBoundary() != vgVectorStates[i]->IsOnBoundary()) mismatch = true;
    if( safeties[i] != nav.GetSafety( points[i], *origStates[i] ))   mismatch = true;
    if(mismatch) {
      result = false;
      if(mismatch) ++errorCount;
      std::cout<<"Vector navigation problems: mismatch="<< mismatch <<" - track["<< i <<"]=("
               << points[i].x() <<"; "<< points[i].y() <<"; "<< points[i].z() <<") "
               <<" / dir=("<< dirs[i].x() <<"; "<< dirs[i].y() <<"; "<< dirs[i].z() <<") "
               <<" steps: "<< refSteps[i] <<" / "<< vecSteps[i]
               <<" navStates: "<< ( void1 ? vgSerialStates[i]->Top()->GetLabel() : "NULL")
               << (vgSerialStates[i]->IsOnBoundary() ? "" : "-notOnBoundary")
               <<" / "<< (void2 ? vgVectorStates[i]->Top()->GetLabel() : "NULL")
               << (vgVectorStates[i]->IsOnBoundary() ? "" : "-notOnBoundary")
               << "\n";
    }
  }

  std::cout<<"VecGeom navigation - vector interface: #mismatches = "<< errorCount <<" / "<< np << "\n";

#ifdef VECGEOM_CUDA
  Precision * gpuSteps = (Precision *) _mm_malloc( np*sizeof(Precision), 32);
  NavStatePool gpuStates( np, GeoManager::Instance().getMaxDepth() );

  // load GPU geometry
  CudaManager::Instance().set_verbose(0);
  CudaManager::Instance().LoadGeometry(GeoManager::Instance().GetWorld());
  CudaManager::Instance().Synchronize();

  origStates.CopyToGpu();
  gpuStates.CopyToGpu();

  printf("Start validating GPU navigation...\n");
  runNavigationCuda(origStates.GetGPUPointer(), gpuStates.GetGPUPointer(),
                    GeoManager::Instance().getMaxDepth(),
                    GeoManager::Instance().GetWorld(),
                    np, points.x(),  points.y(), points.z(),
                    dirs.x(), dirs.y(), dirs.z(), maxSteps, gpuSteps );

  gpuStates.CopyFromGpu();

  //*** Comparing results from GPU against serialized navigation
  // TODO: move checks into a separate function, like e.g.:
  //  ok = compareNavigationResults(refSteps, vgSerialStates, gpuSteps, gpuStates);
  errorCount = 0;
  for(int i=0; i<np; ++i) {
    bool mismatch = false;
    if( Abs( gpuSteps[i] - refSteps[i] ) > 10.*kTolerance )                   mismatch = true;
    // if( gpuStates[i]->Top() != vgSerialStates[i]->Top() )                  mismatch = true;
    // if( gpuStates[i]->IsOnBoundary() != vgSerialStates[i]->IsOnBoundary()) mismatch = true;
    // if( safeties[i] != nav.GetSafety( points[i], *origStates[i] ))         mismatch = true;
    if(mismatch) {
      result = false;
      ++errorCount;
      std::cout<<"GPU navigation problems: track["<< i <<"]=("
               << points[i].x() <<"; "<< points[i].y() <<"; "<< points[i].z() <<") "
               <<" steps: "<< refSteps[i] <<" / "<< gpuSteps[i]
               // <<" navStates: "<< vgSerialStates[i]->Top()->GetLabel()
               // << (vgSerialStates[i]->IsOnBoundary() ? "*" : "")
               // <<" / "<< vgVectorStates[i]->Top()->GetLabel()
               // << (vgVectorStates[i]->IsOnBoundary() ? "*" : "")
               << "\n";
    }
  }

  std::cout<<"VecGeom navigation on the GPUs: #mismatches = "<< errorCount <<" / "<< np << "\n";
#endif // VECGEOM_CUDA

  // if(mismatches>0) std::cout << "Navigation test failed with "<< mismatches <<" mismatches\n";
  // else std::cout<<"Navigation test passed.\n";

  //=== cleanup
  for(int i=0; i<np; ++i) NavigationState::ReleaseInstance( origStates[i] );
  for(int i=0; i<np; ++i) NavigationState::ReleaseInstance( vgSerialStates[i] );
  for(int i=0; i<np; ++i) NavigationState::ReleaseInstance( vgVectorStates[i] );
  // delete origStates;
  // delete vgSerialStates;
  // delete vgVectorStates;
  _mm_free(intworkspace);
  _mm_free(maxSteps);
  _mm_free(refSteps);
  _mm_free(vecSteps);
  _mm_free(safeties);
#ifdef VECGEOM_NVCC
  _mm_free(gpuSteps);
#endif
  return result;
}

} // End namespace vecgeom
