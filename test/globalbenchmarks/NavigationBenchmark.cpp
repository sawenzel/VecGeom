/*
 *  File: NavigationBenchmark.cpp
 *
 *  Created on: Oct 25, 2014
 *      Author: swenzel, lima
 */

#ifdef VECGEOM_ROOT
  #include "management/RootGeoManager.h"
  #include "utilities/Visualizer.h"
#endif

#ifdef VECGEOM_GEANT4
  #include "management/G4GeoManager.h"
  #include "G4ThreeVector.hh"
  // #include "G4TouchableHistoryHandle.hh"
  #include "G4LogicalVolume.hh"
  #include "G4LogicalVolumeStore.hh"
  #include "G4PVPlacement.hh"
  #include "G4GeometryManager.hh"
#endif

#include "benchmarking/NavigationBenchmarker.h"
#include "test/benchmark/ArgParser.h"
#include "volumes/utilities/VolumeUtilities.h"

#include "management/GeoManager.h"
#include "volumes/Box.h"
#include "volumes/Orb.h"
#include "volumes/Trapezoid.h"

using namespace VECGEOM_NAMESPACE;


VPlacedVolume* SetupGeometry() {

  UnplacedBox *worldUnplaced = new UnplacedBox(10, 10, 10);
  UnplacedTrapezoid *trapUnplaced = new UnplacedTrapezoid(4,0,0,4,4,4,0,4,4,4,0);
  UnplacedBox *boxUnplaced = new UnplacedBox(2.5, 2.5, 2.5);
  UnplacedOrb *orbUnplaced = new UnplacedOrb(2.8);

  LogicalVolume *world = new LogicalVolume("world",worldUnplaced);
  LogicalVolume *trap  = new LogicalVolume("trap",trapUnplaced);
  LogicalVolume *box   = new LogicalVolume("box",boxUnplaced);
  LogicalVolume *orb   = new LogicalVolume("orb",orbUnplaced);

  Transformation3D *ident = new Transformation3D( 0,  0,  0,  0,  0,  0);
  orb->PlaceDaughter("orb1",box, ident);
  trap->PlaceDaughter("box1",orb, ident);

  Transformation3D *placement1 = new Transformation3D( 5,  5,  5,  0,  0,  0);
  Transformation3D *placement2 = new Transformation3D(-5,  5,  5,  0,  0,  0); //45,  0,  0);
  Transformation3D *placement3 = new Transformation3D( 5, -5,  5,  0,  0,  0); // 0, 45,  0);
  Transformation3D *placement4 = new Transformation3D( 5,  5, -5,  0,  0,  0); // 0,  0, 45);
  Transformation3D *placement5 = new Transformation3D(-5, -5,  5,  0,  0,  0); //45, 45,  0);
  Transformation3D *placement6 = new Transformation3D(-5,  5, -5,  0,  0,  0); //45,  0, 45);
  Transformation3D *placement7 = new Transformation3D( 5, -5, -5,  0,  0,  0); // 0, 45, 45);
  Transformation3D *placement8 = new Transformation3D(-5, -5, -5,  0,  0,  0); //45, 45, 45);

  world->PlaceDaughter("trap1",trap, placement1);
  world->PlaceDaughter("trap2",trap, placement2);
  world->PlaceDaughter("trap3",trap, placement3);
  world->PlaceDaughter("trap4",trap, placement4);
  world->PlaceDaughter("trap5",trap, placement5);
  world->PlaceDaughter("trap6",trap, placement6);
  world->PlaceDaughter("trap7",trap, placement7);
  world->PlaceDaughter("trap8",trap, placement8);

  VPlacedVolume* w = world->Place();
  GeoManager::Instance().SetWorld(w);
  GeoManager::Instance().CloseGeometry();
  return w;
}


int main(int argc, char* argv[])
{
  OPTION_INT(ntracks, 10000);
  OPTION_INT(nreps, 3);
  OPTION_STRING(geometry, "navBench");
  OPTION_STRING(logvol, "world");
  OPTION_DOUBLE(bias, 0.8f);
#ifdef VECGEOM_ROOT
  OPTION_BOOL(vis, false);
#endif

  // default values used above are always printed.  If help true, stop now, so user will know which options
  // are available, and what the default values are.
  OPTION_BOOL(help, false);
  if(help) return 0;

  if(geometry.compare("navBench")==0) {
    const VPlacedVolume *world = SetupGeometry();

#ifdef VECGEOM_ROOT
    // Exporting to ROOT file
    RootGeoManager::Instance().ExportToROOTGeometry( world, "navBench.root" );
    RootGeoManager::Instance().Clear();
#endif
  }

  // Now try to read back in.  This is needed to make comparisons to VecGeom easily,
  // since it builds VecGeom geometry based on the ROOT geometry and its TGeoNodes.
#ifdef VECGEOM_ROOT
  auto rootgeom = geometry+".root";
  RootGeoManager::Instance().set_verbose(0);
  RootGeoManager::Instance().LoadRootGeometry(rootgeom.c_str());
#endif

#ifdef VECGEOM_GEANT4
  auto g4geom = geometry+".gdml";
  G4GeoManager::Instance().LoadG4Geometry( g4geom.c_str() );
#endif

  // Visualization
#ifdef VECGEOM_ROOT
  if(vis) {  // note that visualization block returns, excluding the rest of benchmark
    Visualizer visualizer;
    const VPlacedVolume* world = GeoManager::Instance().GetWorld();
    world = GeoManager::Instance().FindPlacedVolume(logvol.c_str());
    visualizer.AddVolume( *world );

    Vector<Daughter> const* daughters = world->GetLogicalVolume()->GetDaughtersp();
    for(int i=0; i<daughters->size(); ++i) {
      VPlacedVolume const* daughter = (*daughters)[i];
      Transformation3D const& trf1 = *(daughter->GetTransformation());
      visualizer.AddVolume(*daughter, trf1);

      // Vector<Daughter> const* daughters2 = daughter->GetLogicalVolume()->daughtersp();
      // for(int ii=0; ii<daughters2->size(); ++ii) {
      //   VPlacedVolume const* daughter2 = (*daughters2)[ii];
      //   Transformation3D const& trf2 = *(daughter2->transformation());
      //   Transformation3D comb = trf1;
      //   comb.MultiplyFromRight(trf2);
      //   visualizer.AddVolume(*daughter2, comb);
      // }
    }

    visualizer.Show();
    return 0;
  }
#endif


  std::cout<<"\n*** Validating VecGeom navigation...\n";

  const LogicalVolume* startVolume = GeoManager::Instance().GetWorld()->GetLogicalVolume();
  if( logvol.compare("world")!=0 ) {
    startVolume = GeoManager::Instance().FindLogicalVolume(logvol.c_str());
  }

  std::cout<<"NavigationBenchmark: logvol=<"<< logvol
           <<">, startVolume=<"<< (startVolume ? startVolume->GetLabel() : "NULL") <<">\n";
  if(startVolume) std::cout<< *startVolume <<"\n";

  // prepare tracks to be used for validation
  int np = Min( ntracks, 1000 );  // no more than 1000 points used for validation
  SOA3D<Precision> points(np);
  SOA3D<Precision> dirs(np);
  SOA3D<Precision> locpts(np);
  vecgeom::volumeUtilities::FillGlobalPointsAndDirectionsForLogicalVolume(startVolume, locpts, points, dirs, bias, np);

  Precision * maxSteps = (Precision *) _mm_malloc(sizeof(Precision)*np,32);
  for (int i=0;i<np;++i) maxSteps[i] = 10. * RNG::Instance().uniform();

  // Must be validated before being benchmarked
  bool ok = validateVecGeomNavigation(np, points, dirs, maxSteps);
  if(!ok) {
    std::cout<<"VecGeom validation failed."<< std::endl;
    return 1;
  }
  std::cout<<"VecGeom validation passed."<< std::endl;

  // on mic.fnal.gov CPUs, loop execution takes ~70sec for ntracks=10M
  while(ntracks<=10000) {
    std::cout<<"\n*** Running navigation benchmarks with ntracks="<<ntracks<<" and nreps="<< nreps <<".\n";
    runNavigationBenchmarks(startVolume, ntracks, nreps, maxSteps, bias);
    ntracks*=10;
  }


/*
// GPU part
  int nDevice;
  cudaGetDeviceCount(&nDevice);

  if(nDevice > 0) {
    cudaDeviceReset();
  }
  else {
    std::cout << "No Cuda Capable Device ... " << std::endl;
    return 0;
  }
*/

  // cleanup
  delete [] maxSteps;
  return 0;
}
