#include "volumes/LogicalVolume.h"
#include "volumes/UnplacedPolycone.h"
#include "volumes/Box.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "base/Vector3D.h"
#include "base/Global.h"
#include "ArgParser.h"
#ifdef VECGEOM_ROOT
#include "utilities/Visualizer.h"
#endif

using namespace vecgeom;


// we need a json reader

int main(int argc, char* argv[]) {
  OPTION_INT(npoints,1024);
  OPTION_INT(nrep,1024);
  OPTION_DOUBLE(phistart,0.);
  OPTION_DOUBLE(phidelta,kTwoPi);

  int Nz = 4;
  double rmin[] = { 0.1, 0., 0. , 0.2 };
  double rmax[] = { 1., 2., 2. , 1.5 };
  double z[] = { -1, -0.5, 0.5, 10 };

  UnplacedBox worldUnplaced( 5, 5, 15 );
  UnplacedPolycone pconUnplaced(phistart, phidelta, Nz, rmin, rmax, z);

  pconUnplaced.Print();

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume pcon ("pcon", &pconUnplaced);

  //Transformation3D placement(5, 5, 5);
  Transformation3D placement(0, 0, 0);
  VPlacedVolume const * vol = world.PlaceDaughter("pcon", &pcon, &placement);

  pcon.Place(&placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorld(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetTolerance(1E-8);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.SetInsideBias( 0.5 );

  tester.RunBenchmark();


#if 0
  Visualizer visualizer;
  visualizer.AddVolume( *vol );
  if( tester.GetProblematicContainPoints().size() > 0 ) {
      for(auto v : tester.GetProblematicContainPoints())
      {
        visualizer.AddPoint(v);

        // for debugging purpose
        std::cerr << " " << vol->Contains(v) << "\n";
        std::cout << v<<"\n";
      }
      visualizer.Show();
  }
#endif

  return 0;
}
