#include "volumes/LogicalVolume.h"
#include "volumes/Tube.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

void benchmark(double rmin, double rmax, double dz, double sphi, double dphi, int npoints, int nrep) {
  UnplacedBox worldUnplaced = UnplacedBox(rmax*4, rmax*4, dz*4);
  UnplacedTube tubeUnplaced = UnplacedTube(rmin, rmax, dz, sphi, dphi);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume tube = LogicalVolume("tube", &tubeUnplaced);

  Transformation3D placement(5, 5, 5);
  world.PlaceDaughter("tube", &tube, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorld(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.RunBenchmark();
}

int main(int argc, char* argv[]) {
  OPTION_INT(npoints,1024);
  OPTION_INT(nrep,1024);
  OPTION_DOUBLE(rmin,0);
  OPTION_DOUBLE(rmax,5);
  OPTION_DOUBLE(dz,10);
  OPTION_DOUBLE(sphi,0);
  OPTION_DOUBLE(dphi,kTwoPi);

  benchmark(rmin, rmax, dz, sphi, dphi, npoints, nrep);
  
  // for(int hasrmin = 0; hasrmin < 2; hasrmin++) {
  //   double rmin = 10;
  //   for(int hasphi = 0; hasphi < 4; hasphi++) {
  //       double dphi = 2*M_PI;
  //       if(hasphi == 1) dphi = M_PI / 4;
  //       if(hasphi == 2) dphi = M_PI;
  //       if(hasphi == 3) dphi = 3*M_PI / 2;
  //
  //       std::cout << "=========================================================================================" << std::endl;
  //       if(!hasrmin) rmin = 0;
  //       if(hasrmin) std::cout << "rmin";
  //       if(!hasrmin) std::cout << "no rmin";
  //       std::cout << " + ";
  //       
  //       if(hasphi == 0) std::cout << "no phi"; 
  //       if(hasphi == 1) std::cout << "phi smaller than PI"; 
  //       if(hasphi == 2) std::cout << "phi == PI"; 
  //       if(hasphi == 3) std::cout << "phi bigger than PI"; 
  //       std::cout << std::endl;
  //       std::cout << "=========================================================================================" << std::endl;
  //
  //       benchmark(rmin, 20., 30., 0, dphi);
  //   }
  //
  // }
}
