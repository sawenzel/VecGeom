#include "volumes/LogicalVolume.h"
#include "volumes/Tube.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"

using namespace vecgeom;

void benchmark(double rmin, double rmax, double z, double sphi, double dphi) {
  UnplacedBox worldUnplaced = UnplacedBox(100., 100., 100.);
  UnplacedTube tubeUnplaced = UnplacedTube(rmin, rmax, z, sphi, dphi);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume tube = LogicalVolume("tube", &tubeUnplaced);

  Transformation3D placement(5, 5, 5);
  world.PlaceDaughter("tube", &tube, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetRepetitions(1024);
  tester.SetPointCount(1<<6);
  tester.RunBenchmark();
}

int main() {

  for(int hasrmin = 0; hasrmin < 2; hasrmin++) {
    double rmin = 10;
    for(int hasphi = 0; hasphi < 4; hasphi++) {
        double dphi = 2*M_PI;
        if(hasphi == 1) dphi = M_PI / 4;
        if(hasphi == 2) dphi = M_PI;
        if(hasphi == 3) dphi = 3*M_PI / 2;

        std::cout << "=========================================================================================" << std::endl;
        if(!hasrmin) rmin = 0;
        if(hasrmin) std::cout << "rmin";
        if(!hasrmin) std::cout << "no rmin";
        std::cout << " + ";
        
        if(hasphi == 0) std::cout << "no phi"; 
        if(hasphi == 1) std::cout << "phi smaller than PI"; 
        if(hasphi == 2) std::cout << "phi == PI"; 
        if(hasphi == 3) std::cout << "phi bigger than PI"; 
        std::cout << std::endl;
        std::cout << "=========================================================================================" << std::endl;

        benchmark(rmin, 20., 30., 0, dphi);
    }

  }
}
