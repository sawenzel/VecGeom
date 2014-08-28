/// @file ParaboloidBenchmark.cpp
/// @author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "volumes/Paraboloid.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"
using namespace vecgeom;

int main(int argc, char* argv[]) {
    OPTION_INT(npoints);
    OPTION_INT(nrep);
    OPTION_DOUBLE(rlo);
    OPTION_DOUBLE(rhi);
    OPTION_DOUBLE(dz);
    
    std::cout<<"Paraboloid Benchmark\n";
    //UnplacedBox worldUnplaced = UnplacedBox(10., 10., 10.);
    // UnplacedParaboloid paraboloidUnplaced = UnplacedParaboloid(3., 5., 7.); //rlo=3. - rhi=5. dz=7
    UnplacedBox worldUnplaced = UnplacedBox(rhi*4, rhi*4, dz * 4);
    UnplacedParaboloid paraboloidUnplaced = UnplacedParaboloid(rlo, rhi, dz); //rlo=3. - rhi=5. dz=7
    std::cout<<"Paraboloid created\n";
    LogicalVolume world = LogicalVolume("MBworld", &worldUnplaced);
    LogicalVolume paraboloid = LogicalVolume("paraboloid", &paraboloidUnplaced);
    world.PlaceDaughter(&paraboloid, &Transformation3D::kIdentity);
    VPlacedVolume *worldPlaced = world.Place();
    GeoManager::Instance().set_world(worldPlaced);
    std::cout<<"World set\n";
    
    
    Benchmarker tester(GeoManager::Instance().world());
    tester.SetVerbosity(3);
    tester.SetPointCount(npoints);
    tester.SetRepetitions(nrep);
    std::cout<<"Prepared to run benchmarker\n";
    tester.RunBenchmark();
    
    return 0;
}

