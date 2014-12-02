#include "base/Transformation3D.h"
#include "base/Vector3D.h"
#include "volumes/Box.h"
#include "volumes/Tube.h"
#include "volumes/kernel/shapetypes/TubeTypes.h"
#include "volumes/BooleanVolume.h"
#include "management/GeoManager.h"
#include "benchmarking/Benchmarker.h"
#include "ArgParser.h"


using namespace vecgeom;


int main(int argc, char * argv[])
{
    OPTION_INT(npoints,1024);
    OPTION_INT(nrep,1024);

    UnplacedBox worldUnplaced(2.,2.,2.);
    LogicalVolume world = LogicalVolume("world", &worldUnplaced);

    // components for boolean solid
    UnplacedBox  box(0.9, 0.9 , 0.9);
    UnplacedTube hollowtube(0.8,1.2,1.5,0,kTwoPi);

    // translation for boolean solid right shape ( it should now stick outside )
    Transformation3D translation( 0.5, 0, 0 );

    VPlacedVolume *worldPlaced = world.Place();
    GeoManager::Instance().SetWorld(worldPlaced);


    VPlacedVolume * placedtube
        = (new LogicalVolume("",&hollowtube))->Place(&translation);
    VPlacedVolume * placedmotherbox = (new LogicalVolume("",&box))->Place();

    // now make the unplaced boolean solid
    UnplacedBooleanVolume booleansolid(kIntersection, placedmotherbox, placedtube);
    LogicalVolume booleanlogical("booleanL",&booleansolid);

    // place the boolean volume into the world

    // placement of boolean solid in world
    Transformation3D placement(0.2, 0, 0);

    // add this boolean solid to the world
    world.PlaceDaughter( &booleanlogical, &placement );

    Benchmarker tester(GeoManager::Instance().GetWorld());
    tester.SetVerbosity(3);
    tester.SetPoolMultiplier(1);
    tester.SetRepetitions(nrep);
    tester.SetPointCount(npoints);
    tester.RunBenchmark();

    return 0;
}

