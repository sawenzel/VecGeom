/// @file ParaboloidBenchmark.cpp
/// @author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/logical_volume.h"
#include "volumes/box.h"
#include "volumes/Paraboloid.h"
#include "benchmarking/Benchmarker.h"
#include "management/geo_manager.h"

using namespace vecgeom;

int main() {
    
    std::cout<<"Paraboloid Benchmark\n";
    UnplacedBox worldUnplaced = UnplacedBox(18., 11., 11.);
    UnplacedParaboloid paraboloidUnplaced = UnplacedParaboloid(3., 5., 7.); //rlo=3. - rhi=5. dz=7
    std::cout<<"Paraboloid created\n";
    LogicalVolume world = LogicalVolume("MBworld", &worldUnplaced);
    LogicalVolume paraboloid = LogicalVolume("paraboloid", &paraboloidUnplaced);
    world.PlaceDaughter(&paraboloid, &Transformation3D::kIdentity);
    VPlacedVolume *worldPlaced = world.Place();
    GeoManager::Instance().set_world(worldPlaced);
    std::cout<<"World set\n";
    
    int np=1000, nIn=250, nOut=np-nIn;

    
    Vector3D <Precision> *points = new Vector3D<Precision>[np];
    
    for(int i=0; i<nIn; i++) // points inside
    {
        points[i].x()=rand()%3;
        points[i].y()=0;
        points[i].z()=0;
        

    }
    for (int i=nIn; i<np; i++) //points outside
    {
        points[i].x()=i*rand();
        points[i].y()=i*rand();
        points[i].z()=10;
        
    }
    int countIn=0, countOut=0;
    bool inside=10;
    VPlacedVolume *paraboloidPlaced=paraboloid.Place();
    
    for(int i=0; i<np; i++)
    {
        inside=paraboloidPlaced->Inside(points[i]);
        if(inside==0) countOut++;
        else countIn++;
    }
    
    std::cout<<"NIn: "<<nIn<<" NOut: "<<nOut<<" \n";
    std::cout<<"NPointsInside: "<<countIn<<" NPointsOutside: "<<countOut<<" \n";
    
    
    Benchmarker tester(GeoManager::Instance().world());
    tester.SetVerbosity(3);
    tester.SetPointCount(1<<13);
    std::cout<<"Prepared to run benchmarker\n";
    tester.RunBenchmark();
    
    return 0;
}

