/*
 * BenchmarkShapeFromROOTFile.cpp
 *
 *  Created on: Jan 26, 2015
 *      Author: swenzel
 */


#include "management/RootGeoManager.h"
#include "volumes/LogicalVolume.h"
#include "volumes/UnplacedBox.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoBBox.h"
#include <string>

using namespace vecgeom;

// benchmarking any available shape (logical volume) found in a ROOT file
// usage: BenchmarkShapeFromROOTFile detector.root logicalvolumename
// logicalvolumename should not contain trailing pointer information

int main( int argc, char *argv[] ) {

    if( argc < 3 )
    {
      std::cerr << "need to give root geometry file and logical volume name";
    }

    TGeoManager::Import( argv[1] );
    std::string testvolume( argv[2] );


    int found = 0;
    TGeoVolume * foundvolume = NULL;
    // now try to find shape with logical volume name given on the command line
    TObjArray *vlist = gGeoManager->GetListOfVolumes( );
    for( auto i = 0; i < vlist->GetEntries(); ++i )
    {
        TGeoVolume * vol = reinterpret_cast<TGeoVolume*>(vlist->At( i ));
        std::string fullname(vol->GetName());

	// strip off pointer information
	std::string strippedname(fullname, 0, fullname.length()-4);

        std::size_t founds = strippedname.compare(testvolume);
        if (founds == 0){
            found++;
            foundvolume = vol;

	    std::cerr << "found matching volume " << fullname << " of type " << vol->GetShape()->ClassName() << "\n";
        }
    }

    std::cerr << "volume found " << found << " times \n";

    // now get the VecGeom shape and benchmark it
    LogicalVolume * converted = RootGeoManager::Instance().Convert( foundvolume );

    // get the bounding box
    TGeoBBox * bbox = dynamic_cast<TGeoBBox *>(foundvolume->GetShape( ));
    double bx, by, bz;
    double const * offset;
    if(bbox)
    {
        bbox->InspectShape();
        offset = bbox->GetOrigin();


        UnplacedBox worldUnplaced( (bx + fabs(offset[0])*4, (by + fabs(offset[1]))*4 , (bz + fabs(offset[2]))*4));
        LogicalVolume world("world", &worldUnplaced);
        Transformation3D placement(5, 0, 0);
        world.PlaceDaughter("testshape", converted, &placement);

        VPlacedVolume *worldPlaced = world.Place();
        GeoManager::Instance().SetWorld(worldPlaced);

        Benchmarker tester(GeoManager::Instance().GetWorld());
        tester.SetTolerance(1E-6);
        tester.SetVerbosity(3);
        tester.SetPoolMultiplier(1);
        tester.SetRepetitions(1);
        tester.SetPointCount(1000);
        int returncodeIns   = tester.RunInsideBenchmark();

        int returncodeToIn  = tester.RunToInBenchmark();

        int returncodeToOut = tester.RunToOutBenchmark();

        return returncodeIns + returncodeToIn + returncodeToOut;
     }
    return 1;
}



