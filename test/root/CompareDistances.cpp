#include "management/RootGeoManager.h"
#include "volumes/LogicalVolume.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedBox.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "volumes/UnplacedBox.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoBBox.h"
#include "utilities/Visualizer.h"
#include <string>
#include <cmath>

#ifdef VECGEOM_GEANT4
#include "G4ThreeVector.hh"
#include "G4VSolid.hh"
#endif

using namespace vecgeom;


int main( int argc, char *argv[] ) {

    if( argc < 9 )
    {
      std::cerr << "need to give root geometry file + logical volume name + local point + local dir\n";
      std::cerr << "example: " << argv[0] << " cms2015.root CALO 10.0 0.8 -3.5 1 0 0\n";
      return 1;
    }

    TGeoManager::Import( argv[1] );
    std::string testvolume( argv[2] );
    double px = atof(argv[3]);
    double py = atof(argv[4]);
    double pz = atof(argv[5]);
    double dirx = atof(argv[6]);
    double diry = atof(argv[7]);
    double dirz = atof(argv[8]);

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
    foundvolume->GetShape()->InspectShape();
    std::cerr << "volume capacity " << foundvolume->GetShape()->Capacity() << "\n";

    // now get the VecGeom shape and benchmark it
    if( foundvolume )
    {
        LogicalVolume * converted = RootGeoManager::Instance().Convert( foundvolume );
        VPlacedVolume * vecgeomplaced = converted->Place();
        Vector3D<Precision> point(px,py,pz);
        Vector3D<Precision> dir(dirx,diry,dirz);

        std::cout << "VecGeom Capacity " << vecgeomplaced->Capacity( ) << "\n";
        std::cout << "VecGeom CONTAINS " << vecgeomplaced->Contains( point ) << "\n";
        std::cout << "VecGeom DI " << vecgeomplaced->DistanceToIn( point, dir ) << "\n";
        std::cout << "VecGeom DO " << vecgeomplaced->DistanceToOut( point, dir ) << "\n";
        std::cout << "VecGeom SI " << vecgeomplaced->SafetyToIn( point ) << "\n";
        std::cout << "VecGeom SO " << vecgeomplaced->SafetyToOut( point ) << "\n";

        std::cout << "ROOT Capacity " << foundvolume->GetShape()->Capacity(  ) << "\n";
        std::cout << "ROOT CONTAINS " << foundvolume->GetShape()->Contains( &point[0] ) << "\n";
        std::cout << "ROOT DI " << foundvolume->GetShape()->DistFromOutside( &point[0], &dir[0] ) << "\n";
        std::cout << "ROOT DO " << foundvolume->GetShape()->DistFromInside( &point[0], &dir[0] ) << "\n";
        std::cout << "ROOT SI " << foundvolume->GetShape()->Safety( &point[0], false ) << "\n";
        std::cout << "ROOT SO " << foundvolume->GetShape()->Safety( &point[0], true ) << "\n";

        TGeoShape const * rootback = vecgeomplaced->ConvertToRoot();
        std::cout << "ROOTBACKCONV CONTAINS " << rootback->Contains( &point[0] ) << "\n";
        std::cout << "ROOTBACKCONV DI " << rootback->DistFromOutside( &point[0], &dir[0] ) << "\n";
        std::cout << "ROOTBACKCONV DO " << rootback->DistFromInside( &point[0], &dir[0] ) << "\n";
        std::cout << "ROOTBACKCONV SI " << rootback->Safety( &point[0], false ) << "\n";
        std::cout << "ROOTBACKCONV SO " << rootback->Safety( &point[0], true ) << "\n";

#ifdef VECGEOM_USOLIDS
        VUSolid const * usolid = vecgeomplaced->ConvertToUSolids();
        std::cout << "USolids Capacity " << const_cast<VUSolid*>(usolid)->Capacity(  ) << "\n";
        std::cout << "USolids CONTAINS " << usolid->Inside( point ) << "\n";
        std::cout << "USolids DI " << usolid->DistanceToIn( point, dir ) << "\n";

        Vector3D<Precision> norm; bool valid;
        std::cout << "USolids DO " << usolid->DistanceToOut( point, dir, norm, valid ) << "\n";
        std::cout << "USolids SI " << usolid->SafetyFromInside( point ) << "\n";
        std::cout << "USolids SO " << usolid->SafetyFromOutside( point ) << "\n";
#endif

#ifdef VECGEOM_GEANT4
        G4ThreeVector g4p( point.x(), point.y(), point.z() );
        G4ThreeVector g4d( dir.x(), dir.y(), dir.z());

        G4VSolid const * g4solid = vecgeomplaced->ConvertToGeant4();
        std::cout << "G4 CONTAINS " << g4solid->Inside( g4p ) << "\n";
        std::cout << "G4 DI " << g4solid->DistanceToIn( g4p, g4d ) << "\n";
        std::cout << "G4 DO " << g4solid->DistanceToOut( g4p, g4d ) << "\n";
        std::cout << "G4 SI " << g4solid->DistanceToIn( g4p ) << "\n";
        std::cout << "G4 SO " << g4solid->DistanceToOut( g4p ) << "\n";
#endif

        double step=0;
        if( foundvolume->GetShape()->Contains( &point[0] ) ){
            step = foundvolume->GetShape()->DistFromInside( &point[0], &dir[0] );
        }
        else
        {
            step = foundvolume->GetShape()->DistFromOutside( &point[0], &dir[0] );
        }
        Visualizer visualizer;
        visualizer.AddVolume( *vecgeomplaced );
        visualizer.AddPoint( point );
        visualizer.AddLine( point, point + step * dir );
        visualizer.Show();

    }
    else
    {
        std::cerr << " NO SUCH VOLUME [" <<  testvolume << "] FOUND ... EXITING \n";
        return 1;
    }
}

