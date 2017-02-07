
//#include "VUSolid.hh"
#include "management/RootGeoManager.h"
#include "volumes/PlacedVolume.h"
#include "volumes/utilities/VolumeUtilities.h"

#include "base/Global.h"
#include "base/Vector3D.h"
#include "base/Stopwatch.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cassert>

#include "TGeoManager.h"
#include <iomanip>

#undef NDEBUG

using namespace vecgeom;

bool ExitingMethod1( VPlacedVolume const * pvol,
                       Vector3D<Precision> const & point,
                       Vector3D<Precision> const & dir )
{
    Vector3D<Precision> normal;
    bool valid = pvol->Normal( point, normal );
    return valid && normal.Dot( dir ) > 0;
}

// determines if track is exiting or entering volume
// solely based on the Contains functions
// expects a point on the surface
bool ExitingMethod2( VPlacedVolume const * pvol,
                       Vector3D<Precision> const & point,
                       Vector3D<Precision> const & dir )
{
    bool originalin = pvol->Contains(point);
    bool movedin    = pvol->Contains( point + dir*1E-6 );
    return ( originalin && ! movedin ) || ( !originalin && ! movedin );
}

//////////////////////////////////
// main function
int main(int argc, char * argv[])
{
  if( argc < 3 )
  {
    std::cerr<< std::endl;
    std::cerr<< "Need to give rootfile + volumename"<< std::endl;
    return 1;
  }

  TGeoManager::Import( argv[1] );
  std::string testvolume( argv[2] );

  int found = 0;
  TGeoVolume * foundvolume = NULL;
  // now try to find shape with logical volume name given on the command line
  TObjArray *vlist = gGeoManager->GetListOfVolumes( );
  for( auto i = 0; i < vlist->GetEntries(); ++i ) {
    TGeoVolume * vol = reinterpret_cast<TGeoVolume*>(vlist->At( i ));
    std::string fullname(vol->GetName());
    std::size_t founds = fullname.compare(testvolume);
    if ( founds==0 ){
      found++;
      foundvolume = vol;
      std::cerr << "("<< i<< ")found matching volume " << foundvolume->GetName()
        << " of type " << foundvolume->GetShape()->ClassName() << "\n";
    }
  }
  std::cerr << "volume found " << found << " times \n\n";

  // if volume not found take world
  if( ! foundvolume ) {
      std::cerr << "specified volume not found; exiting\n";
      return 1;
  }

  if( foundvolume ) {
    // convert current gGeoManager to a VecGeom geometry
    VPlacedVolume const * vecgeompvol
    = RootGeoManager::Instance().Convert( foundvolume )->Place();

    for(int i = 0; i<20; ++i )
    {
        Vector3D<Precision> point = vecgeompvol->GetPointOnSurface();

        if( vecgeompvol->Inside(point) != vecgeom::kSurface){
            std::cerr << " WARNING : Inside does not report surface state \n";
        }
        Vector3D<Precision> dir = volumeUtilities::SampleDirection();
        bool contained = vecgeompvol->Contains(point);
        bool exiting = ExitingMethod2(vecgeompvol,point,dir);
        double DO = vecgeompvol->DistanceToOut( point , dir );
        double DI = vecgeompvol->DistanceToIn( point , dir );

        std::cerr <<  i << "  " << point << "  " << contained
//                  << "   ExitingM1  " << ExitingMethod1(vecgeompvol,point,dir)
                  << "   ExitingM2  " << exiting
                  << " DI " << DI
                  << " DO " << DO
                  << " SO " << vecgeompvol->SafetyToOut( point )
                  << " SI " << vecgeompvol->SafetyToIn( point ) << "\n";

        if( exiting && DO > vecgeom::kTolerance ){
            std::cout << " WARNING FOR DO  : should be zero \n";
            std::cout << "./CompareDistances " << argv[1] << " " << argv[2]
                  << std::setprecision(20) << " " << point[0] << " " << point[1] << " " << point[2]
                  << " " << dir[0] << " " << dir[1] << " " << dir[2] << "\n";
        }
        if( contained && ! exiting && DO < vecgeom::kTolerance ) {
            std::cout << " FATAL WARNING FOR DO : should be finite \n";
            std::cout << "./CompareDistances " << argv[1] << " " << argv[2]
                      << std::setprecision(20) << " " << point[0] << " " << point[1] << " " << point[2]
                      << " " << dir[0] << " " << dir[1] << " " << dir[2] << "\n";
        }

        if( ! exiting && DI > vecgeom::kTolerance ){
            std::cout << " WARNING FOR DI : should be zero \n";
            std::cout << "./CompareDistances " << argv[1] << " " << argv[2]
                          << std::setprecision(20) << " " << point[0] << " " << point[1] << " " << point[2]
                          << " " << dir[0] << " " << dir[1] << " " << dir[2] << "\n";
        }
        if( ! exiting && ! contained && DI > 1E20 ){
                    std::cout << " FATAL WARNING FOR DI : should be zero \n";
                    std::cout << "./CompareDistances " << argv[1] << " " << argv[2]
                                  << std::setprecision(20) << " " << point[0] << " " << point[1] << " " << point[2]
                                  << " " << dir[0] << " " << dir[1] << " " << dir[2] << "\n";
        }

        std::cout << "\n\n";
    } // end for
    return 0;
  } // end if found volume
  return 1;
}

