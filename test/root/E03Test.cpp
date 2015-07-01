/*
 * E03Test.cpp
 *
 *  Created on: Jan 7, 2015
 *      Author: swenzel
 */

// tests on a simple geometry from the Geant4 E03 example
// only makes sense with the proper geometry description (EN03.root)

#include "management/RootGeoManager.h"
#include "management/GeoManager.h"
#include "navigation/NavigationState.h"
#include "navigation/SimpleNavigator.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedBox.h"

#ifdef VECGEOM_ROOT
#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TGeoVolume.h"
#endif

typedef vecgeom::NavigationState VolumePath_t;
using namespace vecgeom;

void locatetest()
{
    VolumePath_t * a;
    a = VolumePath_t::MakeInstance(3);
    SimpleNavigator nav;

    vecgeom::UnplacedBox * const box = ( vecgeom::UnplacedBox * const ) GeoManager::Instance().GetWorld()->GetUnplacedVolume();
    std::cerr << "\n" << box << "\n";
    std::cerr << box->dimensions() << "\n";
    std::cerr << box->dimensions().x() << "\n";
    std::cerr << box->dimensions().y() << "\n";
    std::cerr << box->dimensions().z() << "\n";
    std::cerr << box->x() << "\n";
    std::cerr << box->y() << "\n";
    std::cerr << box->z() << "\n";
    std::cerr << box->volume() << "\n"; // not OK
    box->Print(); // OK

    nav.LocatePoint( GeoManager::Instance().GetWorld(),
                          Vector3D<Precision>(-8,0,0), *a, true );
    a->Print();
    if( a->GetCurrentNode() != NULL )
    {
       a->GetCurrentNode()->GetVolume()->Print();
    }
    NavigationState::ReleaseInstance(a);
}

void loadvecgeomgeometry()
{
if( vecgeom::GeoManager::Instance().GetWorld() == NULL )
   {
    Printf("Now loading VecGeom geometry\n");
    vecgeom::RootGeoManager::Instance().LoadRootGeometry();
    Printf("Loading VecGeom geometry done\n");
    Printf("Have depth %d\n", vecgeom::GeoManager::Instance().getMaxDepth());
    std::vector<vecgeom::LogicalVolume *> v1;
    vecgeom::GeoManager::Instance().getAllLogicalVolumes( v1 );
    Printf("Have logical volumes %ld\n", v1.size() );
    std::vector<vecgeom::VPlacedVolume *> v2;
    vecgeom::GeoManager::Instance().getAllPlacedVolumes( v2 );
    Printf("Have placed volumes %ld\n", v2.size() );
    vecgeom::RootGeoManager::Instance().world()->PrintContent();
   }

}

//______________________________________________________________________________
void loadgeometry(const char *filename)
{
// Load the detector geometry from file, unless already loaded.
   TGeoManager *geom = (gGeoManager)? gGeoManager : TGeoManager::Import(filename);
   if (geom)
       {
         loadvecgeomgeometry();
         // fMaxDepth = TGeoManager::GetMaxLevels();
       }
}


int main(  )
{
   // read in detector passed as argument
   //RootGeoManager::Instance().set_verbose(3);
   loadgeometry( "ExN03.root" );

   locatetest();

   return 0;
}

