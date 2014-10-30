/*
 * ImportGDMLTest.cpp
 *
 *  Created on: Jun 24, 2014
 *      Author: swenzel
 */

#include "management/GeoManager.h"
#include "management/RootGeoManager.h"
#include "TGeoManager.h"
#include <cstdio>

//______________________________________________________________________________
void LoadVecGeomGeometry()
{
   if( vecgeom::GeoManager::Instance().GetWorld() == NULL )
   {
    printf("Now loading VecGeom geometry\n");
    vecgeom::RootGeoManager::Instance().LoadRootGeometry();
    printf("Loading VecGeom geometry done\n");
    printf("Have depth %d\n", vecgeom::GeoManager::Instance().getMaxDepth());
    std::vector<vecgeom::LogicalVolume *> v1;
    vecgeom::GeoManager::Instance().getAllLogicalVolumes( v1 );
    printf("Have logical volumes %ld\n", v1.size() );
    std::vector<vecgeom::VPlacedVolume *> v2;
    vecgeom::GeoManager::Instance().getAllPlacedVolumes( v2 );
    printf("Have placed volumes %ld\n", v2.size() );
    vecgeom::RootGeoManager::Instance().world()->PrintContent();
   }
}

//______________________________________________________________________________
void LoadGeometry(const char *filename)
{
   TGeoManager *geom = (gGeoManager)? gGeoManager : TGeoManager::Import(filename);
   if (geom)
   {
      LoadVecGeomGeometry();
   }
}

int main(int argc, char * argv[])
{
    if(argc>1)
    LoadGeometry(argv[1]);
}
