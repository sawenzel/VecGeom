/*
 * ImportGDMLTest.cpp
 *
 *  Created on: Jun 24, 2014
 *      Author: swenzel
 */

#include "management/GeoManager.h"
#include "management/RootGeoManager.h"
#include "management/CppExporter.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "management/CudaManager.h"
#endif
#include "TGeoManager.h"
#include <cstdio>

//______________________________________________________________________________
void LoadVecGeomGeometry(bool printcontent = false)
{
   if( vecgeom::GeoManager::Instance().GetWorld() == NULL )
   {
    vecgeom::RootGeoManager::Instance().set_verbose(3);
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
    if( printcontent )
    {
        vecgeom::RootGeoManager::Instance().world()->PrintContent();
    }
   }
   else
   {
       printf("GeoManager already initialized; cannot load from ROOT file\n");
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

    vecgeom::GeomCppExporter::Instance().DumpGeometry( std::cout );

#ifdef VECGEOM_CUDA_INTERFACE
    #pragma message "VECGEOM NVCC enabled"
    if( vecgeom::GeoManager::Instance().GetWorld() != NULL ){
        printf("copying to GPU\n");
        vecgeom::CudaManager::Instance().set_verbose(3);
        vecgeom::CudaManager::Instance().LoadGeometry( vecgeom::GeoManager::Instance().GetWorld() );
        vecgeom::CudaManager::Instance().Synchronize();
    }
    printf("CUDA manager successfully finished");
    #endif

    return 0;
}
