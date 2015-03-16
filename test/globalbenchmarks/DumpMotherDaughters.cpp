/*
 * DumpMotherDaughters.cpp
 *
 *  Created on: 20150227
 *      Author: Guilherme Lima
 */

#include "volumes/PlacedVolume.h"
#include "management/RootGeoManager.h"
#include "management/GeoManager.h"
#include <iostream>

using namespace vecgeom;

int main( int argc, char * argv[] ) {

    // load geometry from root file passed as argument
    char* geometry = new char[256];
    if( argc > 1 ) {
        geometry = argv[1];
    }
    else {
        std::cerr << "No ROOT geometry file provided -- using default geom1.root\n";
        strcpy(geometry,"geom1.root");
    }
    //RootGeoManager::Instance().set_verbose(3);
    RootGeoManager::Instance().LoadRootGeometry( geometry );
    std::cout<<"World volume: <"<< GeoManager::Instance().GetWorld()->GetLabel() <<">\n";

    char* testName = new char[256];
    if( argc > 2 ) {
        testName = argv[2];
    }
    else {
        std::cerr << "NO test volume name provided -- using world volume as default\n";
        strcpy(testName, GeoManager::Instance().GetWorld()->GetLabel().c_str());
    }
    std::cout<<" testName=<"<< testName <<">\n";

    VPlacedVolume const* testVol = GeoManager::Instance().FindPlacedVolume(testName);
    std::cout<< *testVol <<"\n\n";

    const Vector<VPlacedVolume const*>& dauVector = testVol->daughters();
    int idau = 0;
    for (Vector<Daughter>::const_iterator j = dauVector.cbegin(),
             jEnd = dauVector.cend(); j != jEnd; ++j) {
        //if ((*j)->Contains( points[i] )) {
        std::cout<<"*** Daughter "<< idau <<" --> "<< (*j)->GetLabel() <<" "<< *(*j) <<"\n\n";
        ++idau;
    }

    delete[] testName;
    delete[] geometry;
    return 0;
}


