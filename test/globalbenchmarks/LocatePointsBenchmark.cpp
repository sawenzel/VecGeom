/*
 * LocatePoints.cpp
 *
 *  Created on: 18.11.2014
 *      Author: swenzel
 */

// benchmarking the locate point functionality
#include "volumes/utilities/VolumeUtilities.h"
#include "base/Global.h"
#include "base/Vector3D.h"
#include "base/SOA3D.h"
#include "navigation/SimpleNavigator.h"
#include "navigation/NavStatePool.h"
#include "navigation/NavigationState.h"
#include "volumes/PlacedVolume.h"
#include "management/RootGeoManager.h"
#include "management/GeoManager.h"
#include "base/Stopwatch.h"

#ifdef VECGEOM_ROOT
#include "TGeoNavigator.h"
#include "TGeoNode.h"
#include "TGeoManager.h"
#include "TGeoBranchArray.h"
#include "TGeoBBox.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Navigator.hh"
#include "G4VPhysicalVolume.hh"
#include "G4ThreeVector.hh"
#include "G4TouchableHistoryHandle.hh"
#include "G4GDMLParser.hh"
#endif

#include <vector>
#include <iostream>

using namespace vecgeom;

// could template on storage type
void benchVecGeom( SOA3D<Precision> const & points, NavStatePool & statepool )
{
    Stopwatch timer;
    timer.Start();
    SimpleNavigator nav;
    for(int i=0; i< points.size(); ++i){
        nav.LocatePoint(
                GeoManager::Instance().GetWorld(), points[i], *(statepool[i]), true );
    }
    timer.Stop();
    std::cout << "VecGeom locate took " << timer.Elapsed() << " s\n";
}


void benchROOT( double * points, TGeoBranchArray ** pool, int npoints )
{
    Stopwatch timer;
    timer.Start();
    TGeoNavigator * nav = gGeoManager->GetCurrentNavigator();
    for(int i=0; i< npoints; ++i)
    {
        nav->FindNode( points[3*i], points[3*i+1], points[3*i+2] );
        pool[i]->InitFromNavigator(nav);
    }
    timer.Stop();
    std::cout << "ROOT locate took " << timer.Elapsed() << " s\n";
}


#ifdef VECGEOM_GEANT4
// we have to make sure that pool are valid touchables for this geometry
void benchGeant4( G4Navigator * nav, std::vector<G4ThreeVector> const & points,
                  G4TouchableHistory ** pool ){
    Stopwatch timer;
    timer.Start();
    G4VPhysicalVolume * vol;
    for(int i=0; i < points.size(); ++i){
        vol = nav->LocateGlobalPointAndSetup(points[i], NULL, false, true);

        // this takes ages:
        // nav->LocateGlobalPointAndUpdateTouchable(points[i], pool[i], false);
    }
    timer.Stop();
    std::cout << "GEANT4 locate took " << timer.Elapsed() << " s\n";
}
#endif


void benchCUDA(){
   // put the code here
};


int main( int argc, char * argv[] )
{
    // read in detector passed as argument
    if( argc > 1 ){
        RootGeoManager::Instance().set_verbose(3);
        RootGeoManager::Instance().LoadRootGeometry( std::string(argv[1]) );
    }
    else{
        std::cerr << "please give a ROOT geometry file\n";
        return 1;
    }
    // for geant4, we include the gdml file for the moment


    // setup data structures
    int npoints = 1000000;
    SOA3D<Precision> points(npoints);
    NavStatePool statepool(npoints, GeoManager::Instance().getMaxDepth() );

    // setup test points
    TGeoBBox const * rootbbox = dynamic_cast<TGeoBBox const*>(gGeoManager->GetTopVolume()->GetShape());
    Vector3D<Precision> bbox( rootbbox->GetDX(),
                   rootbbox->GetDY(),
                   rootbbox->GetDZ() );

    volumeUtilities::FillRandomPoints( bbox, points);

    benchVecGeom(points, statepool);

    TGeoBranchArray ** ROOTstatepool = new TGeoBranchArray*[npoints];
    double * rootpoints = new double[3*npoints];
#ifdef VECGEOM_GEANT4
    std::vector<G4ThreeVector> g4points;
#endif

    for( int i=0;i<npoints;++i )
    {
        rootpoints[3*i]=points.x(i);
        rootpoints[3*i+1]=points.y(i);
        rootpoints[3*i+2]=points.z(i);
//#if ROOTVERSION
        ROOTstatepool[i] = TGeoBranchArray::MakeInstance( GeoManager::Instance().getMaxDepth()+1 );
//#else
        //ROOTstatepool[i] = new TGeoBranchArray(GeoManager::Instance().getMaxDepth());
//#endif
#ifdef VECGEOM_GEANT4
        g4points.push_back(G4ThreeVector(points.x(i),points.y(i),points.z(i)));
#endif
    }

    benchROOT( rootpoints, ROOTstatepool, npoints );

    // *** checks can follow **** /

#ifdef VECGEOM_GEANT4
    // *** now try G4
    G4GDMLParser parser;
    parser.Read( argv[2] );

    G4Navigator * nav = new G4Navigator();
          nav->SetWorldVolume( const_cast<G4VPhysicalVolume *>(parser.GetWorldVolume()) );
          nav->LocateGlobalPointAndSetup(G4ThreeVector(0,0,0), false );
    G4TouchableHistory ** Geant4statepool = new G4TouchableHistory*[npoints];
    for( int i=0;i<npoints;++i )
        {
            Geant4statepool[i] = new G4TouchableHistory();//nav->CreateTouchableHistory();
        }

    benchGeant4( nav, g4points, Geant4statepool );
#endif

    return 0;
}


