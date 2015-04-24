/*
 * ResultComparator.h
 *
 *  Created on: Nov 27, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_RESULTCOMPARATOR_H_
#define VECGEOM_RESULTCOMPARATOR_H_

#include "volumes/PlacedVolume.h"
#include "base/Global.h"
#include "utilities/Visualizer.h"

// we should put this into a source file
#ifdef VECGEOM_DISTANCE_DEBUG
#include <iostream>
#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#include "TGeoManager.h"
#include "management/RootGeoManager.h"
#endif
#ifdef VECGEOM_GEANT4
#include "G4VSolid.hh"
#include "G4ThreeVector.hh"
#include "geomdefs.hh"
#endif
#endif

#include <sstream>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

// reusable utility function to compare distance results against ROOT/Geant4/ etc.
namespace DistanceComparator{


inline
    void CompareUnplacedContains( VPlacedVolume const * vol, bool vecgeomresult,
                              Vector3D<Precision> const & point ) {

    // convenience counter to enumerate the problems
    // TODO: make this thread safe
    static uint callcounter = 0;
    callcounter++;

    bool  mismatch=false;

    #ifdef VECGEOM_ROOT
        std::shared_ptr<TGeoShape const> rootshape( vol->ConvertToRoot() );
        bool rootresult = rootshape->Contains(
                (double*)&point[0] );
        mismatch |= rootresult != vecgeomresult;
//        if( rootresult != vecgeomresult ){
//            std::cout << "## WARNING (  " <<  callcounter  << " ) ## UnplacedContains VecGeom  " << vecgeomresult;
//            std::cout << " ROOT: " << rootresult << "\n";
//
//            // generate ROOT file and macro to debug this error offline / asynchronously
////    TGeoVolume * worldbackup = gGeoManager->GetTopVolume();
////    std::stringstream s; s << "DO-Debug" << errorcounter;
////    std::stringstream fn;
////    fn << "DO-Debug" << errorcounter << ".root";
////    gGeoManager->SetTopVolume( new TGeoVolume( s.str().c_str(), rootshape.get() ) );
////    gGeoManager->Export( fn.str().c_str() );
////
////    // recover old geometry
////    gGeoManager->SetTopVolume( worldbackup ); // gGeoManager->CloseGeometry();
////
//    }
    #endif

#ifdef VECGEOM_GEANT4
    std::shared_ptr<G4VSolid const> g4shape( vol->ConvertToGeant4() );
    auto g4inside = g4shape->Inside( G4ThreeVector(point[0],point[1],point[2] ));

    // due to possible ambigouity here use fix values from Geant4 ( 0 == kOutside ; 2 == kInside )
    bool bothoutside = (g4inside == 0 ) && (!vecgeomresult);
    bool bothinside  = (g4inside == 2 ) && (vecgeomresult) ;
    mismatch |= g4inside!=1 && ! ( bothoutside || bothinside );
//    if( ! ( bothoutside || bothinside ) ){
//                std::cout << "## WARNING (  " <<  callcounter  << " ) ## UnplacedContains VecGeom  " << vecgeomresult;
//                std::cout << " G4: " << g4inside << "\n";
//       }
#endif

    if( mismatch )
    {
        std::cout << "## WARNING (  " <<  callcounter  << " ) ## UnplacedContains VecGeom  " << vecgeomresult;
#ifdef VECGEOM_ROOT
        std::cout << " ROOT: " << rootresult;
#endif
#ifdef VECGEOM_GEANT4
        std::cout << " G4: " << g4inside;
#endif
        std::cout << "\n";
    }


}

    inline
    void PrintPointInformation( VPlacedVolume const * vol, Vector3D<Precision> const & point ){
        std::cout << " INFORMATION FOR POINT " << point << "\n";
        std::cout << " RELATIVE TO VOLUME " << vol << "\n";
        std::cout << " Volume Name " << vol->GetLabel() << "\n";
        std::cout << " Volume Type ";vol->PrintType(); std::cout << "\n";
        std::cout << " CONTAINS " << vol->Contains( point ) << "\n";
        std::cout << " INSIDE " << vol->Inside( point ) << "\n";
        std::cout << " SafetyToIn " << vol->SafetyToIn( point ) << "\n";
        std::cout << " SafetyToOut " << vol->SafetyToOut( point ) << "\n";
    }

    inline
    void CompareDistanceToIn( VPlacedVolume const * vol, Precision vecgeomresult,
                              Vector3D<Precision> const & point,
                              Vector3D<Precision> const & direction,
                              Precision const stepMax = VECGEOM_NAMESPACE::kInfinity ) {
        // this allows to compare distance calculations in each calculation (during a simulation)
        // and to report errors early

        // other packages usually require transformed points
        Vector3D<Precision> tpoint = vol->GetTransformation()->Transform(point);
        Vector3D<Precision> tdirection = vol->GetTransformation()->TransformDirection(direction);

        #ifdef VECGEOM_ROOT
        std::shared_ptr<TGeoShape const> rootshape( vol->ConvertToRoot() );
        Precision rootresult = rootshape->DistFromOutside(
                (double*)&tpoint[0],
                (double*)&tdirection[0], 3, stepMax );

	// can try of RootGeoManager has original shape
	double rootresult2 = RootGeoManager::Instance().tgeonode(vol)->GetVolume()->GetShape()->DistFromOutside(
											   (double*)&tpoint[0],(double*)&tdirection[0],3,stepMax);


        if( Abs(rootresult - vecgeomresult) > 1e-8 && Abs(rootresult - vecgeomresult) < 1e30 ){
            std::cout << "## WARNING ## DI VecGeom  " << vecgeomresult;
            std::cout << " ROOT: " << rootresult << "\n";
  std::cout << " ROOT2: " << rootresult2 << "\n";
        }

	

#endif

    #ifdef VECGEOM_GEANT4
        G4VSolid const * g4shape = vol->ConvertToGeant4();
        Precision g4result = g4shape->DistanceToIn(
                G4ThreeVector(tpoint[0],tpoint[1],tpoint[2]),
                G4ThreeVector(tdirection[0], tdirection[1], tdirection[2]));
        if( Abs(g4result - vecgeomresult) > 1e-8 && Abs(rootresult - vecgeomresult) < 1e30 ){
                std::cout << "## WARNING ## DI VecGeom  " << vecgeomresult;
                std::cout << " G4: " << g4result << "\n";
            }
        if(g4shape != NULL) delete g4shape;
    #endif
    }

inline
    void CompareDistanceToOut( VPlacedVolume const * vol, Precision vecgeomresult,
                                  Vector3D<Precision> const & point,
                                  Vector3D<Precision> const & direction,
                                  Precision const stepMax = VECGEOM_NAMESPACE::kInfinity ) {
#ifdef VECGEOM_ROOT
    std::shared_ptr<TGeoShape const> rootshape( vol->ConvertToRoot() );
    Precision rootresult = rootshape->DistFromInside(
            (double*)&point[0],
            (double*)&direction[0], 3, stepMax );
    if( vecgeomresult < 0 )
        std::cout << "## WARNING ## DO VecGeom negative (ROOT = " << rootresult << ")\n";
    if( Abs(rootresult - vecgeomresult) > 1e-8 ){
                std::cout << "## WARNING ## DO VecGeom  " << vecgeomresult;
                std::cout << " ROOT: " << rootresult << "\n";
                PrintPointInformation(vol, point);
                Visualizer vis;
    }
#endif

#ifdef VECGEOM_GEANT4
    Precision g4result = vol->ConvertToGeant4()->DistanceToOut(
            G4ThreeVector(point[0],point[1],point[2]),
            G4ThreeVector(direction[0], direction[1], direction[2]), false);
    if( Abs(g4result - vecgeomresult) > 1e-8 ){
                std::cout << "## WARNING ## DO VecGeom  " << vecgeomresult;
                std::cout << " G4: " << g4result << "\n";
       }
    #endif
    }
} // end inner namespace

}} // end namespace


#endif /* VECGEOM_RESULTCOMPARATOR_H_ */
