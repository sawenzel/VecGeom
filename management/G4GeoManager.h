/*
 * G4GeoManager.h
 *
 *  Created on: 09.03.2015
 *      Author: swenzel
 */

#ifndef VECGEOM_G4GEOMANAGER_H_
#define VECGEOM_G4GEOMANAGER_H_

#ifdef VECGEOM_GEANT4

#include "G4VSolid.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Navigator.hh"
#include "G4GeometryManager.hh"
#include "G4GDMLParser.hh"
#undef NDEBUG

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {


/// \brief singleton class to handle interaction with a G4 geometry.
/// \details Allows integration with G4 geometries mainly for debugging reasons.
///          Is not necessary for VecGeom library, and will only have source
///          compiled if the VECGEOM_ROOT flag is set by the compiler, activated
///          with -DGeant4=ON in CMake.
/// this class is more lightweight than the RootGeoManager; for the moment it only
/// keeps track of the world volume of the G4 geometry and a navigator object
/// in the future we might add maps between VecGeom and G4 logical volumes and placed volumes


class G4GeoManager {

public:
    /// Access singleton instance.
     static G4GeoManager& Instance() {
       static G4GeoManager instance;
       return instance;
     }

    // loads a G4 geometry from a gdmlfile
    void LoadG4Geometry( std::string gdmlfile ){
        G4GDMLParser parser;
        parser.Read( gdmlfile );

        // if there is an existing geometry
        if( fNavigator!=nullptr ) delete fNavigator;
        fNavigator = new G4Navigator();
        fNavigator->SetWorldVolume( const_cast<G4VPhysicalVolume *>(parser.GetWorldVolume()) );

        // voxelize
        G4GeometryManager::GetInstance()->CloseGeometry( fNavigator->GetWorldVolume() );
    }

    // sets a G4 geometry from existing G4PhysicalVolume
    void LoadG4Geometry( G4VPhysicalVolume * world ){
        // if there is an existing geometry
        if( fNavigator!=nullptr ) delete fNavigator;
        fNavigator = new G4Navigator();
        fNavigator->SetWorldVolume( world );

        // voxelize
        G4GeometryManager::GetInstance()->CloseGeometry( fNavigator->GetWorldVolume() );
    }

    G4Navigator * GetNavigator() const {
        assert( fNavigator!=nullptr && "Please load a G4geometry !! ");
        return fNavigator;
    }

private:
    G4Navigator * fNavigator; // a navigator object to navigate in detector with fWorld as geometry
                              // can access the world object via fNavigator->GetWorldVolume()

    // default private constructor
    G4GeoManager() : fNavigator(nullptr) {}
};

}} // end namespaces


#endif // VECGEOM_GEANT4

#endif /* VECGEOM_G4GEOMANAGER_H_ */
