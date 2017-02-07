/*
 * G4GeoManager.h
 *
 *  Created on: 09.03.2015
 *      Author: swenzel
 *
 *  Note: This class introduces incompatibilities between Geant4 and USolids, through
 *    the #include "G4GDMLParser.hh".  Here's the error message:
 *

${Geant4_DIR}/include/Geant4/G4UMultiUnion.hh:119:52:
error: no matching function for call to ‘CLHEP::Hep3Vector::Hep3Vector(<unresolved overloaded function type>, <unresolved overloaded function type>, <unresolved overloaded function type>)’
   G4ThreeVector transl(tr.fTr.x, tr.fTr.y, tr.fTr.z);
                                                    ^

 *
 */

#ifndef VECGEOM_G4GEOMANAGER_H_
#define VECGEOM_G4GEOMANAGER_H_

#if defined(VECGEOM_GEANT4) // and !defined(VECGEOM_USOLIDS)

#include "G4VSolid.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Navigator.hh"
#include "G4GeometryManager.hh"

#ifndef VECGEOM_USOLIDS
  #include "G4GDMLParser.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {


/// \brief singleton class to handle interaction with a G4 geometry.
/// \details Allows integration with G4 geometries mainly for debugging reasons.
///          Is not necessary for VecGeom library, and will only have source
///          compiled if the VECGEOM_GEANT4 flag is set by the compiler, activated
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
    void LoadG4Geometry( std::string gdmlfile, bool validate = false ){
#ifndef VECGEOM_USOLIDS
        G4GDMLParser parser;
        parser.Read( gdmlfile, validate );

        LoadG4Geometry( const_cast<G4VPhysicalVolume *>(parser.GetWorldVolume()) );
#else
        std::cerr<<"\n*** WARNING: LoadG4Geometry() is incompatible with USOLIDS!\n";
        std::cerr<<"      Please turn off USOLIDS and rebuild.  Aborting...\n\n";
        // Assert(false);
        exit(-1);
#endif
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
