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
error: no matching function for call to ‘CLHEP::Hep3Vector::Hep3Vector(<unresolved overloaded function type>,
<unresolved overloaded function type>, <unresolved overloaded function type>)’
   G4ThreeVector transl(tr.fTr.x, tr.fTr.y, tr.fTr.z);
                                                    ^

 *
 */

#ifndef VECGEOM_G4GEOMANAGER_H_
#define VECGEOM_G4GEOMANAGER_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Assert.h"

class G4Navigator;
class G4VPhysicalVolume;

#include "VecGeom/base/Global.h"

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
  static G4GeoManager &Instance()
  {
    static G4GeoManager instance;
    return instance;
  }

  // loads a G4 geometry from a gdmlfile
  void LoadG4Geometry(std::string gdmlfile, bool validate = false);

  // converts to a G4 geometry from ROOT using VGM
  // expects a valid gGeoManager object for the import
  // returns world pointer of in-memory representation of G4
  // the geometry still has to be closed + fully initialized afterwards using LoadG4Geometry
  G4VPhysicalVolume *GetG4GeometryFromROOT();

  // sets and finalizes G4 geometry from existing G4PhysicalVolume
  void LoadG4Geometry(G4VPhysicalVolume *world);

  G4Navigator *GetNavigator() const
  {
    VECGEOM_VALIDATE(fNavigator != nullptr, << "Please load a G4geometry !! ");
    return fNavigator;
  }

private:
  G4Navigator *fNavigator; // a navigator object to navigate in detector with fWorld as geometry
                           // can access the world object via fNavigator->GetWorldVolume()

  // default private constructor
  G4GeoManager();
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VECGEOM_G4GEOMANAGER_H_ */
