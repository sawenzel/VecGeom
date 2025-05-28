/// @file PlacedMultiUnion.cpp
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "VecGeom/volumes/MultiUnion.h"

#ifndef VECCORE_CUDA
#ifdef VECGEOM_GEANT4
#include "G4MultiUnion.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include "G4Transform3D.hh"
#endif

#endif // VECCORE_CUDA

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedMultiUnion::PrintType() const
{
  printf("PlacedMultiUnion");
}

void PlacedMultiUnion::PrintType(std::ostream &s) const
{
  s << "PlacedMultiUnion";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedMultiUnion::ConvertToUnspecialized() const
{
  return new SimpleMultiUnion(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedMultiUnion::ConvertToGeant4() const
{
  G4MultiUnion *munion               = new G4MultiUnion("g4multiunion");
  const UnplacedMultiUnion *unplaced = GetUnplacedVolume();
  for (size_t i = 0; i < unplaced->GetNumberOfSolids(); ++i) {
    G4VSolid *g4solid = (G4VSolid *)unplaced->GetNode(i)->ConvertToGeant4();
    VECGEOM_VALIDATE(g4solid, << "Cannot convert component to Geant4 solid");
    auto trans = unplaced->GetNode(i)->GetTransformation();
    // Vector3D<double> point(1, 1, 1);
    // Vector3D<double> pnew = trans->Transform(point);
    const double *rotm = trans->Rotation();
    G4RotationMatrix g4rot(G4ThreeVector(rotm[0], rotm[3], rotm[6]), G4ThreeVector(rotm[1], rotm[4], rotm[7]),
                           G4ThreeVector(rotm[2], rotm[5], rotm[8]));
    G4Transform3D g4trans(g4rot, G4ThreeVector(trans->Translation(0), trans->Translation(1), trans->Translation(2)));
    // G4ThreeVector global(1, 1, 1);
    // G4ThreeVector local = g4trans.inverse()*G4Point3D(global);
    munion->AddNode(*g4solid, g4trans);
  }
  munion->Voxelize();
  return munion;
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedMultiUnion)

#endif

} // End global namespace
