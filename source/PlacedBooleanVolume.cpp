/*
 * PlacedBooleanVolume.cpp
 *
 *  Created on: Nov 7, 2014
 *      Author: swenzel
 */

#include "VecGeom/volumes/PlacedBooleanVolume.h"
#include "VecGeom/volumes/SpecializedBooleanVolume.h"
#include "VecGeom/volumes/UnplacedBooleanVolume.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/management/Logger.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/RNG.h"
#include <map>

#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#include "TGeoVolume.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"
#include "TGeoMatrix.h"
#include "TGeoManager.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4SubtractionSolid.hh"
#include "G4UnionSolid.hh"
#include "G4IntersectionSolid.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#endif

#include <iostream>

namespace vecgeom {

#ifdef VECGEOM_GEANT4
namespace {
bool HasGeant4UnsupportedHalfSpace(VPlacedVolume const *left, VPlacedVolume const *right)
{
  return BooleanHelper::ContainsHalfSpace(left->GetUnplacedVolume()) ||
         BooleanHelper::ContainsHalfSpace(right->GetUnplacedVolume());
}
} // namespace
#endif

template <>
VECCORE_ATT_HOST_DEVICE void PlacedBooleanVolume<kUnion>::PrintType() const
{
  VPlacedVolume const *pleft  = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *pright = GetUnplacedVolume()->GetRight();
  printf("PlacedBooleanVolume<kUnion>(");
  pleft->PrintType();
  printf(",");
  pright->PrintType();
  printf(")");
}

template <>
VECCORE_ATT_HOST_DEVICE void PlacedBooleanVolume<kIntersection>::PrintType() const
{
  VPlacedVolume const *pleft  = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *pright = GetUnplacedVolume()->GetRight();
  printf("PlacedBooleanVolume<kIntersection>(");
  pleft->PrintType();
  printf(",");
  pright->PrintType();
  printf(")");
}

template <>
VECCORE_ATT_HOST_DEVICE void PlacedBooleanVolume<kSubtraction>::PrintType() const
{
  VPlacedVolume const *pleft  = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *pright = GetUnplacedVolume()->GetRight();
  printf("PlacedBooleanVolume<kSubtraction>(");
  pleft->PrintType();
  printf(",");
  pright->PrintType();
  printf(")");
}

template <>
void PlacedBooleanVolume<kUnion>::PrintType(std::ostream &s) const
{
  VPlacedVolume const *pleft  = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *pright = GetUnplacedVolume()->GetRight();
  s << "PlacedBooleanVolume<kUnion>(";
  pleft->PrintType(s);
  s << ",";
  pright->PrintType(s);
  s << ")";
}

template <>
void PlacedBooleanVolume<kIntersection>::PrintType(std::ostream &s) const
{
  VPlacedVolume const *pleft  = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *pright = GetUnplacedVolume()->GetRight();
  s << "PlacedBooleanVolume<kIntersection>(";
  pleft->PrintType(s);
  s << ",";
  pright->PrintType(s);
  s << ")";
}

template <>
void PlacedBooleanVolume<kSubtraction>::PrintType(std::ostream &s) const
{
  VPlacedVolume const *pleft  = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *pright = GetUnplacedVolume()->GetRight();
  s << "PlacedBooleanVolume<kSubtraction>(";
  pleft->PrintType();
  s << ",";
  pright->PrintType();
  s << ")";
}

#ifdef VECGEOM_ROOT
template <>
TGeoShape const *PlacedBooleanVolume<kUnion>::ConvertToRoot() const
{
  VPlacedVolume const *left      = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *right     = GetUnplacedVolume()->GetRight();
  Transformation3D const *leftm  = left->GetTransformation();
  Transformation3D const *rightm = right->GetTransformation();

  TGeoUnion *node =
      new TGeoUnion(const_cast<TGeoShape *>(left->ConvertToRoot()), const_cast<TGeoShape *>(right->ConvertToRoot()),
                    Transformation3D::ConvertToTGeoMatrix(*leftm), Transformation3D::ConvertToTGeoMatrix(*rightm));
  return new TGeoCompositeShape("RootComposite", node);
}

template <>
TGeoShape const *PlacedBooleanVolume<kIntersection>::ConvertToRoot() const
{
  VPlacedVolume const *left      = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *right     = GetUnplacedVolume()->GetRight();
  Transformation3D const *leftm  = left->GetTransformation();
  Transformation3D const *rightm = right->GetTransformation();
  TGeoIntersection *node         = new TGeoIntersection(
      const_cast<TGeoShape *>(left->ConvertToRoot()), const_cast<TGeoShape *>(right->ConvertToRoot()),
      Transformation3D::ConvertToTGeoMatrix(*leftm), Transformation3D::ConvertToTGeoMatrix(*rightm));
  return new TGeoCompositeShape("RootComposite", node);
}

template <>
TGeoShape const *PlacedBooleanVolume<kSubtraction>::ConvertToRoot() const
{
  VPlacedVolume const *left      = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *right     = GetUnplacedVolume()->GetRight();
  Transformation3D const *leftm  = left->GetTransformation();
  Transformation3D const *rightm = right->GetTransformation();
  TGeoSubtraction *node          = new TGeoSubtraction(
      const_cast<TGeoShape *>(left->ConvertToRoot()), const_cast<TGeoShape *>(right->ConvertToRoot()),
      Transformation3D::ConvertToTGeoMatrix(*leftm), Transformation3D::ConvertToTGeoMatrix(*rightm));
  return new TGeoCompositeShape("RootComposite", node);
}
#endif

#ifdef VECGEOM_GEANT4
template <>
G4VSolid const *PlacedBooleanVolume<kUnion>::ConvertToGeant4() const
{
  VPlacedVolume const *left  = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *right = GetUnplacedVolume()->GetRight();

  if (!left->GetTransformation()->IsIdentity()) {
    VECGEOM_LOG(warning) << "Left transformations are not implemented\n";
  }
  if (HasGeant4UnsupportedHalfSpace(left, right)) return nullptr;

  auto const *leftSolid  = left->ConvertToGeant4();
  auto const *rightSolid = right->ConvertToGeant4();
  if (!leftSolid || !rightSolid) return nullptr;

  Transformation3D const *rightm = right->GetTransformation();
  G4RotationMatrix *g4rot        = new G4RotationMatrix();
  auto rot                       = rightm->Rotation();
  // HepRep3x3 seems? to be column major order:
  g4rot->set(CLHEP::HepRep3x3(rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8]));
  return new G4UnionSolid(GetLabel(), const_cast<G4VSolid *>(leftSolid), const_cast<G4VSolid *>(rightSolid), g4rot,
                          G4ThreeVector(rightm->Translation(0), rightm->Translation(1), rightm->Translation(2)));
}
template <>
G4VSolid const *PlacedBooleanVolume<kIntersection>::ConvertToGeant4() const
{
  VPlacedVolume const *left  = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *right = GetUnplacedVolume()->GetRight();

  if (!left->GetTransformation()->IsIdentity()) {
    VECGEOM_LOG(warning) << "Left transformations are not implemented";
  }
  if (HasGeant4UnsupportedHalfSpace(left, right)) return nullptr;

  auto const *leftSolid  = left->ConvertToGeant4();
  auto const *rightSolid = right->ConvertToGeant4();
  if (!leftSolid || !rightSolid) return nullptr;

  Transformation3D const *rightm = right->GetTransformation();
  G4RotationMatrix *g4rot        = new G4RotationMatrix();
  auto rot                       = rightm->Rotation();
  // HepRep3x3 seems? to be column major order:
  g4rot->set(CLHEP::HepRep3x3(rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8]));
  return new G4IntersectionSolid(GetLabel(), const_cast<G4VSolid *>(leftSolid), const_cast<G4VSolid *>(rightSolid),
                                 g4rot,
                                 G4ThreeVector(rightm->Translation(0), rightm->Translation(1), rightm->Translation(2)));
}
template <>
G4VSolid const *PlacedBooleanVolume<kSubtraction>::ConvertToGeant4() const
{
  VPlacedVolume const *left  = GetUnplacedVolume()->GetLeft();
  VPlacedVolume const *right = GetUnplacedVolume()->GetRight();

  if (!left->GetTransformation()->IsIdentity()) {
    VECGEOM_LOG(warning) << "Left transformations are not implemented";
  }
  if (HasGeant4UnsupportedHalfSpace(left, right)) return nullptr;

  auto const *leftSolid  = left->ConvertToGeant4();
  auto const *rightSolid = right->ConvertToGeant4();
  if (!leftSolid || !rightSolid) return nullptr;

  Transformation3D const *rightm = right->GetTransformation();
  G4RotationMatrix *g4rot        = new G4RotationMatrix();
  auto rot                       = rightm->Rotation();
  // HepRep3x3 seems? to be column major order:
  g4rot->set(CLHEP::HepRep3x3(rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8]));
  return new G4SubtractionSolid(GetLabel(), const_cast<G4VSolid *>(leftSolid), const_cast<G4VSolid *>(rightSolid),
                                g4rot,
                                G4ThreeVector(rightm->Translation(0), rightm->Translation(1), rightm->Translation(2)));
}
#endif

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN(SpecializedBooleanVolume, kUnion)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN(SpecializedBooleanVolume, kIntersection)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN(SpecializedBooleanVolume, kSubtraction)

#endif // VECCORE_CUDA

} // End namespace vecgeom
