//!    \file ReflFactory.cpp
//!    \brief Reflection factory for placed volumes
//!
//!    \authors Author:  Andrei Gheata <andrei.gheata@cern.ch>
//!

#include "VecGeom/management/ReflFactory.h"

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedScaledShape.h"

namespace vecgeom {
inline namespace cxx {

ReflFactory::ReflFactory()
{
  fNameExtension = "_refl";
}

bool ReflFactory::Place(Transformation3D const &pureTransform3D, Vector3 const &scale, std::string const &name,
                        vecgeom::LogicalVolume *LV, vecgeom::LogicalVolume *motherLV, int copyNo)
{
  std::string name_refl = name + "_refl";
  if (fVerboseLevel > 0) {
    std::cerr << "Place " << name << " lv " << LV << " " << LV->GetName() << " inside " << motherLV->GetName()
              << std::endl;
  }

  //
  // reflection IS NOT present in transform3D
  //

  if (!IsReflection(scale)) {
    // Place with pureTransform3D in the mother
    auto pv1 = LV->Place(name.c_str(), &pureTransform3D);
    pv1->SetCopyNo(copyNo);
    motherLV->PlaceDaughter(pv1);

    if (LogicalVolume *reflMotherLV = GetReflectedLV(motherLV)) {
      // if mother was reflected
      // reflect this LV and place it in reflected mother
      auto newTransform3D = ConvertScaledToPureTransformation(pureTransform3D, scale);

      auto pv2 = ReflectLV(LV, scale)->Place(name_refl.c_str(), &newTransform3D);
      pv2->SetCopyNo(copyNo);
      reflMotherLV->PlaceDaughter(pv2);
    }

    return true;
  }

  //
  //  reflection IS present in transform3D. Only (1,1,-1) currently supported
  //
  // assert((scale - Vector3(1, 1, -1)).Mag() < 1.e-8);

  auto pv1 = ReflectLV(LV, scale)->Place(name_refl.c_str(), &pureTransform3D);
  pv1->SetCopyNo(copyNo);
  motherLV->PlaceDaughter(pv1);

  if (LogicalVolume *reflMotherLV = GetReflectedLV(motherLV)) {

    // if mother was reflected
    // place the refLV consituent in reflected mother
    auto newTransform3D = ConvertScaledToPureTransformation(pureTransform3D, scale);

    auto pv2 = LV->Place(name.c_str(), &newTransform3D);
    pv2->SetCopyNo(copyNo);
    reflMotherLV->PlaceDaughter(pv2);
  }
  return true;
}

vecgeom::LogicalVolume *ReflFactory::GetConstituentLV(vecgeom::LogicalVolume *reflLV) const
{
  // Returns the consituent volume of the given reflected volume,
  // nullptr if the given reflected volume was not found.

  LogicalVolumesMapIterator it = fReflectedLVMap.find(reflLV);
  if (it == fReflectedLVMap.end()) return nullptr;
  return (*it).second;
}

vecgeom::LogicalVolume *ReflFactory::GetReflectedLV(vecgeom::LogicalVolume *lv) const
{
  // Returns the reflected volume of the given consituent volume,
  // nullptr if the given volume was not reflected.

  LogicalVolumesMapIterator it = fConstituentLVMap.find(lv);
  if (it == fConstituentLVMap.end()) return nullptr;
  return (*it).second;
}

vecgeom::Transformation3D ReflFactory::ConvertScaledToPureTransformation(Transformation3D const &pureTransform3D,
                                                                         Vector3 const &scale)
{
  Transformation3D scale3D(0, 0, 0, 0, 0, 0, scale[0], scale[1], scale[2]);
  Transformation3D scale3Dinv;
  scale3Dinv = scale3D.Inverse();
  // long-cut for scale * pureTransfore * scale.Inverse()
  Transformation3D newTransform3D = scale3D;
  newTransform3D.MultiplyFromRight(pureTransform3D);
  newTransform3D.MultiplyFromRight(scale3Dinv);
  return newTransform3D;
}

bool ReflFactory::IsConstituent(vecgeom::LogicalVolume const *lv) const
{
  // Returns true if the given volume has been already reflected
  // (is in the map of constituent volumes).

  return (fConstituentLVMap.find((vecgeom::LogicalVolume *)(lv)) != fConstituentLVMap.end());
}

bool ReflFactory::IsReflected(vecgeom::LogicalVolume const *lv) const
{
  // Returns true if the given volume is a reflected volume
  // (is in the map reflected  volumes).

  return (fReflectedLVMap.find((vecgeom::LogicalVolume *)(lv)) != fReflectedLVMap.end());
}

vecgeom::LogicalVolume *ReflFactory::ReflectLV(vecgeom::LogicalVolume *LV, Vector3 const &scale)
{
  // Gets/creates the reflected solid and logical volume
  // and copies + transforms LV daughters.
  vecgeom::LogicalVolume *refLV = GetReflectedLV(LV);
  if (!refLV) {
    // create new (reflected) objects
    refLV = CreateReflectedLV(LV, scale);
    // process daughters
    ReflectDaughters(LV, refLV, scale);
  }
  return refLV;
}

vecgeom::LogicalVolume *ReflFactory::CreateReflectedLV(vecgeom::LogicalVolume *LV, Vector3 const &scale)
{
  // Creates the reflected solid and logical volume
  // and add the logical volumes pair in the maps.

  // consistency check
  //
  assert(fReflectedLVMap.find(LV) == fReflectedLVMap.end());
  vecgeom::UnplacedScaledShape *refSolid =
      new vecgeom::UnplacedScaledShape(LV->GetUnplacedVolume(), scale[0], scale[1], scale[2]);

  vecgeom::LogicalVolume *refLV = new LogicalVolume((std::string(LV->GetName()) + "_refl").c_str(), refSolid);
  fConstituentLVMap[LV]         = refLV;
  fReflectedLVMap[refLV]        = LV;

  return refLV;
}

void ReflFactory::ReflectDaughters(vecgeom::LogicalVolume *LV, vecgeom::LogicalVolume *refLV, Vector3 const &scale)
{
  // Reflects daughters recursively.
  for (auto *dPV : LV->GetDaughters()) {
    ReflectPlacedVolume(dPV, refLV, scale);
  }
}

void ReflFactory::ReflectPlacedVolume(vecgeom::VPlacedVolume const *dPV, vecgeom::LogicalVolume *refLV,
                                      Vector3 const &scale)
{
  // Copies and transforms daughter of PVPlacement type of
  // a constituent volume into a reflected volume.

  auto dLV = (vecgeom::LogicalVolume *)dPV->GetLogicalVolume();

  // update daughter transformation
  //
  auto dt = ConvertScaledToPureTransformation(*dPV->GetTransformation(), scale);

  vecgeom::LogicalVolume *refDLV;

  if (!IsReflected(dLV)) {

    // get reflected volume if already created, or create one
    refDLV = GetReflectedLV(dLV);

    if (refDLV == nullptr) {
      // create new daughter solid and logical volume
      //
      if (fVerboseLevel > 0) {
        std::cerr << "Daughter: " << dPV << "  " << dLV->GetName() << "_" << dPV->GetCopyNo() << " will be reflected."
                  << std::endl;
      }
      refDLV = CreateReflectedLV(dLV, scale);

      // recursive call
      //
      ReflectDaughters(dLV, refDLV, scale);
    }

  } else {
    if (fVerboseLevel > 0) {
      std::cerr << "Daughter: " << dPV << "  " << dLV->GetName() << "_" << dPV->GetCopyNo()
                << " will be reconstitued.\n";
    }

    refDLV = GetConstituentLV(dLV);
  }
  // create new daughter physical volume
  // with updated transformation
  auto pv = refDLV->Place(dPV->GetName(), &dt);
  pv->SetCopyNo(dPV->GetCopyNo());
  refLV->PlaceDaughter(pv);
}

void ReflFactory::Clean()
{
  fConstituentLVMap.clear();
  fReflectedLVMap.clear();
}

} // namespace cxx
} // namespace vecgeom