//!    \file Middleware.h
//!    \brief Declares the class for converting files from DOM to a VecGeom volume and back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#pragma once

#ifndef VGDMLMiddleware_h
#define VGDMLMiddleware_h

#include <string>
#include <vector>
#include <map>

#include "xercesc/util/XercesDefs.hpp"

#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/BooleanStruct.h"
#include "VecGeom/volumes/LogicalVolume.h"

#include "Auxiliary.h"
#include "MaterialInfo.h"
#include "RegionInfo.h"

XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class DOMLSParser;
class DOMNode;
class DOMNamedNodeMap;
XERCES_CPP_NAMESPACE_END

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
class VUnplacedVolume;
class UnplacedTessellated;
// class UnplacedOrb;
// class UnplacedBox;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

namespace vgdml {
class Middleware {
public:
  using MaterialMap_t         = std::map<std::string, vgdml::Material>;
  using VolumeMatMap_t        = std::map<int, vgdml::Material>;
  using VolumeMap_t           = std::map<std::string, vecgeom::LogicalVolume *>;
  using VolumeAuxiliaryInfo_t = std::map<int, std::vector<Auxiliary>>;
  using UserInfo_t            = std::vector<Auxiliary>;

  bool Load(XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument const *aDOMDocument);
  XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *Save(void const *);

  /// Return map of GDML material name to material data
  MaterialMap_t const &GetMaterialMap() const { return materialMap; }

  /// Return map of VecGeom LogicalVolume Id to GDML material
  VolumeMatMap_t const &GetVolumeMatMap() const { return volumeMaterialMap; }

  /// Return map of VecGeom LogicalVolume Id to GDML material
  VolumeMap_t const &GetVolumeMap() const { return volumeMap; }

  /// Return map of VecGeom LogicalVolume Id to list of GDML auxiliary tags for that volume
  VolumeAuxiliaryInfo_t const &GetVolumeAuxiliaryInfo() const { return volumeAuxiliaryInfo; }

  /// Return list of auxiliary tags in GDML userinfo tag
  UserInfo_t const &GetUserInfo() const { return userInfo; }

private:
  bool processNode(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processSolid(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processLogicVolume(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processPhysicalVolume(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode,
                             vecgeom::LogicalVolume *motherLogical);
  bool processWorld(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processConstant(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processPosition(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processScale(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processRotation(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);

  bool processIsotope(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processElement(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processMaterial(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  bool processAuxiliary(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode, Auxiliary &aux);
  bool processUserInfo(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);

  bool processFacet(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode,
                    vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTessellated &storage);
  template <vecgeom::BooleanOperation Op>
  vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *processBoolean(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *processMultiUnion(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  vecgeom::VECGEOM_IMPL_NAMESPACE::VPlacedVolume const *processMultiUnionNode(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);

  vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *processOrb(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processBox(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTube(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processElTube(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processCutTube(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processCone(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processElCone(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processPolycone(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processGenPolycone(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processPolyhedron(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTorus(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processSphere(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processEllipsoid(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processParallelepiped(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTrd(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTrapezoid(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processGenTrap(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processParaboloid(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processHype(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTesselated(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processExtruded(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processTet(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *processScaledShape(
      XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);

  double GetLengthMultiplier(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);
  double GetAngleMultiplier(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode);

private:
  std::map<std::string, vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *> unplacedVolumeMap;
  std::map<std::string, double> constantMap;
  std::map<std::string, vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>> positionMap;
  std::map<std::string, vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>> scaleMap;
  std::map<std::string, vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>> rotationMap;
  std::map<std::string, vgdml::Isotope> isotopeMap;
  std::map<std::string, vgdml::Element> elementMap;

  MaterialMap_t materialMap;                 ///< map of material name to a material record
  VolumeMatMap_t volumeMaterialMap;          ///< map of VecGeom logical volume id to a material record
  VolumeMap_t volumeMap;                     ///< map of logical volume name to logical volume pointer
  VolumeAuxiliaryInfo_t volumeAuxiliaryInfo; ///< map of VecGeom logical volume id to a list of auxiliary tags
  UserInfo_t userInfo;                       ///< list of auxiliary tags in userinfo

  double GetDoubleAttribute(std::string const &attrName,
                            XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes);
};
} // namespace vgdml
#endif
