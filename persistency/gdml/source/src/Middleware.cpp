//!    \file Middleware.cpp
//!    \brief Defines the class for converting files from DOM to a VecGeom volume and back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#include "Middleware.h"
#include "Helper.h"
#include "ReflFactory.h"
#include "xercesc/dom/DOMNode.hpp"
#include "xercesc/dom/DOMNodeList.hpp"
#include "xercesc/dom/DOMDocument.hpp"
#include "xercesc/dom/DOMElement.hpp"
#include "xercesc/dom/DOMNamedNodeMap.hpp"
#include "xercesc/util/XMLString.hpp"

#include "VecGeom/volumes/UnplacedOrb.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeom/volumes/UnplacedTube.h"
#include "VecGeom/volumes/UnplacedEllipticalTube.h"
#include "VecGeom/volumes/UnplacedCone.h"
#include "VecGeom/volumes/UnplacedEllipticalCone.h"
#include "VecGeom/volumes/UnplacedPolycone.h"
#include "VecGeom/volumes/UnplacedGenericPolycone.h"
#include "VecGeom/volumes/UnplacedPolyhedron.h"
#include "VecGeom/volumes/UnplacedTorus2.h"
#include "VecGeom/volumes/UnplacedSphere.h"
#include "VecGeom/volumes/UnplacedEllipsoid.h"
#include "VecGeom/volumes/UnplacedParallelepiped.h"
#include "VecGeom/volumes/UnplacedTrd.h"
#include "VecGeom/volumes/UnplacedParaboloid.h"
#include "VecGeom/volumes/UnplacedTrapezoid.h"
#include "VecGeom/volumes/UnplacedGenTrap.h"
#include "VecGeom/volumes/UnplacedHype.h"
#include "VecGeom/volumes/UnplacedCutTube.h"
#include "VecGeom/volumes/UnplacedBooleanVolume.h"
#include "VecGeom/volumes/UnplacedMultiUnion.h"
#include "VecGeom/volumes/UnplacedTessellated.h"
#include "VecGeom/volumes/UnplacedTet.h"
#include "VecGeom/volumes/UnplacedScaledShape.h"
#include "VecGeom/volumes/UnplacedSExtruVolume.h"

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/management/Logger.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/management/GeoManager.h"

#include <iostream>
#include <string>
#include <vector>
#include <array>

#include <iomanip>

// requires lengthMultiplier, angleMultiplier and attributes already declared
#define DECLAREANDGETLENGTVAR(x) auto const x = lengthMultiplier * GetDoubleAttribute(#x, attributes);
#define DECLAREANDGETANGLEVAR(x) auto const x = angleMultiplier * GetDoubleAttribute(#x, attributes);
#define DECLAREANDGETPLAINVAR(x) auto const x = GetDoubleAttribute(#x, attributes);
#define DECLAREANDGETINTVAR(x) auto const x = GetAttribute<int>(#x, attributes);

#define DECLAREHALF(x) auto const half##x = x / 2.;

namespace {
#ifdef GDMLDEBUG
constexpr bool debug = true;
#else
constexpr bool debug = false;
#endif

// a container with simple solids description (fixed number of attributes, no loops), generated from the schema
auto const gdmlSolids = std::map<std::string, std::vector<std::string>>{
    {"multiUnion", {}},
    {"reflectedSolid", {"solid", "sx", "sy", "sz", "rx", "ry", "rz", "dx", "dy", "dz"}},
    {"scaledSolid", {}},
    {"union", {}},
    {"subtraction", {}},
    {"intersection", {}},
    {"box", {"x", "y", "z"}},
    {"twistedbox", {"x", "y", "z", "PhiTwist"}},
    {"twistedtrap", {"PhiTwist", "z", "Theta", "Phi", "y1", "x1", "y2", "x2", "x3", "x4", "Alph"}},
    {"twistedtrd", {"PhiTwist", "z", "y1", "x1", "y2", "x2"}},
    {"paraboloid", {"rlo", "rhi", "dz"}},
    {"sphere", {"rmin", "rmax", "startphi", "deltaphi", "starttheta", "deltatheta"}},
    {"ellipsoid", {"ax", "by", "cz", "zcut1", "zcut2"}},
    {"tube", {"z", "rmin", "rmax", "startphi", "deltaphi"}},
    {"twistedtubs",
     {"twistedangle", "endinnerrad", "endouterrad", "midinnerrad", "midouterrad", "negativeEndz", "positiveEndz",
      "zlen", "nseg", "totphi", "phi"}},
    {"cutTube", {"z", "rmin", "rmax", "startphi", "deltaphi", "lowX", "lowY", "lowZ", "highX", "highY", "highZ"}},
    {"cone", {"z", "rmin1", "rmin2", "rmax1", "rmax2", "startphi", "deltaphi"}},
    {"elcone", {"dx", "dy", "zmax", "zcut"}},
    {"polycone", {"deltaphi", "startphi"}},
    {"genericPolycone", {"deltaphi", "startphi"}},
    {"para", {"x", "y", "z", "alpha", "theta", "phi"}},
    {"trd", {"x1", "x2", "y1", "y2", "z"}},
    {"trap", {"z", "theta", "phi", "y1", "x1", "x2", "alpha1", "y2", "x3", "x4", "alpha2"}},
    {"torus", {"rmin", "rmax", "rtor", "startphi", "deltaphi"}},
    {"orb", {"r"}},
    {"polyhedra", {"startphi", "deltaphi", "numsides"}},
    {"genericPolyhedra", {"startphi", "deltaphi", "numsides"}},
    {"xtru", {}},
    {"hype", {"rmin", "rmax", "inst", "outst", "z"}},
    {"eltube", {"dx", "dy", "dz"}},
    {"tet", {"vertex1", "vertex2", "vertex3", "vertex4"}},
    {"arb8",
     {"v1x", "v1y", "v2x", "v2y", "v3x", "v3y", "v4x", "v4y", "v5x", "v5y", "v6x", "v6y", "v7x", "v7y", "v8x", "v8y",
      "dz"}},
    {"tessellated", {}}};

template <typename T = std::string>
T GetAttribute(std::string const &attrName, XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes)
{
  return vgdml::Helper::GetAttribute<T>(attrName, theAttributes);
}

vecgeom::Transformation3D makeRotationMatrixFromCartesianAngles(double x, double y, double z)
{
  // Similar to the way Geant4 constructs this
  vecgeom::Transformation3D rotxyz;
  if (x == 0 && y == 0 && z == 0) return rotxyz;
  rotxyz.RotateX(x);
  rotxyz.RotateY(y);
  rotxyz.RotateZ(z);
  rotxyz.Rectify();
  return rotxyz;
}
} // namespace

namespace vgdml {
extern template std::string Helper::Transcode(const XMLCh *const anXMLstring);
extern template double Helper::Transcode(const XMLCh *const anXMLstring);
extern template int Helper::Transcode(const XMLCh *const anXMLstring);
extern template std::string Helper::GetAttribute(std::string const &attrName,
                                                 XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes);
extern template double Helper::GetAttribute(std::string const &attrName,
                                            XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes);

double Middleware::GetDoubleAttribute(std::string const &attrName,
                                      XERCES_CPP_NAMESPACE_QUALIFIER DOMNamedNodeMap const *theAttributes)
{
  auto const simpleDouble = vgdml::Helper::GetAttribute<double>(attrName, theAttributes);
  if (std::isnan(simpleDouble)) {
    auto const referencedConstant = constantMap[vgdml::Helper::GetAttribute(attrName, theAttributes)];
    return referencedConstant;
  }
  return simpleDouble;
}

bool Middleware::Load(XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument const *aDOMDocument)
{
  auto *rootDocElement = aDOMDocument->getDocumentElement();
  if (!rootDocElement) return false;
  auto *rootDocNode = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMNode *>(rootDocElement);
  return processNode(rootDocNode);
}

XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *Middleware::Save(void const *) { exit(-1); }

bool Middleware::processNode(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processNode: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const name = Helper::Transcode(aDOMNode->getNodeName());

  if (gdmlSolids.count(name)) {
    auto const success = processSolid(aDOMNode);
    return success;
  } else if (name == "constant") {
    auto const success = processConstant(aDOMNode);
    return success;
  } else if (name == "position") {
    auto const success = processPosition(aDOMNode);
    return success;
  } else if (name == "scale") {
    auto const success = processScale(aDOMNode);
    return success;
  } else if (name == "rotation") {
    auto const success = processRotation(aDOMNode);
    return success;
  } else if (name == "isotope") {
    auto const success = processIsotope(aDOMNode);
    return success;
  } else if (name == "element") {
    auto const success = processElement(aDOMNode);
    return success;
  } else if (name == "material") {
    auto const success = processMaterial(aDOMNode);
    return success;
  } else if (name == "volume") {
    auto const success = processLogicVolume(aDOMNode);
    return success;
  } else if (name == "world") {
    auto const success = processWorld(aDOMNode);
    return success;
  } else if (name == "userinfo") {
    auto const success = processUserInfo(aDOMNode);
    return success;
  }
  auto result = true;
  //  if do not know what to do, process the children
  for (auto *child = aDOMNode->getFirstChild(); child != nullptr; child = child->getNextSibling()) {
    auto const nodeResult = processNode(child);
    result                = result && nodeResult;
  }
  return result;
}

bool Middleware::processConstant(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processConstant: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const constantName      = GetAttribute("name", attributes);
  auto const constantValue     = GetAttribute<double>("value", attributes);
  auto const success           = constantMap.insert(std::make_pair(constantName, constantValue)).second;
  if (!success) {
    VECGEOM_LOG(warning) << "Middleware::processNode: failed to insert constant with name " << constantName
                         << " and value " << constantValue;
  }
  return success;
}

bool Middleware::processPosition(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processPosition: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const positionName      = GetAttribute("name", attributes);
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(x)
  DECLAREANDGETLENGTVAR(y)
  DECLAREANDGETLENGTVAR(z)
  auto const positionValue = vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>{x, y, z};
  if (positionName.empty()) {
    // Anonymous positions inside volumes are OK
    return true;
  }

  auto const success = positionMap.insert(std::make_pair(positionName, positionValue)).second;
  if (!success) {
    VECGEOM_LOG(warning) << "Middleware::processNode: failed to insert position with name " << positionName
                         << " and value " << positionValue;
  }
  return success;
}

bool Middleware::processScale(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processScale: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const scaleName         = GetAttribute("name", attributes);
  DECLAREANDGETPLAINVAR(x)
  DECLAREANDGETPLAINVAR(y)
  DECLAREANDGETPLAINVAR(z)
  auto const scaleValue = vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>{x, y, z};
  auto const success    = scaleMap.insert(std::make_pair(scaleName, scaleValue)).second;
  if (!success) {
    VECGEOM_LOG(warning) << "Middleware::processNode: failed to insert position with name " << scaleName
                         << " and value " << scaleValue;
  }
  return success;
}

bool Middleware::processRotation(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processRotation: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const rotationName      = GetAttribute("name", attributes);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETANGLEVAR(x)
  DECLAREANDGETANGLEVAR(y)
  DECLAREANDGETANGLEVAR(z)
  auto const rotationValue = vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>{x, y, z};
  auto const success       = rotationMap.insert(std::make_pair(rotationName, rotationValue)).second;
  if (!success) {
    VECGEOM_LOG(warning) << "Middleware::processNode: failed to insert rotation with name " << rotationName
                         << " and value " << rotationValue;
  }
  return success;
}

bool Middleware::processIsotope(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processIsotope: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  Isotope anIsotope;
  auto const *const attributes = aDOMNode->getAttributes();
  auto const isotopeName       = GetAttribute("name", attributes);
  auto const isotopeZ          = GetAttribute("Z", attributes);
  auto const isotopeN          = GetAttribute("N", attributes);
  anIsotope.attributes["Z"]    = isotopeZ;
  anIsotope.attributes["N"]    = isotopeN;

  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    if (debug) {
      VECGEOM_LOG(debug) << "procIsotope Child: " << Helper::GetNodeInformation(it);
    }
    if (it->getNodeType() == XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::ELEMENT_NODE) {
      auto const *const childAttributes = it->getAttributes();
      auto const atomType               = GetAttribute("type", childAttributes);
      auto const atomUnit               = GetAttribute("unit", childAttributes);
      auto const atomValue              = GetAttribute("value", childAttributes);
      anIsotope.attributes["atomType"]  = atomType;
      anIsotope.attributes["atomUnit"]  = atomUnit;
      anIsotope.attributes["atomValue"] = atomValue;
    }
  }
  auto const success = isotopeMap.insert(std::make_pair(isotopeName, anIsotope)).second;
  if (!success) {
    VECGEOM_LOG(warning) << "Middleware::processIsotope: failed to insert isotope with name " << isotopeName;
  }
  return success;
}

bool Middleware::processElement(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processElement: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const nAttributes       = attributes->getLength();
  auto const elementName       = GetAttribute("name", attributes);
  Element anElement;
  for (auto ind = 0u; ind < nAttributes; ++ind) {
    auto const *const anAttribute = attributes->item(ind);
    auto const attributeName      = Helper::Transcode(anAttribute->getNodeName());
    auto const attributeValue     = GetAttribute(attributeName, attributes);
    if (attributeName != "name") anElement.attributes[attributeName] = attributeValue;
  }
  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    if (debug) {
      VECGEOM_LOG(debug) << "procElement Child: " << Helper::GetNodeInformation(it);
    }
    if (it->getNodeType() == XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::ELEMENT_NODE) {
      auto const *const childAttributes = it->getAttributes();
      auto const theNodeName            = Helper::Transcode(it->getNodeName());
      if (theNodeName == "atom") {
        auto const atomValue              = GetAttribute("value", childAttributes);
        anElement.attributes["atomValue"] = atomValue;
      } else { // theNodeName == "fraction"
        auto const isotope                  = GetAttribute("ref", childAttributes);
        auto const fraction                 = GetAttribute("n", childAttributes);
        anElement.isotopeFractions[isotope] = fraction;
      }
    }
  }
  auto const success = elementMap.insert(std::make_pair(elementName, anElement)).second;
  if (!success) {
    VECGEOM_LOG(warning) << "Middleware::processElement: failed to insert element with name " << elementName;
  }
  return success;
}

bool Middleware::processMaterial(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processMaterial: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const nAttributes       = attributes->getLength();
  auto const materialName      = GetAttribute("name", attributes);
  Material aMaterial;
  for (auto ind = 0u; ind < nAttributes; ++ind) {
    auto const *const anAttribute = attributes->item(ind);
    auto const attributeName      = Helper::Transcode(anAttribute->getNodeName());
    auto const attributeValue     = GetAttribute(attributeName, attributes);
    if (attributeName != "name") aMaterial.attributes[attributeName] = attributeValue;
  }
  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    if (debug) {
      VECGEOM_LOG(debug) << "procMaterial Child: " << Helper::GetNodeInformation(it);
    }
    if (it->getNodeType() == XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::ELEMENT_NODE) {
      auto const *const childAttributes = it->getAttributes();
      auto const theNodeName            = Helper::Transcode(it->getNodeName());
      if (theNodeName == "atom") {
        auto const atomValue              = GetAttribute("value", childAttributes);
        aMaterial.attributes["atomValue"] = atomValue;
      } else if (theNodeName == "D") {
        auto const Dvalue             = GetAttribute("value", childAttributes);
        auto const Dunit              = GetAttribute("unit", childAttributes);
        auto const Dtype              = GetAttribute("type", childAttributes);
        aMaterial.attributes["D"]     = Dvalue;
        aMaterial.attributes["Dunit"] = Dunit;
        aMaterial.attributes["Dtype"] = Dtype;
      } else if (theNodeName == "composite") {
        auto const element = GetAttribute("ref", childAttributes);
        auto const count   = GetAttribute("n", childAttributes);
        if (element.size()) aMaterial.components["element"] = count;
      } else { // theNodeName == "fraction"
        auto const element  = GetAttribute("ref", childAttributes);
        auto const fraction = GetAttribute("n", childAttributes);
        if (element.size()) aMaterial.fractions[element] = fraction;
      }
    }
  }
  aMaterial.name     = materialName;
  auto const success = materialMap.insert(std::make_pair(materialName, aMaterial)).second;
  if (!success) {
    VECGEOM_LOG(warning) << "Middleware::processMaterial: failed to insert material with name " << materialName;
  }
  return success;
}

template <vecgeom::BooleanOperation Op>
vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *Middleware::processBoolean(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processBoolean: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *firstSolid  = nullptr;
  vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *secondSolid = nullptr;
  std::string name1st, name2nd;
  vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double> position;
  vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double> rotation;

  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    auto aDOMElement = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *>(it);
    if (!aDOMElement) {
      // Skip whitespace/text element
      if (debug) {
        VECGEOM_LOG(warning) << "Skipping null DOM element: " << Helper::GetNodeInformation(it);
      }
      continue;
    }
    auto const theChildNodeName = Helper::Transcode(it->getNodeName());
    if (debug) {
      VECGEOM_LOG(debug) << "procBoolean Child: " << Helper::GetNodeInformation(it);
    }
    if (theChildNodeName == "first") {
      auto const solidName = GetAttribute("ref", aDOMElement->getAttributes());

      auto foundSolid = unplacedVolumeMap.find(solidName);
      if (foundSolid == unplacedVolumeMap.end()) {
        VECGEOM_LOG(error) << "Could not find solid " << solidName;
        return nullptr;
      } else {
        if (debug) VECGEOM_LOG(debug) << "Found solid " << solidName;
        firstSolid = foundSolid->second;
        name1st    = solidName;
      }
    } else if (theChildNodeName == "second") {
      auto const solidName = GetAttribute("ref", aDOMElement->getAttributes());
      auto foundSolid      = unplacedVolumeMap.find(solidName);
      if (foundSolid == unplacedVolumeMap.end()) {
        VECGEOM_LOG(error) << "Could not find solid " << solidName;
        return nullptr;
      } else {
        if (debug) VECGEOM_LOG(debug) << "Found solid " << solidName;
        secondSolid = foundSolid->second;
        name2nd     = solidName;
      }
    } else if (theChildNodeName == "positionref") {
      auto const positionName = GetAttribute("ref", aDOMElement->getAttributes());
      position                = positionMap[positionName];
    } else if (theChildNodeName == "rotationref") {
      auto const rotationName = GetAttribute("ref", aDOMElement->getAttributes());
      rotation                = rotationMap[rotationName];
    } else if (theChildNodeName == "position") {
      auto attributes = aDOMElement->getAttributes();
      // auto const positionName = GetAttribute("name", attributes);
      auto const lengthMultiplier = GetLengthMultiplier(aDOMElement);
      DECLAREANDGETLENGTVAR(x)
      DECLAREANDGETLENGTVAR(y)
      DECLAREANDGETLENGTVAR(z)
      position = vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>{x, y, z};
    } else if (theChildNodeName == "rotation") {
      auto const *const attributes = aDOMElement->getAttributes();
      // auto const rotationName      = GetAttribute("name", attributes);
      auto const angleMultiplier = GetAngleMultiplier(aDOMElement);
      DECLAREANDGETANGLEVAR(x)
      DECLAREANDGETANGLEVAR(y)
      DECLAREANDGETANGLEVAR(z)
      rotation = vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>{x, y, z};
    }
  }
  if (!secondSolid || !firstSolid) {
    VECGEOM_LOG(error) << "Middleware::processBoolean: one of the requested soilds not found";
    return nullptr;
  }
  auto const rotxyz         = makeRotationMatrixFromCartesianAngles(rotation.x(), rotation.y(), rotation.z());
  auto const transformation = vecgeom::Transformation3D(position.x(), position.y(), position.z(), rotxyz.Inverse());

  auto const logicFirstVolume  = new vecgeom::VECGEOM_IMPL_NAMESPACE::LogicalVolume(name1st.c_str(), firstSolid);
  auto const logicSecondVolume = new vecgeom::VECGEOM_IMPL_NAMESPACE::LogicalVolume(name2nd.c_str(), secondSolid);
  auto *placedFirstSolidPtr    = logicFirstVolume->Place();
  auto *placedSecondSolidPtr   = logicSecondVolume->Place(&transformation);

  auto *booleanPtr = vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedBooleanVolume<Op>>(
      Op, placedFirstSolidPtr, placedSecondSolidPtr);
  return booleanPtr;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *Middleware::processMultiUnion(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processMultiUnion: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  std::vector<vecgeom::VECGEOM_IMPL_NAMESPACE::VPlacedVolume const *> placedNodes;

  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    if (debug) {
      VECGEOM_LOG(debug) << "procMultiUnion Child: " << Helper::GetNodeInformation(it);
    }
    if (it->getNodeType() == XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::ELEMENT_NODE) {
      placedNodes.emplace_back(processMultiUnionNode(it));
    }
  }
  auto *multiUnionPtr = vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedMultiUnion>();

  for (auto const *const node : placedNodes) {
    multiUnionPtr->AddNode(node);
  }
  return multiUnionPtr;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VPlacedVolume const *Middleware::processMultiUnionNode(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processMultiUnionNode: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *solid = nullptr;
  vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double> position;
  vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double> rotation;
  auto const name = Helper::GetAttribute("name", aDOMNode->getAttributes());
  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    auto aDOMElement = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *>(it);
    if (!aDOMElement) {
      // Skip whitespace/text element
      if (debug) {
        VECGEOM_LOG(warning) << "Skipping null DOM element: " << Helper::GetNodeInformation(it);
      }
      continue;
    }
    auto const theChildNodeName = Helper::Transcode(it->getNodeName());
    if (debug) {
      VECGEOM_LOG(debug) << "procMultiUnNode Child: " << Helper::GetNodeInformation(it);
    }
    if (theChildNodeName == "solid") {
      auto const solidName = GetAttribute("ref", aDOMElement->getAttributes());

      auto foundSolid = unplacedVolumeMap.find(solidName);
      if (foundSolid == unplacedVolumeMap.end()) {
        VECGEOM_LOG(error) << "Could not find solid " << solidName;
        return nullptr;
      } else {
        if (debug) VECGEOM_LOG(debug) << "Found solid " << solidName;
        solid = foundSolid->second;
      }
    } else if (theChildNodeName == "positionref") {
      auto const positionName = GetAttribute("ref", aDOMElement->getAttributes());
      position                = positionMap[positionName];
    } else if (theChildNodeName == "rotationref") {
      auto const rotationName = GetAttribute("ref", aDOMElement->getAttributes());
      rotation                = rotationMap[rotationName];
    }
  }
  if (!solid) {
    VECGEOM_LOG(error) << "Middleware::processUnion: one of the requested soilds not found";
    return nullptr;
  }
  auto const rotxyz         = makeRotationMatrixFromCartesianAngles(rotation.x(), rotation.y(), rotation.z());
  auto const transformation = vecgeom::Transformation3D(position.x(), position.y(), position.z(), rotxyz);
  // vecgeom::Transformation3D transformation(position.x(), position.y(), position.z(), rotation);
  auto const logicVolume = new vecgeom::VECGEOM_IMPL_NAMESPACE::LogicalVolume(name.c_str(), solid);
  auto *placedSolidPtr   = logicVolume->Place(name.c_str(), &transformation);
  return placedSolidPtr;
}

bool Middleware::processSolid(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processSolid: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const name              = Helper::Transcode(aDOMNode->getNodeName());
  auto const *const attributes = aDOMNode->getAttributes();
  auto const solidName         = GetAttribute("name", attributes);

  auto const *const anUnplacedSolid = [this, name, aDOMNode]() {
    if (name == "orb") {
      return processOrb(aDOMNode);
    } else if (name == "box") {
      return processBox(aDOMNode);
    } else if (name == "tube") {
      return processTube(aDOMNode);
    } else if (name == "eltube") {
      return processElTube(aDOMNode);
    } else if (name == "cutTube") {
      return processCutTube(aDOMNode);
    } else if (name == "cone") {
      return processCone(aDOMNode);
    } else if (name == "elcone") {
      return processElCone(aDOMNode);
    } else if (name == "polycone") {
      return processPolycone(aDOMNode);
    } else if (name == "genericPolycone") {
      return processGenPolycone(aDOMNode);
    } else if (name == "polyhedra") {
      return processPolyhedron(aDOMNode);
    } else if (name == "torus") {
      return processTorus(aDOMNode);
    } else if (name == "sphere") {
      return processSphere(aDOMNode);
    } else if (name == "ellipsoid") {
      return processEllipsoid(aDOMNode);
    } else if (name == "para") {
      return processParallelepiped(aDOMNode);
    } else if (name == "trd") {
      return processTrd(aDOMNode);
    } else if (name == "trap") {
      return processTrapezoid(aDOMNode);
    } else if (name == "arb8") {
      return processGenTrap(aDOMNode);
    } else if (name == "paraboloid") {
      return processParaboloid(aDOMNode);
    } else if (name == "intersection") {
      return processBoolean<vecgeom::BooleanOperation::kIntersection>(aDOMNode);
    } else if (name == "subtraction") {
      return processBoolean<vecgeom::BooleanOperation::kSubtraction>(aDOMNode);
    } else if (name == "union") {
      return processBoolean<vecgeom::BooleanOperation::kUnion>(aDOMNode);
    } else if (name == "multiUnion") {
      return processMultiUnion(aDOMNode);
    } else if (name == "hype") {
      return processHype(aDOMNode);
    } else if (name == "tessellated") {
      return processTesselated(aDOMNode);
    } else if (name == "tet") {
      return processTet(aDOMNode);
    } else if (name == "xtru") {
      return processExtruded(aDOMNode);
    } else if (name == "scaledSolid") {
      return processScaledShape(aDOMNode);
    } else
      return static_cast<vecgeom::VUnplacedVolume const *>(nullptr); // TODO more volumes
  }();
  if (!anUnplacedSolid) {
    VECGEOM_LOG(error) << "Middleware::processNode: an unknown solid " << name << " with name " << solidName;
    return false;
  }
  auto const success = unplacedVolumeMap.insert(std::make_pair(solidName, anUnplacedSolid)).second;
  if (!success) {
    VECGEOM_LOG(warning) << "Middleware::processNode: failed to insert volume with name " << solidName;
  }
  return success;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *Middleware::processOrb(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processOrb: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(r)
  auto const anUnplacedOrbPtr = vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedOrb>(r);
  return anUnplacedOrbPtr;
  // TODO precision
}

double Middleware::GetLengthMultiplier(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  const double mm              = vecgeom::GeoManager::GetMillimeterUnit();
  auto const *const attributes = aDOMNode->getAttributes();
  auto const nodeName          = Helper::Transcode(aDOMNode->getNodeName());
  auto const unitTag           = (nodeName == "position") ? "unit" : (nodeName == "auxiliary") ? "auxunit" : "lunit";
  auto const unit              = GetAttribute(unitTag, attributes);
  // if no unit attribute is specified, the default is mm
  auto const lengthMultiplier = (unit == "mm")   ? mm
                                : (unit == "m")  ? 1e3 * mm
                                : (unit == "km") ? 1e6 * mm
                                : (unit == "um") ? 1e-3 * mm
                                : (unit == "nm") ? 1e-6 * mm
                                : (unit == "cm") ? 10 * mm
                                                 : mm; // TODO more units
  return lengthMultiplier;
}

double Middleware::GetAngleMultiplier(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  auto const *const attributes = aDOMNode->getAttributes();
  auto const nodeName          = Helper::Transcode(aDOMNode->getNodeName());
  auto const unitTag           = (nodeName == "rotation") ? "unit" : "aunit";
  auto const unit              = GetAttribute(unitTag, attributes);
  auto const angleMultiplier   = (unit == "deg") ? vecgeom::VECGEOM_IMPL_NAMESPACE::kPi / 180. : 1.;
  return angleMultiplier;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *Middleware::processBox(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processBox: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(x)
  DECLAREANDGETLENGTVAR(y)
  DECLAREANDGETLENGTVAR(z)
  DECLAREHALF(x)
  DECLAREHALF(y)
  DECLAREHALF(z)
  auto const anUnplacedBoxPtr =
      vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedBox>(halfx, halfy, halfz);
  return anUnplacedBoxPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processTube(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processTube: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(z)
  DECLAREANDGETLENGTVAR(rmin)
  DECLAREANDGETLENGTVAR(rmax)
  DECLAREANDGETANGLEVAR(startphi)
  DECLAREANDGETANGLEVAR(deltaphi) // FIXME the default value is not 0
  DECLAREHALF(z)
  auto const anUnplacedTubePtr = vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTube>(
      rmin, rmax, halfz, startphi, deltaphi);
  return anUnplacedTubePtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processElTube(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processElTube: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(dx)
  DECLAREANDGETLENGTVAR(dy)
  DECLAREANDGETLENGTVAR(dz)
  auto const anUnplacedEllipticalTubePtr =
      vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedEllipticalTube>(dx, dy, dz);
  return anUnplacedEllipticalTubePtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processCutTube(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processCutTube: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(z)
  DECLAREANDGETLENGTVAR(rmin)
  DECLAREANDGETLENGTVAR(rmax)
  DECLAREANDGETANGLEVAR(startphi)
  DECLAREANDGETANGLEVAR(deltaphi) // FIXME the default value is not 0
  DECLAREANDGETLENGTVAR(lowX)
  DECLAREANDGETLENGTVAR(lowY)
  DECLAREANDGETLENGTVAR(lowZ)
  DECLAREANDGETLENGTVAR(highX)
  DECLAREANDGETLENGTVAR(highY)
  DECLAREANDGETLENGTVAR(highZ)
  DECLAREHALF(z)
  if (highZ < 0 || lowZ > 0) {
    VECGEOM_LOG(warning)
        << "Middleware::processCutTube: for compatibility, the normal must point outwards, expected to fail";
  }
  auto const bottomNormal         = vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>{lowX, lowY, lowZ};
  auto const topNormal            = vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>{highX, highY, highZ};
  auto const anUnplacedCutTubePtr = vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedCutTube>(
      rmin, rmax, halfz, startphi, deltaphi, bottomNormal, topNormal);
  return anUnplacedCutTubePtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processCone(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processCone: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(z)
  DECLAREANDGETLENGTVAR(rmin1)
  DECLAREANDGETLENGTVAR(rmin2)
  DECLAREANDGETLENGTVAR(rmax1)
  DECLAREANDGETLENGTVAR(rmax2)
  DECLAREANDGETANGLEVAR(startphi)
  DECLAREANDGETANGLEVAR(deltaphi) // FIXME the default value is not 0
  DECLAREHALF(z)
  auto const anUnplacedConePtr = vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedCone>(
      rmin1, rmax1, rmin2, rmax2, halfz, startphi, deltaphi);
  return anUnplacedConePtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processElCone(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processElCone: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETPLAINVAR(dx)
  DECLAREANDGETPLAINVAR(dy)
  DECLAREANDGETLENGTVAR(zmax)
  DECLAREANDGETLENGTVAR(zcut)
  auto const anUnplacedEllipticalConePtr =
      vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedEllipticalCone>(dx, dy, zmax, zcut);
  return anUnplacedEllipticalConePtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processPolycone(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  using vecgeom::Precision;
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processPolycone: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *attributes      = aDOMNode->getAttributes();
  auto const lengthMultiplier = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier  = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETANGLEVAR(startphi)
  DECLAREANDGETANGLEVAR(deltaphi) // FIXME the default value is not 0
  std::vector<Precision> rmins;
  std::vector<Precision> rmaxs;
  std::vector<Precision> zs;
  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    if (debug) {
      VECGEOM_LOG(debug) << "procPolycone Child: " << Helper::GetNodeInformation(it);
    }
    if (it->getNodeType() == XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::ELEMENT_NODE) {
      attributes = it->getAttributes();
      DECLAREANDGETLENGTVAR(rmax)
      DECLAREANDGETLENGTVAR(rmin)
      DECLAREANDGETLENGTVAR(z)
      rmins.push_back(rmin);
      rmaxs.push_back(rmax);
      zs.push_back(z);
    }
  }
  auto const anUnplacedPolyconePtr =
      vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedPolycone>(
          startphi, deltaphi, zs.size(), zs.data(), rmins.data(), rmaxs.data());
  return anUnplacedPolyconePtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processGenPolycone(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  using vecgeom::Precision;
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processGenPolycone: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *attributes      = aDOMNode->getAttributes();
  auto const lengthMultiplier = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier  = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETANGLEVAR(startphi)
  DECLAREANDGETANGLEVAR(deltaphi) // FIXME the default value is not 0
  std::vector<Precision> rvec;
  std::vector<Precision> zvec;
  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    if (debug) {
      VECGEOM_LOG(debug) << "procPolycone Child: " << Helper::GetNodeInformation(it);
    }
    if (it->getNodeType() == XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::ELEMENT_NODE) {
      attributes = it->getAttributes();
      DECLAREANDGETLENGTVAR(r)
      DECLAREANDGETLENGTVAR(z)
      rvec.push_back(r);
      zvec.push_back(z);
    }
  }
  auto const anUnplacedGenericPolyconePtr =
      vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedGenericPolycone>(
          startphi, deltaphi, rvec.size(), rvec.data(), zvec.data());
  return anUnplacedGenericPolyconePtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processPolyhedron(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  using vecgeom::Precision;
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processPolyhedron: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *attributes      = aDOMNode->getAttributes();
  auto const lengthMultiplier = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier  = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETANGLEVAR(startphi)
  DECLAREANDGETANGLEVAR(deltaphi) // FIXME the default value is not 0
  DECLAREANDGETINTVAR(numsides)
  std::vector<Precision> rmins;
  std::vector<Precision> rmaxs;
  std::vector<Precision> zs;
  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    if (debug) {
      VECGEOM_LOG(debug) << "procPolyhedr Child: " << Helper::GetNodeInformation(it);
    }
    if (it->getNodeType() == XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::ELEMENT_NODE) {
      attributes = it->getAttributes();
      DECLAREANDGETLENGTVAR(rmax)
      DECLAREANDGETLENGTVAR(rmin)
      DECLAREANDGETLENGTVAR(z)
      rmins.push_back(rmin);
      rmaxs.push_back(rmax);
      zs.push_back(z);
    }
  }
  auto const anUnplacedConePtr = vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedPolyhedron>(
      startphi, deltaphi, numsides, zs.size(), zs.data(), rmins.data(), rmaxs.data());
  return anUnplacedConePtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processTorus(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processTorus: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(rmin)
  DECLAREANDGETLENGTVAR(rmax)
  DECLAREANDGETLENGTVAR(rtor)
  DECLAREANDGETANGLEVAR(startphi)
  DECLAREANDGETANGLEVAR(deltaphi) // FIXME the default value is not 0
  auto const anUnplacedTorusPtr = vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTorus2>(
      rmin, rmax, rtor, startphi, deltaphi);
  return anUnplacedTorusPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processSphere(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processSphere: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(rmin)
  DECLAREANDGETLENGTVAR(rmax)
  DECLAREANDGETANGLEVAR(startphi)
  DECLAREANDGETANGLEVAR(deltaphi) // FIXME the default value is not 0
  DECLAREANDGETANGLEVAR(starttheta)
  DECLAREANDGETANGLEVAR(deltatheta) // FIXME the default value is not 0
  auto const anUnplacedSpherePtr = vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedSphere>(
      rmin, rmax, startphi, deltaphi, starttheta, deltatheta);
  return anUnplacedSpherePtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processEllipsoid(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processEllipsoid: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(ax)
  DECLAREANDGETLENGTVAR(by)
  DECLAREANDGETLENGTVAR(cz)
  DECLAREANDGETLENGTVAR(zcut1)
  DECLAREANDGETLENGTVAR(zcut2)
  auto const anUnplacedEllipsoidPtr =
      vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedEllipsoid>(ax, by, cz, zcut1, zcut2);
  return anUnplacedEllipsoidPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processParallelepiped(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processParallelepiped: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(x)
  DECLAREANDGETLENGTVAR(y)
  DECLAREANDGETLENGTVAR(z)
  DECLAREANDGETANGLEVAR(alpha)
  DECLAREANDGETANGLEVAR(theta)
  DECLAREANDGETANGLEVAR(phi)
  DECLAREHALF(x)
  DECLAREHALF(y)
  DECLAREHALF(z)
  auto const anUnplacedParallelepipedPtr =
      vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedParallelepiped>(halfx, halfy, halfz,
                                                                                                 alpha, theta, phi);
  return anUnplacedParallelepipedPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processTrd(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processTrd: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(x1)
  DECLAREANDGETLENGTVAR(x2)
  DECLAREANDGETLENGTVAR(y1)
  DECLAREANDGETLENGTVAR(y2)
  DECLAREANDGETLENGTVAR(z)
  DECLAREHALF(x1)
  DECLAREHALF(x2)
  DECLAREHALF(y1)
  DECLAREHALF(y2)
  DECLAREHALF(z)
  auto const anUnplacedTrdPtr = vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTrd>(
      halfx1, halfx2, halfy1, halfy2, halfz);
  return anUnplacedTrdPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processTrapezoid(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processTrapezoid: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(z)
  DECLAREANDGETLENGTVAR(y1)
  DECLAREANDGETLENGTVAR(x1)
  DECLAREANDGETLENGTVAR(x2)
  DECLAREANDGETLENGTVAR(y2)
  DECLAREANDGETLENGTVAR(x3)
  DECLAREANDGETLENGTVAR(x4)
  DECLAREANDGETANGLEVAR(theta)
  DECLAREANDGETANGLEVAR(phi)
  DECLAREANDGETANGLEVAR(alpha1)
  DECLAREANDGETANGLEVAR(alpha2)
  DECLAREHALF(z)
  DECLAREHALF(x1)
  DECLAREHALF(y1)
  DECLAREHALF(x2)
  DECLAREHALF(y2)
  DECLAREHALF(x3)
  DECLAREHALF(x4)
  auto const anUnplacedTrapezoidPtr =
      vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTrapezoid>(
          halfz, theta, phi, halfy1, halfx1, halfx2, alpha1, halfy2, halfx3, halfx4, alpha2);
  return anUnplacedTrapezoidPtr;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *Middleware::processGenTrap(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  using vecgeom::Precision;
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processGenTrap: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(v1x)
  DECLAREANDGETLENGTVAR(v1y)
  DECLAREANDGETLENGTVAR(v2x)
  DECLAREANDGETLENGTVAR(v2y)
  DECLAREANDGETLENGTVAR(v3x)
  DECLAREANDGETLENGTVAR(v3y)
  DECLAREANDGETLENGTVAR(v4x)
  DECLAREANDGETLENGTVAR(v4y)
  DECLAREANDGETLENGTVAR(v5x)
  DECLAREANDGETLENGTVAR(v5y)
  DECLAREANDGETLENGTVAR(v6x)
  DECLAREANDGETLENGTVAR(v6y)
  DECLAREANDGETLENGTVAR(v7x)
  DECLAREANDGETLENGTVAR(v7y)
  DECLAREANDGETLENGTVAR(v8x)
  DECLAREANDGETLENGTVAR(v8y)
  DECLAREANDGETLENGTVAR(dz)
  std::vector<Precision> verticesx;
  std::vector<Precision> verticesy;
  verticesx.push_back(v1x);
  verticesx.push_back(v2x);
  verticesx.push_back(v3x);
  verticesx.push_back(v4x);
  verticesx.push_back(v5x);
  verticesx.push_back(v6x);
  verticesx.push_back(v7x);
  verticesx.push_back(v8x);
  verticesy.push_back(v1y);
  verticesy.push_back(v2y);
  verticesy.push_back(v3y);
  verticesy.push_back(v4y);
  verticesy.push_back(v5y);
  verticesy.push_back(v6y);
  verticesy.push_back(v7y);
  verticesy.push_back(v8y);
  auto const anUnplacedGenTrapPtr = vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedGenTrap>(
      verticesx.data(), verticesy.data(), dz);
  return anUnplacedGenTrapPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processParaboloid(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processParaboloid: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(rlo)
  DECLAREANDGETLENGTVAR(rhi)
  DECLAREANDGETLENGTVAR(dz)
  auto const anUnplacedParaboloidPtr =
      vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedParaboloid>(rlo, rhi, dz);
  return anUnplacedParaboloidPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processHype(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processHype: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  auto const lengthMultiplier  = GetLengthMultiplier(aDOMNode);
  auto const angleMultiplier   = GetAngleMultiplier(aDOMNode);
  DECLAREANDGETLENGTVAR(rmin)
  DECLAREANDGETLENGTVAR(rmax)
  DECLAREANDGETANGLEVAR(inst)
  DECLAREANDGETANGLEVAR(outst)
  DECLAREANDGETLENGTVAR(z)
  DECLAREHALF(z)
  auto const anUnplacedHypePtr =
      vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedHype>(rmin, rmax, inst, outst, halfz);
  return anUnplacedHypePtr;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *Middleware::processTesselated(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processTesselated: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto *const anUnplacedTessellatedPtr =
      vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTessellated>();

  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    if (debug) {
      VECGEOM_LOG(debug) << "procTessel Child: " << Helper::GetNodeInformation(it);
    }
    if (it->getNodeType() == XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::ELEMENT_NODE) {
      processFacet(it, *anUnplacedTessellatedPtr);
    }
  }
  return anUnplacedTessellatedPtr;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *Middleware::processTet(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processTesselated: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();
  std::array<vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>, 4> vertices;
  for (auto const ind : {0u, 1u, 2u, 3u}) {
    auto const positionName = GetAttribute("vertex" + std::to_string(ind + 1), attributes);
    auto const position     = positionMap[positionName];
    vertices.at(ind)        = position;
  }
  auto const anUnplacedTetPtr = vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTet>(
      vertices.at(0), vertices.at(1), vertices.at(2), vertices.at(3));
  return anUnplacedTetPtr;
}

vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *Middleware::processExtruded(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  using vecgeom::Precision;
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processExtruded: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *attributes      = aDOMNode->getAttributes();
  auto const lengthMultiplier = GetLengthMultiplier(aDOMNode);
  std::vector<Precision> xs;
  std::vector<Precision> ys;
  std::vector<Precision> zs; // only first two are used, scaling factor and offset are not supported
  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    if (debug) {
      VECGEOM_LOG(debug) << "procExtru Child: " << Helper::GetNodeInformation(it);
    }
    if (it->getNodeType() == XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::ELEMENT_NODE) {
      auto const theChildNodeName = Helper::Transcode(it->getNodeName());
      attributes                  = it->getAttributes();
      if (theChildNodeName == "twoDimVertex") {
        DECLAREANDGETLENGTVAR(x);
        DECLAREANDGETLENGTVAR(y);
        xs.push_back(x);
        ys.push_back(y);
      } else if (theChildNodeName == "section") {
        DECLAREANDGETPLAINVAR(scalingFactor)
        DECLAREANDGETLENGTVAR(xOffset)
        DECLAREANDGETLENGTVAR(yOffset)
        DECLAREANDGETLENGTVAR(zPosition)
        zs.push_back(zPosition);
        DECLAREANDGETINTVAR(zOrder)
        if (!(zOrder == 0 || zOrder == 1)) {
          VECGEOM_LOG(warning) << "Ignoring unsupported xtru section attribute zOrder=" << zOrder;
        }
        if (scalingFactor != 1.0) {
          VECGEOM_LOG(warning) << "Ignoring unsupported xtru section attribute scalingFactor=" << scalingFactor;
        }
        if (xOffset != 0.0) {
          VECGEOM_LOG(warning) << "Ignoring unsupported xtru section attribute xOffset=" << xOffset;
        }
        if (yOffset != 0.0) {
          VECGEOM_LOG(warning) << "Ignoring unsupported xtru section attribute yOffset=" << yOffset;
        }
        auto const z = lengthMultiplier * GetDoubleAttribute("zPosition", attributes);
        zs.push_back(z);
      }
    }
  }
  auto const anUnplacedExtrudedPtr =
      vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedSExtruVolume>(
          xs.size(), xs.data(), ys.data(), zs.front(), zs.back());
  return anUnplacedExtrudedPtr;
}

const vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume *Middleware::processScaledShape(
    XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processScaledShape: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  vecgeom::VECGEOM_IMPL_NAMESPACE::VUnplacedVolume const *solid = nullptr;
  vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double> scale(1., 1., 1.);
  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    if (debug) {
      VECGEOM_LOG(debug) << "procScaledShape Child: " << Helper::GetNodeInformation(it);
    }
    if (it->getNodeType() == XERCES_CPP_NAMESPACE_QUALIFIER DOMNode::ELEMENT_NODE) {
      auto const theChildNodeName = Helper::Transcode(it->getNodeName());
      if (theChildNodeName == "solidref") {
        auto const solidName = GetAttribute("ref", it->getAttributes());
        auto foundSolid      = unplacedVolumeMap.find(solidName);
        if (foundSolid == unplacedVolumeMap.end()) {
          VECGEOM_LOG(error) << "Could not find solid " << solidName;
          return nullptr;
        } else {
          if (debug) VECGEOM_LOG(debug) << "Found solid " << solidName;
          solid = foundSolid->second;
        }
      } else if (theChildNodeName == "scale") {
        processScale(it);
        auto const scaleName = GetAttribute("name", it->getAttributes());
        scale                = scaleMap[scaleName];
      } else if (theChildNodeName == "scaleref") {
        auto const scaleName = GetAttribute("ref", it->getAttributes());
        scale                = scaleMap[scaleName];
      }
    }
  }
  auto const anUnplacedScaledPtr =
      vecgeom::GeoManager::MakeInstance<vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedScaledShape>(solid, scale.x(),
                                                                                              scale.y(), scale.z());
  return anUnplacedScaledPtr;
}

bool Middleware::processFacet(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode,
                              vecgeom::VECGEOM_IMPL_NAMESPACE::UnplacedTessellated &storage)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processFacet: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const *const attributes = aDOMNode->getAttributes();

  enum FacetType { fTriangular, fQuadrangular };
  auto const facetName  = Helper::Transcode(aDOMNode->getNodeName());
  auto const facetType  = (facetName == "triangular") ? fTriangular : fQuadrangular;
  auto const vertexType = GetAttribute("type", attributes);
  auto const absolute   = (vertexType == "ABSOLUTE");
  if (facetType == fTriangular) {
    std::array<vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>, 3> vertices;
    for (auto const ind : {0u, 1u, 2u}) {
      auto const positionName = GetAttribute("vertex" + std::to_string(ind + 1), attributes);
      auto const position     = positionMap[positionName];
      vertices.at(ind)        = position;
    }
    storage.AddTriangularFacet(vertices.at(0), vertices.at(1), vertices.at(2), absolute);
  } else {
    std::array<vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double>, 4> vertices;
    for (auto const ind : {0u, 1u, 2u, 3u}) {
      auto const positionName = GetAttribute("vertex" + std::to_string(ind + 1), attributes);
      auto const position     = positionMap[positionName];
      vertices.at(ind)        = position;
    }
    storage.AddQuadrilateralFacet(vertices.at(0), vertices.at(1), vertices.at(2), vertices.at(3), absolute);
  }
  return true;
}

bool Middleware::processLogicVolume(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processLogicVolume: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  vecgeom::VECGEOM_IMPL_NAMESPACE::LogicalVolume *logicVolume = nullptr;
  vgdml::Material const *foundMaterial                        = nullptr;
  auto const *const attributes                                = aDOMNode->getAttributes();
  auto const volumeName                                       = GetAttribute("name", attributes);

  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    auto aDOMElement = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *>(it);
    if (!aDOMElement) {
      // Skip whitespace/text element
      if (debug) {
        VECGEOM_LOG(warning) << "Skipping null DOM element: " << Helper::GetNodeInformation(it);
      }
      continue;
    }
    auto const theChildNodeName = Helper::Transcode(it->getNodeName());
    if (debug) {
      VECGEOM_LOG(debug) << "procLogVol Child: " << Helper::GetNodeInformation(it);
    }
    if (theChildNodeName == "solidref") {
      auto const solidName = GetAttribute("ref", aDOMElement->getAttributes());
      if (debug) VECGEOM_LOG(debug) << "volume " << volumeName << " references solid " << solidName;

      auto foundSolid = unplacedVolumeMap.find(solidName);
      if (foundSolid == unplacedVolumeMap.end()) {
        VECGEOM_LOG(error) << "processLogicVolume: Could not find solid reference " << solidName
                           << " for logical volume " << volumeName;
        return false;
      } else {
        if (debug) VECGEOM_LOG(debug) << "Found solid " << solidName;
        logicVolume = new vecgeom::VECGEOM_IMPL_NAMESPACE::LogicalVolume(volumeName.c_str(), foundSolid->second);
        vecgeom::GeoManager::Instance().RegisterLogicalVolume(logicVolume);
        if (foundMaterial) {
          auto const success = volumeMaterialMap.insert(std::make_pair(logicVolume->id(), *foundMaterial)).second;
          if (!success) {
            VECGEOM_LOG(error) << "processLogicVolume: Could not insert volid-material pair: " << logicVolume->id()
                               << " - " << foundMaterial->name;
            return false;
          }
        }
      }
    } else if (theChildNodeName == "materialref") {
      auto const materialName = GetAttribute("ref", aDOMElement->getAttributes());
      if (debug) VECGEOM_LOG(debug) << "volume " << volumeName << " references material " << materialName;

      auto pairMaterial = materialMap.find(materialName);
      if (pairMaterial == materialMap.end()) {
        VECGEOM_LOG(error) << "processLogicVolume: Could not find material reference " << materialName
                           << " for logical volume " << volumeName;
        return false;
      } else {
        if (debug) VECGEOM_LOG(debug) << "Found material " << materialName;
        foundMaterial = &pairMaterial->second;
        if (logicVolume) {
          auto const success = volumeMaterialMap.insert(std::make_pair(logicVolume->id(), *foundMaterial)).second;
          if (!success) {
            VECGEOM_LOG(error) << "processLogicVolume: Could not insert volid-material pair: " << logicVolume->id()
                               << " - " << foundMaterial->name;
            return false;
          }
        }
      }
    } else if (theChildNodeName == "physvol") {
      auto const success = processPhysicalVolume(aDOMElement, logicVolume);
      if (!success) return false;
    } else if (theChildNodeName == "auxiliary") {
      Auxiliary aux;
      auto const success = processAuxiliary(aDOMElement, aux);
      if (!success) {
        VECGEOM_LOG(error) << "processLogicVolume: Could not process auxiliary tag in volume: " << volumeName;
        return false;
      }
      // emplace aux into volumeAux map
      volumeAuxiliaryInfo[logicVolume->id()].emplace_back(std::move(aux));
    }
  }
  return logicVolume;
}

bool Middleware::processPhysicalVolume(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode,
                                       vecgeom::LogicalVolume *motherLogical)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processPhysicalVolume: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  vecgeom::VECGEOM_IMPL_NAMESPACE::LogicalVolume *logicalVolume = nullptr;
  vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double> position;
  vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double> rotation;
  vecgeom::VECGEOM_IMPL_NAMESPACE::Vector3D<double> scale(1, 1, 1);
  auto const *const attributes = aDOMNode->getAttributes();
  std::string PVname           = GetAttribute("name", attributes);
  DECLAREANDGETINTVAR(copynumber);

  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    auto aDOMElement = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *>(it);
    if (!aDOMElement) {
      // Skip whitespace/text element
      if (debug) {
        VECGEOM_LOG(warning) << "Skipping null DOM element: " << Helper::GetNodeInformation(it);
      }
      continue;
    }
    if (debug) {
      VECGEOM_LOG(debug) << "procPhysVol Child: " << Helper::GetNodeInformation(it);
    }
    auto const theChildNodeName = Helper::Transcode(it->getNodeName());
    if (theChildNodeName == "volumeref") {
      auto const logicalVolumeName = GetAttribute("ref", aDOMElement->getAttributes());
      logicalVolume                = vecgeom::GeoManager::Instance().FindLogicalVolume(logicalVolumeName.c_str());
      if (!logicalVolume) {
        VECGEOM_LOG(error) << "Middleware::processPhysicalVolume: could not find volume " << logicalVolumeName;
        return false;
      } else {
        if (debug) VECGEOM_LOG(debug) << "Middleware::processPhysicalVolume: found volume " << logicalVolumeName;
      }
    } else if (theChildNodeName == "position") {
      auto const positionName = GetAttribute("name", aDOMElement->getAttributes());
      if (processPosition(aDOMElement)) {
        position = positionMap[positionName];
      } else {
        VECGEOM_LOG(error) << "Middleware::processPhysicalVolume: could not process position " << positionName;
        return false;
      }
    } else if (theChildNodeName == "rotation") {
      auto const rotationName = GetAttribute("name", aDOMElement->getAttributes());
      if (processRotation(aDOMElement)) {
        rotation = rotationMap[rotationName];
      } else {
        VECGEOM_LOG(error) << "Middleware::processPhysicalVolume: could not process rotation " << rotationName;
        return false;
      }
    } else if (theChildNodeName == "scale") {
      auto const scaleName = GetAttribute("name", aDOMElement->getAttributes());
      if (processScale(aDOMElement)) {
        scale = scaleMap[scaleName];
      } else {
        VECGEOM_LOG(error) << "Middleware::processPhysicalVolume: could not process scale " << scaleName;
        return false;
      }
    } else if (theChildNodeName == "scaleref") {
      auto const scaleName = GetAttribute("ref", aDOMElement->getAttributes());
      scale                = scaleMap[scaleName];
    } else if (theChildNodeName == "positionref") {
      auto const positionName = GetAttribute("ref", aDOMElement->getAttributes());
      position                = positionMap[positionName];
    } else if (theChildNodeName == "rotationref") {
      auto const rotationName = GetAttribute("ref", aDOMElement->getAttributes());
      rotation                = rotationMap[rotationName];
    } else {
      if (debug) {
        VECGEOM_LOG(error) << "Middleware::processPhysicalVolume: tag not understood: " << theChildNodeName;
        return false;
      }
    }
  }
  if (!logicalVolume) return false;
  auto const rotxyz = makeRotationMatrixFromCartesianAngles(rotation.x(), rotation.y(), rotation.z());
  vecgeom::Transformation3D transformation(position.x(), position.y(), position.z(), rotxyz);

  // make sure PVname is not empty
  if (PVname == "") {
    PVname.append(logicalVolume->GetName());
    PVname.append("_PV");
  }
  bool success = ReflFactory::Instance().Place(transformation, scale, PVname, logicalVolume, motherLogical, copynumber);
  return success;
}

bool Middleware::processUserInfo(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    auto *aDOMElement = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *>(it);
    if (aDOMElement == nullptr) {
      // Skip whitespace/text element
      continue;
    }

    const std::string tag = Helper::Transcode(aDOMElement->getTagName());
    if (tag != "auxiliary") {
      return false;
    }

    Auxiliary tmp;
    if (!processAuxiliary(aDOMElement, tmp)) {
      return false;
    }

    userInfo.emplace_back(std::move(tmp));
  }

  return true;
}

bool Middleware::processAuxiliary(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode, Auxiliary &aux)
{
  aux = Auxiliary();
  aux.SetType(GetAttribute("auxtype", aDOMNode->getAttributes()));
  aux.SetValue(GetAttribute("auxvalue", aDOMNode->getAttributes()));
  aux.SetUnit(GetAttribute("auxunit", aDOMNode->getAttributes()));

  for (auto *it = aDOMNode->getFirstChild(); it != nullptr; it = it->getNextSibling()) {
    auto *aDOMElement = dynamic_cast<XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *>(it);
    if (aDOMElement == nullptr) {
      // Skip whitespace/text element
      continue;
    }

    // Check that child is an auxiliary
    const std::string tag = Helper::Transcode(aDOMElement->getTagName());
    if (tag != "auxiliary") {
      return false;
    }

    Auxiliary tmp;
    if (!processAuxiliary(aDOMElement, tmp)) {
      return false;
    }

    aux.AddChild(tmp);
  }

  return true;
}

bool Middleware::processWorld(XERCES_CPP_NAMESPACE_QUALIFIER DOMNode const *aDOMNode)
{
  if (debug) {
    VECGEOM_LOG(debug) << "Middleware::processWorld: processing: " << Helper::GetNodeInformation(aDOMNode);
  }
  auto const logicalVolumeName = GetAttribute("ref", aDOMNode->getAttributes());
  auto logicalVolume           = vecgeom::GeoManager::Instance().FindLogicalVolume(logicalVolumeName.c_str());

  if (!logicalVolume) {
    VECGEOM_LOG(error) << "Middleware::processWorld: could not find world volume " << logicalVolumeName;
    return false;
  } else {
    if (debug) VECGEOM_LOG(debug) << "Middleware::processWorld: found world volume " << logicalVolumeName;
    std::string PVname(logicalVolumeName);
    PVname.append("_PV");
    auto placedWorld = logicalVolume->Place(PVname.c_str());           // TODO use the setup name
    vecgeom::GeoManager::Instance().RegisterPlacedVolume(placedWorld); // FIXME is it needed?
    vecgeom::GeoManager::Instance().SetWorldAndClose(placedWorld);
  }
  return true;
}

} // namespace vgdml
