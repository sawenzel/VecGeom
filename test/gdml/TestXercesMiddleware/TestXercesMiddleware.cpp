//!    \file TestXercesMiddleware.cpp
//!    \brief reads a gdml file to DOM and writes it back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#include <iostream>
#include "Backend.h"
#include "Middleware.h"
#include "MaterialInfo.h"
#include "xercesc/dom/DOMDocument.hpp"
#include "VecGeom/management/GeoManager.h"

#ifdef VECGEOM_ROOT
// if ROOT is available will export the loaded geometry
#include "VecGeomTest/RootGeoManager.h"
#include "TGeoManager.h"
#endif

namespace {
static void usage()
{
  std::cout << "\nUsage:\n"
               "    TestXercesMiddleware <filename>.gdml \n"
               "      will use TestXercesMiddleware.gdml by default\n"
            << std::endl;
}
} // namespace

int main(int argC, char *argV[])
{
  if (argC != 2) {
    usage();
  }
  auto const filename = std::string((argC > 1) ? argV[1] : "TestXercesMiddleware.gdml");
  // To speed up testing and ensure the middleware and backend work without
  // schema validation, skip it for the middleware test.
  constexpr bool validate_xml_schema = false;
  auto aBackend                      = vgdml::Backend(validate_xml_schema);
  auto const aDOMDoc                 = aBackend.Load(filename);
  aBackend.Save(aDOMDoc, "TestXercesMiddleware.out.gdml");
  auto aMiddleware      = vgdml::Middleware();
  auto loadedMiddleware = aMiddleware.Load(aDOMDoc);
  //  std::cout << loadedMiddleware << std::endl;
  auto const *world = vecgeom::GeoManager::Instance().GetWorld();
  if (!loadedMiddleware || !world) return 1;
#ifdef VECGEOM_ROOT
  auto &aROOTmanager = vecgeom::RootGeoManager::Instance();
  //  aROOTmanager.EnableTGeoUnits(); // does not work at export stage
  aROOTmanager.ExportToROOTGeometry(world, "TestXercesMiddleware.out.RootGeo.gdml");
  TGeoManager::Import(filename.c_str());
  auto const *const topVolume = gGeoManager->GetTopVolume();
  if (!topVolume) {
    std::cout << "TestXercesMiddleware: ROOT failed to load geometry" << std::endl;
  } else {
    try {
      gGeoManager->Export("TestXercesMiddleware.out.TGeo.gdml");
    } catch (...) {
      std::cout << "Unexpected error when exporting by ROOT" << std::endl;
    }
  }
#endif

  return 0;
}
