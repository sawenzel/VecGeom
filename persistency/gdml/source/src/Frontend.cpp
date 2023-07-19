//!    \file Frontend.cpp
//!    \brief implements loading GDML to VecGeom
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#include "Frontend.h"
#include "Backend.h"
#include "VecGeom/management/GeoManager.h"

#include <string>

namespace vgdml {

namespace Frontend {
bool Load(std::string const &aFilename, bool validate, double mm_unit, bool verbose)
{
  Parser p;
  return p.Load(aFilename, validate, mm_unit, verbose) ? true : false;
}
} // namespace Frontend

std::unique_ptr<Middleware> Parser::Load(std::string const &aFilename, bool validate, double mm_unit, bool verbose)
{
  auto aBackend      = vgdml::Backend(validate);
  auto const aDOMDoc = aBackend.Load(aFilename);
  if (!aDOMDoc) {
    std::cerr << "== Error: GDML file " << aFilename << " could not be loaded\n";
    return nullptr;
  }

  vecgeom::GeoManager::SetMillimeterUnit((vecgeom::Precision)mm_unit);
  if (verbose == 1) std::cerr << "(II) vgdml::Frontend::Load: VecGeom millimeter is " << mm_unit << "\n";

  auto aMiddleware = std::unique_ptr<Middleware>(new Middleware());
  if (!aMiddleware->Load(aDOMDoc)) {
    return nullptr;
  }
  return aMiddleware;
}

} // namespace vgdml
