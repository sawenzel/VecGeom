//!    \file Backend.h
//!    \brief Declares class for loading files to DOM and writing back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#pragma once

#ifndef VGDMLBackend_h
#define VGDMLBackend_h

#include "xercesc/util/XercesDefs.hpp"

#include <string>
#include <memory>

XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class XercesDOMParser;
XERCES_CPP_NAMESPACE_END

namespace vgdml {
class Backend {
  std::unique_ptr<xercesc::XercesDOMParser> fDOMParser;

public:
  Backend(bool validate);
  ~Backend();
  XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *Load(std::string const &aFilename);
  void Save(XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument const *aDOMDoc, std::string const &aFilename);
};
} // namespace vgdml
#endif
