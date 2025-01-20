//!    \file Backend.cpp
//!    \brief Defines the class for loading files to DOM and writing back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#include "Backend.h"
#include "Helper.h"

// TODO leave only the needed
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMLSParser.hpp>
#include <xercesc/dom/DOMLSOutput.hpp>
#include <xercesc/framework/psvi/PSVIHandler.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <iostream>

#include <cstdlib>

namespace vgdml {

class ErrorHandler : public xercesc::ErrorHandler {
  bool fSuppress;

public:
  ErrorHandler(const bool set) { fSuppress = set; }

  void warning(const xercesc::SAXParseException &exception)
  {
    if (fSuppress) {
      return;
    }
    char *message = xercesc::XMLString::transcode(exception.getMessage());
    std::cerr << "VGDML: VALIDATION WARNING! " << message << " at line: " << exception.getLineNumber() << std::endl;
    xercesc::XMLString::release(&message);
  }

  void error(const xercesc::SAXParseException &exception)
  {
    if (fSuppress) {
      return;
    }
    char *message = xercesc::XMLString::transcode(exception.getMessage());
    std::cerr << "VGDML: VALIDATION ERROR! " << message << " at line: " << exception.getLineNumber() << std::endl;
    xercesc::XMLString::release(&message);
  }

  void fatalError(const xercesc::SAXParseException &exception) { error(exception); }
  void resetErrors() {}
};

class XercesInitializer {
public:
  static void Initialize() { static XercesInitializer instance; }

private:
  XercesInitializer() { xercesc::XMLPlatformUtils::Initialize(); }
  ~XercesInitializer() { xercesc::XMLPlatformUtils::Terminate(); }

  // Prevent copying and assignment
  XercesInitializer(const XercesInitializer &)            = delete;
  XercesInitializer &operator=(const XercesInitializer &) = delete;
};

Backend::Backend(bool validate)
{
  XercesInitializer::Initialize();

  fDOMParser           = std::make_unique<xercesc::XercesDOMParser>();
  auto const schemaDir = std::getenv("GDMLDIR"); // get the alternative schema location for offline use
  if (schemaDir) {
    auto const schemaFile = (schemaDir + std::string("gdml.xsd"));
    fDOMParser->setExternalNoNamespaceSchemaLocation(schemaFile.c_str());
    fDOMParser->loadGrammar(schemaFile.c_str(), xercesc::Grammar::SchemaGrammarType, true);
  }

  // Should be optional.
  if (validate)
    fDOMParser->setValidationScheme(xercesc::XercesDOMParser::Val_Always);
  else
    fDOMParser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

  fDOMParser->setDoNamespaces(true);
  fDOMParser->setDoSchema(validate);
  fDOMParser->setValidationSchemaFullChecking(validate);
  fDOMParser->setDoXInclude(true);
  fDOMParser->setCreateSchemaInfo(true);
  fDOMParser->setIncludeIgnorableWhitespace(false);

  xercesc::ErrorHandler *handler =
      new ErrorHandler(false); // Geant4 suppression errors when not validating (!validate);
  fDOMParser->setErrorHandler(handler);
}

Backend::~Backend()
{
  // Ensure Xerces-C++ parser is deleted before terminating the platform
  if (fDOMParser) {
    delete fDOMParser->getErrorHandler();
    fDOMParser.reset();
  }

  // Terminate Xerces last
  xercesc::XMLPlatformUtils::Terminate();
}

XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *Backend::Load(std::string const &aFilename)
{
  fDOMParser->resetDocumentPool();

  try {
    fDOMParser->parse(aFilename.c_str());
  } catch (const xercesc::XMLException &e) {
    std::cerr << "VGDML: " << Helper::Transcode(e.getMessage()) << std::endl;
  } catch (const xercesc::DOMException &e) {
    std::cerr << "VGDML: " << Helper::Transcode(e.getMessage()) << std::endl;
  }

  auto doc = fDOMParser->getDocument();
  return doc;
}

void Backend::Save(XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument const *aDOMDoc, std::string const &aFilename)
{
  // TODO set parameters, maybe take form Geant4
  XMLCh tempStr[3]                        = {xercesc::chLatin_L, xercesc::chLatin_S, xercesc::chNull};
  xercesc::DOMImplementation *impl        = xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
  xercesc::DOMLSSerializer *theSerializer = (static_cast<xercesc::DOMImplementationLS *>(impl))->createLSSerializer();
  xercesc::DOMLSOutput *theOutputDesc     = (static_cast<xercesc::DOMImplementationLS *>(impl))->createLSOutput();
  xercesc::XMLFormatTarget *myFormTarget  = new xercesc::LocalFileFormatTarget(aFilename.c_str());
  theOutputDesc->setByteStream(myFormTarget);
  theSerializer->write(aDOMDoc, theOutputDesc);

  theOutputDesc->release();
  theSerializer->release();
  return;
}
} // namespace vgdml
