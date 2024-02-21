//!    \file TestXercesBackend.cpp
//!    \brief reads a gdml file to DOM and writes it back
//!
//!    \authors Author:  Dmitry Savin <sd57@protonmail.ch>
//!

#include <iostream>
#include "Backend.h"
#include "xercesc/dom/DOMDocument.hpp"
#include "test/benchmark/ArgParser.h"

namespace {
static void usage()
{
  std::cout << "\nUsage:\n"
               "    TestXercesBackend <filename>.gdml \n"
               "      will use TestXercesBackend.gdml by default\n"
            << std::endl;
}
} // namespace

int main(int argc, char *argv[])
{
  if (argc != 2) {
    usage();
  }
  OPTION_BOOL(validate, false);

  auto const filename = std::string((argc > 1) ? argv[1] : "TestXercesBackend.gdml");
  auto aBackend       = vgdml::Backend(validate);
  auto const loaded   = aBackend.Load(filename);
  aBackend.Save(loaded, "TestXercesBackend.out.gdml");
  return 0;
}
