#include "VUSolid.hh"
////////////////////////////////////////////////////////////////////////////////
//  "Universal" Solid Interface
//  Authors: J. Apostolakis, G. Cosmo, A. Gheata, A. Munnich, T. Nikitina (CERN)
//
//  Created: 25 May 2011
//
////////////////////////////////////////////////////////////////////////////////

//double VUSolid::fgTolerance = 1.0E-8;  // to be changed
double VUSolid::fgTolerance = 1.0E-5;  // to be changed

//______________________________________________________________________________
VUSolid::VUSolid() :
         fName(),
         fBBox(0)
{
// Dummy constructor
}

//______________________________________________________________________________
VUSolid::VUSolid(const char *name) :
         fName(name),
         fBBox(0)
{
// Named constructor
   SetName(name);
}

//______________________________________________________________________________
VUSolid::~VUSolid()
{
// Destructor
}



