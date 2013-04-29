//
// ********************************************************************
// * License and Disclaimer																					 *
// *																																	*
// * The	Geant4 software	is	copyright of the Copyright Holders	of *
// * the Geant4 Collaboration.	It is provided	under	the terms	and *
// * conditions of the Geant4 Software License,	included in the file *
// * LICENSE and available at	http://cern.ch/geant4/license .	These *
// * include a list of copyright holders.														 *
// *																																	*
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work	make	any representation or	warranty, express or implied, *
// * regarding	this	software system or assume any liability for its *
// * use.	Please see the license in the file	LICENSE	and URL above *
// * for the full disclaimer and the limitation of liability.				 *
// *																																	*
// * This	code	implementation is the result of	the	scientific and *
// * technical work of the GEANT4 collaboration.											*
// * By using,	copying,	modifying or	distributing the software (or *
// * any work based	on the software)	you	agree	to acknowledge its *
// * use	in	resulting	scientific	publications,	and indicate your *
// * acceptance of all terms of the Geant4 Software license.					*
// ********************************************************************
//
//
// $Id: UVCSGfaceted.hh 66241 2012-12-13 18:34:42Z gunter $
//
// 
// --------------------------------------------------------------------
// GEANT 4 class header file
//
//
// UVCSGfaceted
//
// Class description:
//
//	 Virtual class defining CSG-like type shape that is built entire
//	 of UCSGface faces.

// Author:
//	 David C. Williams (davidw@scipp.ucsc.edu)
// --------------------------------------------------------------------

#ifndef UPolycone2_hh
#define UPolycone2_hh

#include "VUSolid.hh"

#include "UMultiUnion.hh"
//#include "UPolyconeSide.hh"
#include "UPolycone.hh"

class UPolycone2 : public UMultiUnion
{

public:	// with description

  UPolycone2( const std::string& name, 
    double phiStart,		 // initial phi starting angle
    double phiTotal,		 // total phi angle
    int numZPlanes,			// number of z planes
    const double zPlane[],		 // position of z planes
    const double rInner[],		 // tangent distance to inner surface
    const double rOuter[]	);	// tangent distance to outer surface

  UPolycone2( const std::string& name, 
    double phiStart,		// initial phi starting angle
    double phiTotal,		// total phi angle
    int		numRZ,			 // number corners in r,z space
    const double r[],				 // r coordinate of these corners
    const double z[]			 ); // z coordinate of these corners

  virtual ~UPolycone2();

  void Create( double phiStart,				// initial phi starting angle
    double phiTotal,				// total phi angle
    UReduciblePolygon *rz ); // r/z coordinate of these corners


//  inline void SetOriginalParameters(UPolyconeHistorical* pars);

//  inline void SetOriginalParameters();

  std::ostream& StreamInfo( std::ostream& os ) const;

protected:	// without description

  // Here are our parameters

  double startPhi;		// Starting phi value (0 < phiStart < 2pi)
  double endPhi;			// end phi value (0 < endPhi-phiStart < 2pi)
  bool	 phiIsOpen;	 // true if there is a phi segment
  bool	 genericPcon; // true if created through the 2nd generic constructor
  int	 numCorner;		// number RZ points
  UPolyconeSideRZ *corners;	// corner r,z points
  UPolyconeHistorical	*original_parameters;	// original input parameters

  inline
    double GetStartPhi() const
  {
    return startPhi;
  }

  inline
    double GetEndPhi() const
  {
    return endPhi;
  }

  inline
    bool IsOpen() const
  {
    return phiIsOpen;
  }

  inline
    bool IsGeneric() const
  {
    return genericPcon;
  }

  inline
    int GetNumRZCorner() const
  {
    return numCorner;
  }

  inline
    UPolyconeSideRZ GetCorner(int index) const
  {
    return corners[index];
  }

  inline
    UPolyconeHistorical* GetOriginalParameters() const
  {
    return original_parameters;
  }

  inline
    void SetOriginalParameters(UPolyconeHistorical* pars)
  {
    if (!pars)
      // UException("UPolycone2::SetOriginalParameters()", "GeomSolids0002",
      //						FatalException, "NULL pointer to parameters!");
      *original_parameters = *pars;
  }

  inline
    void SetOriginalParameters()
  {
    int numPlanes = (int)numCorner/2; 

    original_parameters = new UPolyconeHistorical;

    original_parameters->Z_values = new double[numPlanes];
    original_parameters->Rmin = new double[numPlanes];
    original_parameters->Rmax = new double[numPlanes];

    for(int j=0; j < numPlanes; j++)
    {
      original_parameters->Z_values[j] = corners[numPlanes+j].z;
      original_parameters->Rmax[j] = corners[numPlanes+j].r;
      original_parameters->Rmin[j] = corners[numPlanes-1-j].r;
    }

    original_parameters->Start_angle = startPhi;
    original_parameters->Opening_angle = endPhi-startPhi;
    original_parameters->Num_z_planes = numPlanes;
  }

};


#endif
