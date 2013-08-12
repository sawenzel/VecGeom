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

#ifndef UPolycone3_hh
#define UPolycone3_hh

#include "VUSolid.hh"

#include "UMultiUnion.hh"
//#include "UPolyconeSide.hh"
#include "UPolycone.hh"

#include "UVoxelizer.hh"

#include "UCons.hh"
#include "UTubs.hh"

class UPolycone3 : public VUSolid
{

public:	// with description

  void Init(
    double phiStart,		 // initial phi starting angle
    double phiTotal,		 // total phi angle
    int numZPlanes,			// number of z planes
    const double zPlane[],		 // position of z planes
    const double rInner[],		 // tangent distance to inner surface
    const double rOuter[]);

  UPolycone3( const std::string& name) : VUSolid(name)
  {

  }

  UPolycone3( const std::string& name, 
    double phiStart,		 // initial phi starting angle
    double phiTotal,		 // total phi angle
    int numZPlanes,			// number of z planes
    const double zPlane[],		 // position of z planes
    const double rInner[],		 // tangent distance to inner surface
    const double rOuter[]	);	// tangent distance to outer surface

  /*
  UPolycone3( const std::string& name, 
    double phiStart,		// initial phi starting angle
    double phiTotal,		// total phi angle
    int		numRZ,			 // number corners in r,z space
    const double r[],				 // r coordinate of these corners
    const double z[]			 ); // z coordinate of these corners
    */

  virtual ~UPolycone3();

//  inline void SetOriginalParameters(UPolyconeHistorical* pars);

//  inline void SetOriginalParameters();

  std::ostream& StreamInfo( std::ostream& os ) const;

  VUSolid::EnumInside Inside( const UVector3 &p ) const;

  double DistanceToIn( const UVector3 &p, const UVector3 &v, double aPstep=UUtils::kInfinity) const;

  double  SafetyFromInside ( const UVector3 &aPoint, 
    bool aAccurate=false) const;
  double  SafetyFromOutside( const UVector3 &aPoint, 
    bool aAccurate=false) const;

  double DistanceToOut     ( const UVector3 &aPoint,
    const UVector3 &aDirection,
    UVector3       &aNormalVector, 
    bool           &aConvex,
    double aPstep = UUtils::kInfinity) const;

  bool Normal ( const UVector3& aPoint, UVector3 &aNormal ) const; 
  //	virtual void Extent ( EAxisType aAxis, double &aMin, double &aMax ) const;
  void Extent (UVector3 &aMin, UVector3 &aMax) const; 
  double Capacity();
  double SurfaceArea();
  UGeometryType GetEntityType() const { return "Polycone";}
  void ComputeBBox(UBBox * /*aBox*/, bool /*aStore = false*/) {}

  //G4Visualisation
  void GetParametersList(int /*aNumber*/, double * /*aArray*/) const{} 
  UPolyhedron* GetPolyhedron() const{return CreatePolyhedron(); }

  inline VUSolid* Clone() const
  {
    return NULL;
  }

  UVector3 GetPointOnSurface() const;

  UPolyhedron* CreatePolyhedron () const;

protected:	// without description

  int fNumSides;

  // Here are our parameters

  double startPhi;		// Starting phi value (0 < phiStart < 2pi)
  double endPhi;			// end phi value (0 < endPhi-phiStart < 2pi)
  bool	 phiIsOpen;	 // true if there is a phi segment
  bool	 genericPcon; // true if created through the 2nd generic constructor
  int	 numCorner;		// number RZ points
  UPolyconeSideRZ *corners;	// corner r,z points
  UPolyconeHistorical	*fOriginalParameters;	// original input parameters

  inline double GetStartPhi() const
  {
    return startPhi;
  }

  inline double GetEndPhi() const
  {
    return endPhi;
  }

  inline bool IsOpen() const
  {
    return phiIsOpen;
  }

  inline bool IsGeneric() const
  {
    return genericPcon;
  }

  inline int GetNumRZCorner() const
  {
    return numCorner;
  }

  inline UPolyconeSideRZ GetCorner(int index) const
  {
    return corners[index];
  }

  inline UPolyconeHistorical* GetOriginalParameters() const
  {
    return fOriginalParameters;
  }

  inline void SetOriginalParameters(UPolyconeHistorical* pars)
  {
    if (!pars)
      // UException("UPolycone3::SetOriginalParameters()", "GeomSolids0002",
      //						FatalException, "NULL pointer to parameters!");
      *fOriginalParameters = *pars;
  }

  inline void SetOriginalParameters()
  {
    int numPlanes = (int)numCorner/2; 

    fOriginalParameters = new UPolyconeHistorical;

    fOriginalParameters->fZValues.resize(numPlanes);
    fOriginalParameters->Rmin.resize(numPlanes);
    fOriginalParameters->Rmax.resize(numPlanes);

    for(int j=0; j < numPlanes; j++)
    {
      fOriginalParameters->fZValues[j] = corners[numPlanes+j].z;
      fOriginalParameters->Rmax[j] = corners[numPlanes+j].r;
      fOriginalParameters->Rmin[j] = corners[numPlanes-1-j].r;
    }

    fOriginalParameters->fStartAngle = startPhi;
    fOriginalParameters->fOpeningAngle = endPhi-startPhi;
    fOriginalParameters->fNumZPlanes = numPlanes;
  }

  UEnclosingCylinder *enclosingCylinder;

  struct UPolyconeSection
  {
    VUSolid *solid;// true if all points in section are concave in regards to whole polycone, will be determined
    double shift;
    bool tubular;
//    double left, right;
    bool convex; // TURE if all points in section are concave in regards to whole polycone, will be determined, currently not implemented
  };

  std::vector<double> fZs; // z coordinates of given sections
  std::vector<UPolyconeSection> fSections;
  int fMaxSection;

  inline VUSolid::EnumInside InsideSection(int index, const UVector3 &p) const;

  inline double SafetyFromInsideSection(int index, const UVector3 &p) const
  {
    const UPolyconeSection &section = fSections[index];
    UVector3 ps(p.x, p.y, p.z - section.shift);
    double res = section.solid->SafetyFromInside(ps, true);
    return res;
  }

  inline double SafetyFromOutsideSection(int index, const UVector3 &p) const
  {
    const UPolyconeSection &section = fSections[index];
    UVector3 ps(p.x, p.y, p.z - section.shift);
    double res = section.solid->SafetyFromOutside(ps, true);
    return res;
  }

  bool NormalSection(int index, const UVector3 &p, UVector3 &n) const
  {
    const UPolyconeSection &section = fSections[index];
    UVector3 ps(p.x, p.y, p.z - section.shift);
    bool res = section.solid->Normal(ps, n);
    return res;
  }

  inline int GetSection(double z) const
  {
    int section = UVoxelizer::BinarySearch(fZs, z);
    if (section < 0) section = 0;
    else if (section > fMaxSection) section = fMaxSection;
    return section;
  }
};

#endif
