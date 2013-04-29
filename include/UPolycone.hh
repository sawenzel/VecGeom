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
// $Id: UPolycone.hh 66241 2012-12-13 18:34:42Z gunter $
//
// 
// --------------------------------------------------------------------
// GEANT 4 class header file
//
//
// UPolycone
//
// Class description:
//
//	 Class implementing a CSG-like type "PCON" Geant 3.21 volume,
//	 inherited from	class UVCSGfaceted:
//
//	 UPolycone( const std::string& name, 
//							 double phiStart,		 // initial phi starting angle
//							 double phiTotal,		 // total phi angle
//							 int numZPlanes,		 // number of z planes
//							 const double zPlane[],	// position of z planes
//							 const double rInner[],	// tangent distance to inner surface
//							 const double rOuter[])	// tangent distance to outer surface
//
//	 UPolycone( const std::string& name, 
//							 double phiStart,	 // initial phi starting angle
//							 double phiTotal,	 // total phi angle
//							 int		numRZ,			// number corners in r,z space
//							 const double r[],	// r coordinate of these corners
//							 const double z[])	// z coordinate of these corners

// Author: 
//	 David C. Williams (davidw@scipp.ucsc.edu)
// --------------------------------------------------------------------

#ifndef UPolycone_hh
#define UPolycone_hh

#include "UVCSGfaceted.hh"
#include "UPolyconeSide.hh"

class UEnclosingCylinder;
class UReduciblePolygon;
class UVCSGface;
class UPolyconeHistorical
{
	public:
		UPolyconeHistorical();
		~UPolyconeHistorical();
		UPolyconeHistorical( const UPolyconeHistorical& source );
		UPolyconeHistorical& operator=( const UPolyconeHistorical& right );

		double Start_angle;
		double Opening_angle;
		int	 Num_z_planes;
		double *Z_values;
		double *Rmin;
		double *Rmax;
};

class UPolycone : public UVCSGfaceted 
{

 public:	// with description

	UPolycone( const std::string& name, 
										double phiStart,		 // initial phi starting angle
										double phiTotal,		 // total phi angle
										int numZPlanes,			// number of z planes
							const double zPlane[],		 // position of z planes
							const double rInner[],		 // tangent distance to inner surface
							const double rOuter[]	);	// tangent distance to outer surface

	UPolycone( const std::string& name, 
										double phiStart,		// initial phi starting angle
										double phiTotal,		// total phi angle
										int		numRZ,			 // number corners in r,z space
							const double r[],				 // r coordinate of these corners
							const double z[]			 ); // z coordinate of these corners

	virtual ~UPolycone();
	
	// Methods for solid

	VUSolid::EnumInside Inside( const UVector3 &p ) const;
	double DistanceToIn( const UVector3 &p, const UVector3 &v ) const;
	double SafetyFromOutside( const UVector3 &p, bool aAccurate=false) const;

	UVector3 GetPointOnSurface() const;

  /*
	void ComputeDimensions(			 UVPVParameterisation* p,
													const int n,
													const UVPhysicalVolume* pRep );
                          */

	UGeometryType GetEntityType() const;

	VUSolid* Clone() const;

	std::ostream& StreamInfo(std::ostream& os) const;

	UPolyhedron* CreatePolyhedron() const;

	bool Reset();

	// Accessors

	inline double GetStartPhi()	const;
	inline double GetEndPhi()		const;
	inline bool IsOpen()				 const;
	inline bool IsGeneric()			const;
	inline int	GetNumRZCorner() const;
	inline UPolyconeSideRZ GetCorner(int index) const;
	inline UPolyconeHistorical* GetOriginalParameters() const;
	inline void SetOriginalParameters(UPolyconeHistorical* pars);

 public:	// without description

	//UPolycone(__void__&);
		// Fake default constructor for usage restricted to direct object
		// persistency for clients requiring preallocation of memory for
		// persistifiable objects.

	UPolycone( const UPolycone &source );
	const UPolycone &operator=( const UPolycone &source );
		// Copy constructor and assignment operator.

 protected:	// without description

	// Generic initializer, called by all constructors

	inline void SetOriginalParameters();

	void Create( double phiStart,				// initial phi starting angle
							 double phiTotal,				// total phi angle
							 UReduciblePolygon *rz ); // r/z coordinate of these corners

	void CopyStuff( const UPolycone &source );

	// Methods for random point generation

	UVector3 GetPointOnCone(double fRmin1, double fRmax1,
															 double fRmin2, double fRmax2,
															 double zOne,	 double zTwo,
															 double& totArea) const;
	
	UVector3 GetPointOnTubs(double fRMin, double fRMax,
															 double zOne,	double zTwo,
															 double& totArea) const;
	
	UVector3 GetPointOnCut(double fRMin1, double fRMax1,
															double fRMin2, double fRMax2,
															double zOne,	 double zTwo,
															double& totArea) const;

	UVector3 GetPointOnRing(double fRMin, double fRMax,
															 double fRMin2, double fRMax2,
															 double zOne) const;

  void GetParametersList(int /*aNumber*/,double * /*aArray*/) const {}

  void ComputeBBox (UBBox * /*aBox*/, bool /*aStore*/)
  {
    // Computes bounding box.
    std::cout << "ComputeBBox - Not implemented" << std::endl;
  }

  void Extent (UVector3 &aMin, UVector3 &aMax) const;

 protected:	// without description

	// Here are our parameters

	double startPhi;		// Starting phi value (0 < phiStart < 2pi)
	double endPhi;			// end phi value (0 < endPhi-phiStart < 2pi)
	bool	 phiIsOpen;	 // true if there is a phi segment
	bool	 genericPcon; // true if created through the 2nd generic constructor
	int	 numCorner;		// number RZ points
	UPolyconeSideRZ *corners;	// corner r,z points
	UPolyconeHistorical	*original_parameters;	// original input parameters

	// Our quick test

	UEnclosingCylinder *enclosingCylinder;
	
};

#include "UPolycone.icc"

#endif
