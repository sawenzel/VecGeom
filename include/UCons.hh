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
// $Id: UCons.hh,v 1.24 2010-10-19 15:42:09 gcosmo Exp $
// GEANT4 tag $Name: not supported by cvs2svn $
//
//
// --------------------------------------------------------------------
// GEANT 4 class header file
//
// UCons
//
// Class description:
//
//	 A UCons is, in the general case, a Phi segment of a cone, with
//	 half-length fDz, inner and outer radii specified at -fDz and +fDz.
//	 The Phi segment is described by a starting fSPhi angle, and the
//	 +fDPhi delta angle for the shape.
//	 If the delta angle is >=2*UUtils::kPi, the shape is treated as continuous
//	 in Phi
//
//	 Member Data:
//
//	fRmin1	inside radius at	-fDz
//	fRmin2	inside radius at	+fDz
//	fRmax1	outside radius at -fDz
//	fRmax2	outside radius at +fDz
//	fDz	half length in z
//
//	fSPhi	starting angle of the segment in radians
//	fDPhi	delta angle of the segment in radians
//
//	fPhiFullCone	 Boolean variable used for indicate the Phi Section
//
//	 Note:
//			Internally fSPhi & fDPhi are adjusted so that fDPhi<=2PI,
//			and fDPhi+fSPhi<=2PI. This enables simpler comparisons to be
//			made with (say) Phi of a point.

// History:
// 19.3.94 P.Kent: Old C++ code converted to tolerant geometry
// 13.9.96 V.Grichine: Final modifications to commit
// --------------------------------------------------------------------

#ifndef UCons_HH
#define UCons_HH



#include "VUSolid.hh"

class UCons : public VUSolid
{
	public:	// with description

		UCons(const std::string& pName,
								 double pRmin1, double pRmax1,
								 double pRmin2, double pRmax2,
								 double pDz,
								 double pSPhi, double pDPhi);
			//
			// Constructs a cone with the given name and dimensions

	 ~UCons() ;
			//
			// Destructor

		// Accessors

		inline double GetInnerRadiusMinusZ() const;
		inline double GetOuterRadiusMinusZ() const;
		inline double GetInnerRadiusPlusZ()	const;
		inline double GetOuterRadiusPlusZ()	const;
		inline double GetZHalfLength()			 const;
		inline double GetStartPhiAngle()		 const;
		inline double GetDeltaPhiAngle()		 const;
	
		// Modifiers

		inline void SetInnerRadiusMinusZ (double Rmin1 );
		inline void SetOuterRadiusMinusZ (double Rmax1 );
		inline void SetInnerRadiusPlusZ	(double Rmin2 );
		inline void SetOuterRadiusPlusZ	(double Rmax2 );
		inline void SetZHalfLength			 (double newDz );
		inline void SetStartPhiAngle		 (double newSPhi, bool trig=true);
		inline void SetDeltaPhiAngle		 (double newDPhi);

		// Other methods for solid

		inline double GetCubicVolume();
		inline double GetSurfaceArea();

		
		//void ComputeDimensions(			 UVPVParameterisation* p,
		//												const int n,
		//												const UVPhysicalVolume* pRep );

		// bool CalculateExtent( const EAxisType pAxis,
		//												const UVoxelLimits& pVoxelLimit,
		//												const UAffineTransform& pTransform,
		//															double& pmin, double& pmax ) const;				 

		VUSolid::EnumInside Inside( const UVector3& p ) const;

		bool Normal( const UVector3& p, UVector3 &n) const;

		double DistanceToIn (const UVector3& p, const UVector3& v, double aPstep = UUtils::kInfinity) const;

		double SafetyFromOutside (const UVector3& p, bool precise) const;



        double DistanceToOut( const UVector3 &aPoint,
                                const UVector3  &aDirection,
                                UVector3       &aNormalVector,
                                bool           &aConvex,
                                double aPstep = UUtils::kInfinity) const;
	 
		double SafetyFromInside(const UVector3& p,bool precise) const;

		UGeometryType GetEntityType() const;
				
		UVector3 GetPointOnSurface() const; 
				
		VUSolid* Clone() const;

		std::ostream& StreamInfo(std::ostream& os) const;

    	void Extent (EAxisType aAxis, double &aMin, double &aMax) const;
	    void Extent (UVector3 &aMin, UVector3 &aMax) const;

		virtual void GetParametersList(int /*aNumber*/,double * /*aArray*/) const{} 

		virtual void ComputeBBox(UBBox * /*aBox*/, bool /*aStore = false*/) {}

		// Visualisation functions

		UPolyhedron* GetPolyhedron() const {return CreatePolyhedron();}

		// void					DescribeYourselfTo( UVGraphicsScene& scene ) const;
		UPolyhedron* CreatePolyhedron() const;
		// UNURBS*			CreateNURBS() const;

	public:	// without description
			 
		UCons();
			//
			// Fake default constructor for usage restricted to direct object
			// persistency for clients requiring preallocation of memory for
			// persistifiable objects.

		UCons(const UCons& rhs);
		UCons& operator=(const UCons& rhs); 
			// Copy constructor and assignment operator.

		//	Old access functions

		inline double		GetRmin1() const;
		inline double		GetRmax1() const;
		inline double		GetRmin2() const;
		inline double		GetRmax2() const;
		inline double		GetDz()		const;
		inline double		GetSPhi() const;
		inline double		GetDPhi() const;

	private:

		double fCubicVolume, fSurfaceArea;
		UPolyhedron *fpPolyhedron;

	    inline double Capacity();
	    inline double SurfaceArea();

//		UVector3List*
//		CreateRotatedVertices(const UAffineTransform& pTransform) const;
	
		inline void Initialize();
			//
			// Reset relevant values to zero

		inline void CheckSPhiAngle(double sPhi);
		inline void CheckDPhiAngle(double dPhi);
		inline void CheckPhiAngles(double sPhi, double dPhi);
			//
			// Reset relevant flags and angle values

		inline void InitializeTrigonometry();
			//
			// Recompute relevant trigonometric values and cache them

		UVector3 ApproxSurfaceNormal(const UVector3& p) const;
			//
			// Algorithm for SurfaceNormal() following the original
			// specification for points not on the surface

	private:

		// Used by distanceToOut
		//
		enum ESide {kNull,kRMin,kRMax,kSPhi,kEPhi,kPZ,kMZ};
	
		// used by normal
		//
		enum ENorm {kNRMin,kNRMax,kNSPhi,kNEPhi,kNZ};

		double kRadTolerance, kAngTolerance;
			//
			// Radial and angular tolerances

		double fRmin1, fRmin2, fRmax1, fRmax2, fDz, fSPhi, fDPhi;
			//
			// Radial and angular dimensions

		double sinCPhi, cosCPhi, cosHDPhiOT, cosHDPhiIT,
						 sinSPhi, cosSPhi, sinEPhi, cosEPhi;
			//
			// Cached trigonometric values

		bool fPhiFullCone;
			//
			// Flag for identification of section or full cone
};

#include "UCons.icc"

#endif
