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

#ifndef UVCSGfaceted_hh
#define UVCSGfaceted_hh

#include "VUSolid.hh"

class UVCSGface;
class UVisExtent;

class UVCSGfaceted : public VUSolid 
{
	public:	// with description

		UVCSGfaceted( const std::string& name );
		virtual ~UVCSGfaceted();
	
		UVCSGfaceted( const UVCSGfaceted &source );
		const UVCSGfaceted &operator=( const UVCSGfaceted &source );

    /*
		virtual bool CalculateExtent( const EAxisType pAxis,
																		const UVoxelLimits& pVoxelLimit,
																		const UAffineTransform& pTransform,
																					double& pmin,double& pmax ) const;
                                          */
	
		virtual VUSolid::EnumInside Inside( const UVector3& p ) const;

    VUSolid::EnumInside InsideOptimized( const UVector3& p ) const;

		virtual bool Normal( const UVector3 &p, UVector3 &n) const;

		virtual double DistanceToIn( const UVector3& p,
																	 const UVector3& v, double aPstep = UUtils::kInfinity) const;


    virtual double SafetyFromOutside( const UVector3 &aPoint, bool aAccurate=false) const;


    double DistanceTo( const UVector3 &p, const bool outgoing ) const;

    /*
		virtual double DistanceToOut( const UVector3& p,
																		const UVector3& v,
																		const bool calcNorm=false,
																					bool *validNorm=0,
																					UVector3 *n=0 ) const;
                                          */

    virtual double DistanceToOut( const UVector3 &p,
      const UVector3  &v,
      UVector3       &n,
      bool           &aConvex,
      double aPstep = UUtils::kInfinity) const;


//		virtual double DistanceToOut( const UVector3& p ) const;

    virtual double SafetyFromInside ( const UVector3 &aPoint, bool aAccurate=false) const;

		virtual UGeometryType GetEntityType() const;

		virtual std::ostream& StreamInfo(std::ostream& os) const;

		virtual UPolyhedron* CreatePolyhedron() const = 0;

//		virtual void DescribeYourselfTo( UVGraphicsScene& scene ) const;

//		virtual UVisExtent GetExtent() const;

		virtual UPolyhedron* GetPolyhedron () const;

		int GetCubVolStatistics() const;
		double GetCubVolEpsilon() const;
		void SetCubVolStatistics(int st);
		void SetCubVolEpsilon(double ep);
		int GetAreaStatistics() const;
		double GetAreaAccuracy() const;
		void SetAreaStatistics(int st);
		void SetAreaAccuracy(double ep);

		virtual double Capacity();
			// Returns an estimation of the geometrical cubic volume of the
			// solid. Caches the computed value once computed the first time.
		virtual double SurfaceArea();
			// Returns an estimation of the geometrical surface area of the
			// solid. Caches the computed value once computed the first time.

	public:	// without description


	protected:	// without description

		int		numFace;
		UVCSGface **faces;
		double fCubicVolume;
		double fSurfaceArea;
		mutable UPolyhedron* fpPolyhedron;

		UVector3 GetPointOnSurfaceGeneric()const;
			// Returns a random point located on the surface of the solid 
			// in case of generic Polycone or generic Polyhedra.

		void CopyStuff( const UVCSGfaceted &source );
		void DeleteStuff();

	private:

		int		fStatistics;
		double fCubVolEpsilon;
		double fAreaAccuracy;
			// Statistics, error accuracy for volume estimation.

};

#endif
