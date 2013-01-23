/*
* Implemented:

UVector3 UTubs::GetPointOnSurface() const;

std::ostream& StreamInfo (std::ostream& os) const;

* Not yet implemented:
UPolyhedron* CreatePolyhedron () const;

UPolyhedron* GetPolyhedron()

??? VUSolid* UTubs::Clone() const

??? void DescribeYourselfTo ( G4VGraphicsScene& scene ) const;

??? void ComputeDimensions(G4VPVParameterisation* p, const int n, const G4VPhysicalVolume* pRep );

??? UNURBS* CreateNURBS () const;
*/

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
// $Id: UTubs.cc,v 1.84 2010-10-19 15:42:10 gcosmo Exp $
// GEANT4 tag $Name: not supported by cvs2svn $
//
// 
// class UTubs
//
// History:
//
// 05.04.12 M.Kelsey:	 Use sqrt(r) in GetPointOnSurface() for uniform points
// 02.08.07 T.Nikitina: bug fixed in DistanceToOut(p,v,..) for negative value under sqrt
//											for the case: p on the surface and v is tangent to the surface
// 11.05.07 T.Nikitina: bug fixed in DistanceToOut(p,v,..) for phi < 2UUtils::kPi
// 03.05.05 V.Grichine: SurfaceNormal(p) according to J. Apostolakis proposal
// 16.03.05 V.Grichine: SurfaceNormal(p) with edges/corners for boolean
// 20.07.01 V.Grichine: bug fixed in Inside(p)
// 20.02.01 V.Grichine: bug fixed in Inside(p) and CalculateExtent was 
//											simplified base on UBox::CalculateExtent
// 07.12.00 V.Grichine: phi-section algorithm was changed in Inside(p)
// 28.11.00 V.Grichine: bug fixed in Inside(p)
// 31.10.00 V.Grichine: assign srd, sphi in Distance ToOut(p,v,...)
// 08.08.00 V.Grichine: more stable roots of 2-equation in DistanceToOut(p,v,..)
// 02.08.00 V.Grichine: point is outside check in Distance ToOut(p)
// 17.05.00 V.Grichine: bugs (#76,#91) fixed in Distance ToOut(p,v,...)
// 31.03.00 V.Grichine: bug fixed in Inside(p)
// 19.11.99 V.Grichine: side = kNull in DistanceToOut(p,v,...)
// 13.10.99 V.Grichine: bugs fixed in DistanceToIn(p,v) 
// 28.05.99 V.Grichine: bugs fixed in DistanceToOut(p,v,...)
// 25.05.99 V.Grichine: bugs fixed in DistanceToIn(p,v) 
// 23.03.99 V.Grichine: bug fixed in DistanceToIn(p,v) 
// 09.10.98 V.Grichine: modifications in DistanceToOut(p,v,...)
// 18.06.98 V.Grichine: n-normalisation in DistanceToOut(p,v)
// 
// 1994-95	P.Kent:		 implementation
//
/////////////////////////////////////////////////////////////////////////

#include "UUtils.hh"
#include <string>
#include <cmath>
#include <sstream>
#include "UTubs.hh"

using namespace std;

/////////////////////////////////////////////////////////////////////////
//
// Constructor - check parameters, convert angles so 0<sphi+dpshi<=2_PI
//						 - note if pdphi>2PI then reset to 2PI

UTubs::UTubs( const std::string &pName,
	double pRMin, double pRMax,
	double pDz,
	double pSPhi, double pDPhi )
	: VUSolid(pName.c_str()), fRMin(pRMin), fRMax(pRMax), fDz(pDz), fSPhi(0), fDPhi(0)
{

	kRadTolerance = frTolerance;
	kAngTolerance = faTolerance;

	if (pDz<=0) // Check z-len
	{
		std::ostringstream message;
		message << "Negative Z half-length (" << pDz << ") in solid: " << GetName();
		// UException("UTubs::UTubs()", "GeomSolids0002", FatalException, message);
	}
	if ( (pRMin >= pRMax) || (pRMin < 0) ) // Check radii
	{
		std::ostringstream message;
		message << "Invalid values for radii in solid: " << GetName()
			<< std::endl
			<< "				pRMin = " << pRMin << ", pRMax = " << pRMax;
		// UException("UTubs::UTubs()", "GeomSolids0002", FatalException, message);
	}

	// Check angles
	//
	CheckPhiAngles(pSPhi, pDPhi);
}

///////////////////////////////////////////////////////////////////////
//
// Fake default constructor - sets only member data and allocates memory
//														for usage restricted to object persistency.
//
UTubs::UTubs()
	: VUSolid(""), kRadTolerance(0.), kAngTolerance(0.),
	fRMin(0.), fRMax(0.), fDz(0.), fSPhi(0.), fDPhi(0.),
	fSinCPhi(0.), fCosCPhi(0.), fCosHDPhiOT(0.), fCosHDPhiIT(0.),
	fSinSPhi(0.), fCosSPhi(0.), fSinEPhi(0.), fCosEPhi(0.),
	fPhiFullTube(false)
{
}

//////////////////////////////////////////////////////////////////////////
//
// Destructor

UTubs::~UTubs()
{
}

//////////////////////////////////////////////////////////////////////////
//
// Copy constructor

UTubs::UTubs(const UTubs& rhs)
	: VUSolid(rhs),
	kRadTolerance(rhs.kRadTolerance), kAngTolerance(rhs.kAngTolerance),
	fRMin(rhs.fRMin), fRMax(rhs.fRMax), fDz(rhs.fDz),
	fSPhi(rhs.fSPhi), fDPhi(rhs.fDPhi),
	fSinCPhi(rhs.fSinCPhi), fCosCPhi(rhs.fSinCPhi),
	fCosHDPhiOT(rhs.fCosHDPhiOT), fCosHDPhiIT(rhs.fCosHDPhiOT),
	fSinSPhi(rhs.fSinSPhi), fCosSPhi(rhs.fCosSPhi),
	fSinEPhi(rhs.fSinEPhi), fCosEPhi(rhs.fCosEPhi), fPhiFullTube(rhs.fPhiFullTube)
{
}

//////////////////////////////////////////////////////////////////////////
//
// Assignment operator

UTubs& UTubs::operator = (const UTubs& rhs) 
{
	// Check assignment to self
	//
	if (this == &rhs) {return *this;}

	// Copy base class data
	//
	VUSolid::operator=(rhs);

	// Copy data
	//
	kRadTolerance = rhs.kRadTolerance; kAngTolerance = rhs.kAngTolerance;
	fRMin = rhs.fRMin; fRMax = rhs.fRMax; fDz = rhs.fDz;
	fSPhi = rhs.fSPhi; fDPhi = rhs.fDPhi;
	fSinCPhi = rhs.fSinCPhi; fCosCPhi = rhs.fSinCPhi;
	fCosHDPhiOT = rhs.fCosHDPhiOT; fCosHDPhiIT = rhs.fCosHDPhiOT;
	fSinSPhi = rhs.fSinSPhi; fCosSPhi = rhs.fCosSPhi;
	fSinEPhi = rhs.fSinEPhi; fCosEPhi = rhs.fCosEPhi;
	fPhiFullTube = rhs.fPhiFullTube;

	return *this;
}

/////////////////////////////////////////////////////////////////////////
//
// Dispatch to parameterisation for replication mechanism dimension
// computation & modification.

/*
void UTubs::ComputeDimensions(			 UVPVParameterisation* p,
const int n,
const UVPhysicalVolume* pRep )
{
p->ComputeDimensions(*this,n,pRep);
}
*/

////////////////////////////////////////////////////////////////////////
//
// Calculate extent under transform and specified limit

/*
bool UTubs::CalculateExtent( const EAxisType							pAxis,
const UVoxelLimits&		 pVoxelLimit,
const UAffineTransform& pTransform,
double&					pMin, 
double&					pMax		) const
{

if ( (!pTransform.IsRotated()) && (fDPhi == 2*UUtils::kPi) && (fRMin == 0) )
{
// Special case handling for unrotated solid tubes
// Compute x/y/z mins and maxs fro bounding box respecting limits,
// with early returns if outside limits. Then switch() on pAxis,
// and compute exact x and y limit for x/y case

double xoffset, xMin, xMax;
double yoffset, yMin, yMax;
double zoffset, zMin, zMax;

double diff1, diff2, maxDiff, newMin, newMax;
double xoff1, xoff2, yoff1, yoff2, delta;

xoffset = pTransform.NetTranslation().x;
xMin = xoffset - fRMax;
xMax = xoffset + fRMax;

if (pVoxelLimit.IsXLimited())
{
if ( (xMin > pVoxelLimit.GetMaxXExtent())
|| (xMax < pVoxelLimit.GetMinXExtent()) )
{
return false;
}
else
{
if (xMin < pVoxelLimit.GetMinXExtent())
{
xMin = pVoxelLimit.GetMinXExtent();
}
if (xMax > pVoxelLimit.GetMaxXExtent())
{
xMax = pVoxelLimit.GetMaxXExtent();
}
}
}
yoffset = pTransform.NetTranslation().y;
yMin		= yoffset - fRMax;
yMax		= yoffset + fRMax;

if ( pVoxelLimit.IsYLimited() )
{
if ( (yMin > pVoxelLimit.GetMaxYExtent())
|| (yMax < pVoxelLimit.GetMinYExtent()) )
{
return false;
}
else
{
if (yMin < pVoxelLimit.GetMinYExtent())
{
yMin = pVoxelLimit.GetMinYExtent();
}
if (yMax > pVoxelLimit.GetMaxYExtent())
{
yMax=pVoxelLimit.GetMaxYExtent();
}
}
}
zoffset = pTransform.NetTranslation().z;
zMin		= zoffset - fDz;
zMax		= zoffset + fDz;

if ( pVoxelLimit.IsZLimited() )
{
if ( (zMin > pVoxelLimit.GetMaxZExtent())
|| (zMax < pVoxelLimit.GetMinZExtent()) )
{
return false;
}
else
{
if (zMin < pVoxelLimit.GetMinZExtent())
{
zMin = pVoxelLimit.GetMinZExtent();
}
if (zMax > pVoxelLimit.GetMaxZExtent())
{
zMax = pVoxelLimit.GetMaxZExtent();
}
}
}
switch ( pAxis )	// Known to cut cylinder
{
case eXaxis :
{
yoff1 = yoffset - yMin;
yoff2 = yMax		- yoffset;

if ( (yoff1 >= 0) && (yoff2 >= 0) ) // Y limits Cross max/min x
{																	 // => no change
pMin = xMin;
pMax = xMax;
}
else
{
// Y limits don't Cross max/min x => compute max delta x,
// hence new mins/maxs

delta	 = fRMax*fRMax - yoff1*yoff1;
diff1	 = (delta>0.) ? std::sqrt(delta) : 0.;
delta	 = fRMax*fRMax - yoff2*yoff2;
diff2	 = (delta>0.) ? std::sqrt(delta) : 0.;
maxDiff = (diff1 > diff2) ? diff1:diff2;
newMin	= xoffset - maxDiff;
newMax	= xoffset + maxDiff;
pMin		= (newMin < xMin) ? xMin : newMin;
pMax		= (newMax > xMax) ? xMax : newMax;
}		
break;
}
case eYaxis :
{
xoff1 = xoffset - xMin;
xoff2 = xMax - xoffset;

if ( (xoff1 >= 0) && (xoff2 >= 0) ) // X limits Cross max/min y
{																	 // => no change
pMin = yMin;
pMax = yMax;
}
else
{
// X limits don't Cross max/min y => compute max delta y,
// hence new mins/maxs

delta	 = fRMax*fRMax - xoff1*xoff1;
diff1	 = (delta>0.) ? std::sqrt(delta) : 0.;
delta	 = fRMax*fRMax - xoff2*xoff2;
diff2	 = (delta>0.) ? std::sqrt(delta) : 0.;
maxDiff = (diff1 > diff2) ? diff1 : diff2;
newMin	= yoffset - maxDiff;
newMax	= yoffset + maxDiff;
pMin		= (newMin < yMin) ? yMin : newMin;
pMax		= (newMax > yMax) ? yMax : newMax;
}
break;
}
case eZaxis:
{
pMin = zMin;
pMax = zMax;
break;
}
default:
break;
}
pMin -= VUSolid::Tolerance();
pMax += VUSolid::Tolerance();
return true;
}
else // Calculate rotated vertex coordinates
{
int i, noEntries, noBetweenSections4;
bool existsAfterClip = false;
UVector3List* vertices = CreateRotatedVertices(pTransform);

pMin =	UUtils::Infinity();
pMax = -UUtils::Infinity();

noEntries = vertices->size();
noBetweenSections4 = noEntries - 4;

for ( i = 0; i < noEntries; i += 4 )
{
ClipCrossSection(vertices, i, pVoxelLimit, pAxis, pMin, pMax);
}
for ( i = 0; i < noBetweenSections4; i += 4 )
{
ClipBetweenSections(vertices, i, pVoxelLimit, pAxis, pMin, pMax);
}
if ( (pMin != UUtils::Infinity()) || (pMax != -UUtils::Infinity()) )
{
existsAfterClip = true;
pMin -= VUSolid::Tolerance(); // Add 2*tolerance to avoid precision troubles
pMax += VUSolid::Tolerance();
}
else
{
// Check for case where completely enveloUUtils::kPing clipUUtils::kPing volume
// If point inside then we are confident that the solid completely
// envelopes the clipUUtils::kPing volume. Hence Set min/max extents according
// to clipUUtils::kPing volume extents along the specified axis.

UVector3 clipCentre(
(pVoxelLimit.GetMinXExtent()+pVoxelLimit.GetMaxXExtent())*0.5,
(pVoxelLimit.GetMinYExtent()+pVoxelLimit.GetMaxYExtent())*0.5,
(pVoxelLimit.GetMinZExtent()+pVoxelLimit.GetMaxZExtent())*0.5 );

if ( Inside(pTransform.Inverse().TransformPoint(clipCentre)) != eOutside )
{
existsAfterClip = true;
pMin						= pVoxelLimit.GetMinExtent(pAxis);
pMax						= pVoxelLimit.GetMaxExtent(pAxis);
}
}
delete vertices;
return existsAfterClip;
}
}
*/

///////////////////////////////////////////////////////////////////////////
//
// Return whether point inside/outside/on surface

VUSolid::EnumInside UTubs::Inside( const UVector3& p ) const
{
	double r2,pPhi,tolRMin,tolRMax;
	VUSolid::EnumInside in = eOutside;
	static const double halfCarTolerance=VUSolid::Tolerance()*0.5;
	static const double halfRadTolerance=kRadTolerance*0.5;
	static const double halfAngTolerance=kAngTolerance*0.5;

	if (std::fabs(p.z) <= fDz - halfCarTolerance)
	{
		r2 = p.x*p.x + p.y*p.y;

		if (fRMin) { tolRMin = fRMin + halfRadTolerance; }
		else			 { tolRMin = 0; }

		tolRMax = fRMax - halfRadTolerance;

		if ((r2 >= tolRMin*tolRMin) && (r2 <= tolRMax*tolRMax))
		{
			if ( fPhiFullTube )
			{
				in = eInside;
			}
			else
			{
				// Try inner tolerant phi boundaries (=>inside)
				// if not inside, try outer tolerant phi boundaries

				if ( (tolRMin==0) && (std::fabs(p.x)<=halfCarTolerance)
					&& (std::fabs(p.y)<=halfCarTolerance) )
				{
					in=eSurface;
				}
				else
				{
					pPhi = std::atan2(p.y,p.x);
					if ( pPhi < -halfAngTolerance )	{ pPhi += 2*UUtils::kPi; } // 0<=pPhi<2UUtils::kPi

					if ( fSPhi >= 0 )
					{
						if ( (std::fabs(pPhi) < halfAngTolerance)
							&& (std::fabs(fSPhi + fDPhi - 2*UUtils::kPi) < halfAngTolerance) )
						{ 
							pPhi += 2*UUtils::kPi; // 0 <= pPhi < 2UUtils::kPi
						}
						if ( (pPhi >= fSPhi + halfAngTolerance)
							&& (pPhi <= fSPhi + fDPhi - halfAngTolerance) )
						{
							in = eInside;
						}
						else if ( (pPhi >= fSPhi - halfAngTolerance)
							&& (pPhi <= fSPhi + fDPhi + halfAngTolerance) )
						{
							in = eSurface;
						}
					}
					else	// fSPhi < 0
					{
						if ( (pPhi <= fSPhi + 2*UUtils::kPi - halfAngTolerance)
							&& (pPhi >= fSPhi + fDPhi	+ halfAngTolerance) ) {;} //eOutside
						else if ( (pPhi <= fSPhi + 2*UUtils::kPi + halfAngTolerance)
							&& (pPhi >= fSPhi + fDPhi	- halfAngTolerance) )
						{
							in = eSurface;
						}
						else
						{
							in = eInside;
						}
					}
				}										
			}
		}
		else	// Try generous boundaries
		{
			tolRMin = fRMin - halfRadTolerance;
			tolRMax = fRMax + halfRadTolerance;

			if ( tolRMin < 0 )	{ tolRMin = 0; }

			if ( (r2 >= tolRMin*tolRMin) && (r2 <= tolRMax*tolRMax) )
			{
				if (fPhiFullTube || (r2 <=halfRadTolerance*halfRadTolerance) )
				{												// Continuous in phi or on z-axis
					in = eSurface;
				}
				else // Try outer tolerant phi boundaries only
				{
					pPhi = std::atan2(p.y,p.x);

					if ( pPhi < -halfAngTolerance)	{ pPhi += 2*UUtils::kPi; } // 0<=pPhi<2UUtils::kPi
					if ( fSPhi >= 0 )
					{
						if ( (std::fabs(pPhi) < halfAngTolerance)
							&& (std::fabs(fSPhi + fDPhi - 2*UUtils::kPi) < halfAngTolerance) )
						{ 
							pPhi += 2*UUtils::kPi; // 0 <= pPhi < 2UUtils::kPi
						}
						if ( (pPhi >= fSPhi - halfAngTolerance)
							&& (pPhi <= fSPhi + fDPhi + halfAngTolerance) )
						{
							in = eSurface;
						}
					}
					else	// fSPhi < 0
					{
						if ( (pPhi <= fSPhi + 2*UUtils::kPi - halfAngTolerance)
							&& (pPhi >= fSPhi + fDPhi + halfAngTolerance) ) {;} // eOutside
						else
						{
							in = eSurface;
						}
					}
				}
			}
		}
	}
	else if (std::fabs(p.z) <= fDz + halfCarTolerance)
	{																					// Check within tolerant r limits
		r2 = p.x*p.x + p.y*p.y;
		tolRMin = fRMin - halfRadTolerance;
		tolRMax = fRMax + halfRadTolerance;

		if ( tolRMin < 0 )	{ tolRMin = 0; }

		if ( (r2 >= tolRMin*tolRMin) && (r2 <= tolRMax*tolRMax) )
		{
			if (fPhiFullTube || (r2 <=halfRadTolerance*halfRadTolerance))
			{												// Continuous in phi or on z-axis
				in = eSurface;
			}
			else // Try outer tolerant phi boundaries
			{
				pPhi = std::atan2(p.y,p.x);

				if ( pPhi < -halfAngTolerance )	{ pPhi += 2*UUtils::kPi; }	// 0<=pPhi<2UUtils::kPi
				if ( fSPhi >= 0 )
				{
					if ( (std::fabs(pPhi) < halfAngTolerance)
						&& (std::fabs(fSPhi + fDPhi - 2*UUtils::kPi) < halfAngTolerance) )
					{ 
						pPhi += 2*UUtils::kPi; // 0 <= pPhi < 2UUtils::kPi
					}
					if ( (pPhi >= fSPhi - halfAngTolerance)
						&& (pPhi <= fSPhi + fDPhi + halfAngTolerance) )
					{
						in = eSurface;
					}
				}
				else	// fSPhi < 0
				{
					if ( (pPhi <= fSPhi + 2*UUtils::kPi - halfAngTolerance)
						&& (pPhi >= fSPhi + fDPhi	+ halfAngTolerance) ) {;}
					else
					{
						in = eSurface;
					}
				}			
			}
		}
	}
	return in;
}

///////////////////////////////////////////////////////////////////////////
//
// Return Unit normal of surface closest to p
// - note if point on z axis, ignore phi divided sides
// - unsafe if point close to z axis a rmin=0 - no explicit checks

bool UTubs::Normal( const UVector3& p, UVector3& n) const
{
	int noSurfaces = 0;
	double rho, pPhi;
	double distZ, distRMin, distRMax;
	double distSPhi = UUtils::Infinity(), distEPhi = UUtils::Infinity();

	static const double halfCarTolerance = 0.5*VUSolid::Tolerance();
	static const double halfAngTolerance = 0.5*kAngTolerance;

	UVector3 norm, sumnorm(0.,0.,0.);
	UVector3 nZ = UVector3(0, 0, 1.0);
	UVector3 nR, nPs, nPe;

	rho = std::sqrt(p.x*p.x + p.y*p.y);

	distRMin = std::fabs(rho - fRMin);
	distRMax = std::fabs(rho - fRMax);
	distZ		= std::fabs(std::fabs(p.z) - fDz);

	if (!fPhiFullTube)		// Protected against (0,0,z) 
	{
		if ( rho > halfCarTolerance )
		{
			pPhi = std::atan2(p.y,p.x);

			if(pPhi	< fSPhi- halfCarTolerance)					 { pPhi += 2*UUtils::kPi; }
			else if(pPhi > fSPhi+fDPhi+ halfCarTolerance) { pPhi -= 2*UUtils::kPi; }

			distSPhi = std::fabs(pPhi - fSPhi);			 
			distEPhi = std::fabs(pPhi - fSPhi - fDPhi); 
		}
		else if( !fRMin )
		{
			distSPhi = 0.; 
			distEPhi = 0.; 
		}
		nPs = UVector3(std::sin(fSPhi),-std::cos(fSPhi),0);
		nPe = UVector3(-std::sin(fSPhi+fDPhi),std::cos(fSPhi+fDPhi),0);
	}
	if ( rho > halfCarTolerance ) { nR = UVector3(p.x/rho,p.y/rho,0); }

	if( distRMax <= halfCarTolerance )
	{
		noSurfaces ++;
		sumnorm += nR;
	}
	if( fRMin && (distRMin <= halfCarTolerance) )
	{
		noSurfaces ++;
		sumnorm -= nR;
	}
	if( fDPhi < 2*UUtils::kPi )	 
	{
		if (distSPhi <= halfAngTolerance)	
		{
			noSurfaces ++;
			sumnorm += nPs;
		}
		if (distEPhi <= halfAngTolerance)	
		{
			noSurfaces ++;
			sumnorm += nPe;
		}
	}
	if (distZ <= halfCarTolerance)	
	{
		noSurfaces ++;
		if ( p.z >= 0.)	{ sumnorm += nZ; }
		else							 { sumnorm -= nZ; }
	}
	if ( noSurfaces == 0 )
	{
#ifdef UCSGDEBUG
		// UException("UTubs::SurfaceNormal(p)", "GeomSolids1002",
		JustWarning, "Point p is not on surface !?" );
		int oldprc = cout.precision(20);
		cout<< "UTubs::SN ( "<<p.x<<", "<<p.y<<", "<<p.z<<" ); "
			<< std::endl << std::endl;
		cout.precision(oldprc);
#endif 
		norm = ApproxSurfaceNormal(p);
	}
	else if ( noSurfaces == 1 )	{ norm = sumnorm; }
	else												 { norm = sumnorm.Unit(); }

	n = norm;

	return noSurfaces; // TODO: return true or false on validity
}

/////////////////////////////////////////////////////////////////////////////
//
// Algorithm for SurfaceNormal() following the original specification
// for points not on the surface

UVector3 UTubs::ApproxSurfaceNormal( const UVector3& p ) const
{
	ENorm side;
	UVector3 norm;
	double rho, phi;
	double distZ, distRMin, distRMax, distSPhi, distEPhi, distMin;

	rho = std::sqrt(p.x*p.x + p.y*p.y);

	distRMin = std::fabs(rho - fRMin);
	distRMax = std::fabs(rho - fRMax);
	distZ		= std::fabs(std::fabs(p.z) - fDz);

	if (distRMin < distRMax) // First minimum
	{
		if ( distZ < distRMin )
		{
			distMin = distZ;
			side		= kNZ;
		}
		else
		{
			distMin = distRMin;
			side		= kNRMin	;
		}
	}
	else
	{
		if ( distZ < distRMax )
		{
			distMin = distZ;
			side		= kNZ	;
		}
		else
		{
			distMin = distRMax;
			side		= kNRMax	;
		}
	}	 
	if (!fPhiFullTube	&&	rho ) // Protected against (0,0,z) 
	{
		phi = std::atan2(p.y,p.x);

		if ( phi < 0 )	{ phi += 2*UUtils::kPi; }

		if ( fSPhi < 0 )
		{
			distSPhi = std::fabs(phi - (fSPhi + 2*UUtils::kPi))*rho;
		}
		else
		{
			distSPhi = std::fabs(phi - fSPhi)*rho;
		}
		distEPhi = std::fabs(phi - fSPhi - fDPhi)*rho;

		if (distSPhi < distEPhi) // Find new minimum
		{
			if ( distSPhi < distMin )
			{
				side = kNSPhi;
			}
		}
		else
		{
			if ( distEPhi < distMin )
			{
				side = kNEPhi;
			}
		}
	}		
	switch ( side )
	{
	case kNRMin : // Inner radius
		{											
			norm = UVector3(-p.x/rho, -p.y/rho, 0);
			break;
		}
	case kNRMax : // Outer radius
		{									
			norm = UVector3(p.x/rho, p.y/rho, 0);
			break;
		}
	case kNZ :		// + or - dz
		{															
			if ( p.z > 0 )	{ norm = UVector3(0,0,1); }
			else							{ norm = UVector3(0,0,-1); }
			break;
		}
	case kNSPhi:
		{
			norm = UVector3(std::sin(fSPhi), -std::cos(fSPhi), 0);
			break;
		}
	case kNEPhi:
		{
			norm = UVector3(-std::sin(fSPhi+fDPhi), std::cos(fSPhi+fDPhi), 0);
			break;
		}
	default:			// Should never reach this case ...
		{
			// DumpInfo();
			// UException("UTubs::ApproxSurfaceNormal()",
			//						"GeomSolids1002", JustWarning,
			//						"Undefined side for valid surface normal to solid.");
			break;
		}		
	}								
	return norm;
}

////////////////////////////////////////////////////////////////////
//
//
// Calculate distance to shape from outside, along normalised vector
// - return UUtils::Infinity() if no intersection, or intersection distance <= tolerance
//
// - Compute the intersection with the z planes 
//				- if at valid r, phi, return
//
// -> If point is outer outer radius, compute intersection with rmax
//				- if at valid phi,z return
//
// -> Compute intersection with inner radius, taking largest +ve root
//				- if valid (in z,phi), save intersction
//
//		-> If phi segmented, compute intersections with phi half planes
//				- return smallest of valid phi intersections and
//					inner radius intersection
//
// NOTE:
// - 'if valid' implies tolerant checking of intersection points

double UTubs::DistanceToIn( const UVector3& p, const UVector3& v, double) const
{
	double snxt = UUtils::Infinity();			// snxt = default return value
	double tolORMin2, tolIRMax2;	// 'generous' radii squared
	double tolORMax2, tolIRMin2, tolODz, tolIDz;
	const double dRmax = 100.*fRMax;

	static const double halfCarTolerance = 0.5*VUSolid::Tolerance();
	static const double halfRadTolerance = 0.5*kRadTolerance;

	// Intersection point variables
	//
	double Dist, sd, xi, yi, zi, rho2, inum, iden, cosPsi, Comp;
	double t1, t2, t3, b, c, d;		 // Quadratic solver variables 

	// Calculate tolerant rmin and rmax

	if (fRMin > kRadTolerance)
	{
		tolORMin2 = (fRMin - halfRadTolerance)*(fRMin - halfRadTolerance);
		tolIRMin2 = (fRMin + halfRadTolerance)*(fRMin + halfRadTolerance);
	}
	else
	{
		tolORMin2 = 0.0;
		tolIRMin2 = 0.0;
	}
	tolORMax2 = (fRMax + halfRadTolerance)*(fRMax + halfRadTolerance);
	tolIRMax2 = (fRMax - halfRadTolerance)*(fRMax - halfRadTolerance);

	// Intersection with Z surfaces

	tolIDz = fDz - halfCarTolerance;
	tolODz = fDz + halfCarTolerance;

	if (std::fabs(p.z) >= tolIDz)
	{
		if ( p.z*v.z < 0 )		// at +Z going in -Z or visa versa
		{
			sd = (std::fabs(p.z) - fDz)/std::fabs(v.z);	// Z intersect distance

			if(sd < 0.0)	{ sd = 0.0; }

			xi	 = p.x + sd*v.x;								// Intersection coords
			yi	 = p.y + sd*v.y;
			rho2 = xi*xi + yi*yi;

			// Check validity of intersection

			if ((tolIRMin2 <= rho2) && (rho2 <= tolIRMax2))
			{
				if (!fPhiFullTube && rho2)
				{
					// Psi = angle made with central (average) phi of shape
					//
					inum	 = xi*fCosCPhi + yi*fSinCPhi;
					iden	 = std::sqrt(rho2);
					cosPsi = inum/iden;
					if (cosPsi >= fCosHDPhiIT)	{ return sd; }
				}
				else
				{
					return sd;
				}
			}
		}
		else
		{
			if ( snxt<halfCarTolerance )	{ snxt=0; }
			return snxt;	// On/outside extent, and heading away
			// -> cannot intersect
		}
	}

	// -> Can not intersect z surfaces
	//
	// Intersection with rmax (possible return) and rmin (must also check phi)
	//
	// Intersection point (xi,yi,zi) on line x=p.x+t*v.x etc.
	//
	// Intersects with x^2+y^2=R^2
	//
	// Hence (v.x^2+v.y^2)t^2+ 2t(p.x*v.x+p.y*v.y)+p.x^2+p.y^2-R^2=0
	//						t1								t2								t3

	t1 = 1.0 - v.z*v.z;
	t2 = p.x*v.x + p.y*v.y;
	t3 = p.x*p.x + p.y*p.y;

	if ( t1 > 0 )				// Check not || to z axis
	{
		b = t2/t1;
		c = t3 - fRMax*fRMax;
		if ((t3 >= tolORMax2) && (t2<0))	 // This also handles the tangent case
		{
			// Try outer cylinder intersection
			//					c=(t3-fRMax*fRMax)/t1;

			c /= t1;
			d = b*b - c;

			if (d >= 0)	// If real root
			{
				sd = c/(-b+std::sqrt(d));
				if (sd >= 0)	// If 'forwards'
				{
					if ( sd>dRmax ) // Avoid rounding errors due to precision issues on
					{							 // 64 bits systems. Split long distances and recompute
						double fTerm = sd-std::fmod(sd,dRmax);
						sd = fTerm + DistanceToIn(p+fTerm*v,v);
					} 
					// Check z intersection
					//
					zi = p.z + sd*v.z;
					if (std::fabs(zi)<=tolODz)
					{
						// Z ok. Check phi intersection if reqd
						//
						if (fPhiFullTube)
						{
							return sd;
						}
						else
						{
							xi		 = p.x + sd*v.x;
							yi		 = p.y + sd*v.y;
							cosPsi = (xi*fCosCPhi + yi*fSinCPhi)/fRMax;
							if (cosPsi >= fCosHDPhiIT)	{ return sd; }
						}
					}	//	end if std::fabs(zi)
				}		//	end if (sd>=0)
			}			//	end if (d>=0)
		}				//	end if (r>=fRMax)
		else 
		{
			// Inside outer radius :
			// check not inside, and heading through tubs (-> 0 to in)

			if ((t3 > tolIRMin2) && (t2 < 0) && (std::fabs(p.z) <= tolIDz))
			{
				// Inside both radii, delta r -ve, inside z extent

				if (!fPhiFullTube)
				{
					inum	 = p.x*fCosCPhi + p.y*fSinCPhi;
					iden	 = std::sqrt(t3);
					cosPsi = inum/iden;
					if (cosPsi >= fCosHDPhiIT)
					{
						// In the old version, the small negative tangent for the point
						// on surface was not taken in account, and returning 0.0 ...
						// New version: check the tangent for the point on surface and 
						// if no intersection, return UUtils::Infinity(), if intersection instead
						// return sd.
						//
						c = t3-fRMax*fRMax; 
						if ( c<=0.0 )
						{
							return 0.0;
						}
						else
						{
							c = c/t1;
							d = b*b-c;
							if ( d>=0.0 )
							{
								snxt = c/(-b+std::sqrt(d)); // using safe solution
								// for quadratic equation 
								if ( snxt < halfCarTolerance ) { snxt=0; }
								return snxt;
							}			
							else
							{
								return UUtils::Infinity();
							}
						}
					} 
				}
				else
				{	 
					// In the old version, the small negative tangent for the point
					// on surface was not taken in account, and returning 0.0 ...
					// New version: check the tangent for the point on surface and 
					// if no intersection, return UUtils::Infinity(), if intersection instead
					// return sd.
					//
					c = t3 - fRMax*fRMax; 
					if ( c<=0.0 )
					{
						return 0.0;
					}
					else
					{
						c = c/t1;
						d = b*b-c;
						if ( d>=0.0 )
						{
							snxt= c/(-b+std::sqrt(d)); // using safe solution
							// for quadratic equation 
							if ( snxt < halfCarTolerance ) { snxt=0; }
							return snxt;
						}			
						else
						{
							return UUtils::Infinity();
						}
					}
				} // end if	 (!fPhiFullTube)
			}	 // end if	 (t3>tolIRMin2)
		}		 // end if	 (Inside Outer Radius) 
		if ( fRMin )		// Try inner cylinder intersection
		{
			c = (t3 - fRMin*fRMin)/t1;
			d = b*b - c;
			if ( d >= 0.0 )	// If real root
			{
				// Always want 2nd root - we are outside and know rmax Hit was bad
				// - If on surface of rmin also need farthest root

				sd =( b > 0. )? c/(-b - std::sqrt(d)) : (-b + std::sqrt(d));
				if (sd >= -halfCarTolerance)	// check forwards
				{
					// Check z intersection
					//
					if(sd < 0.0)	{ sd = 0.0; }
					if ( sd>dRmax ) // Avoid rounding errors due to precision issues seen
					{							 // 64 bits systems. Split long distances and recompute
						double fTerm = sd-std::fmod(sd,dRmax);
						sd = fTerm + DistanceToIn(p+fTerm*v,v);
					} 
					zi = p.z + sd*v.z;
					if (std::fabs(zi) <= tolODz)
					{
						// Z ok. Check phi
						//
						if ( fPhiFullTube )
						{
							return sd; 
						}
						else
						{
							xi		 = p.x + sd*v.x;
							yi		 = p.y + sd*v.y;
							cosPsi = (xi*fCosCPhi + yi*fSinCPhi)/fRMin;
							if (cosPsi >= fCosHDPhiIT)
							{
								// Good inner radius isect
								// - but earlier phi isect still possible

								snxt = sd;
							}
						}
					}				//		end if std::fabs(zi)
				}					//		end if (sd>=0)
			}						//		end if (d>=0)
		}							//		end if (fRMin)
	}

	// Phi segment intersection
	//
	// o Tolerant of points inside phi planes by up to VUSolid::Tolerance()*0.5
	//
	// o NOTE: Large duplication of code between sphi & ephi checks
	//				 -> only diffs: sphi -> ephi, Comp -> -Comp and half-plane
	//						intersection check <=0 -> >=0
	//				 -> use some form of loop Construct ?
	//
	if ( !fPhiFullTube )
	{
		// First phi surface (Starting phi)
		//
		Comp		= v.x*fSinSPhi - v.y*fCosSPhi;

		if ( Comp < 0 )	// Component in outwards normal dirn
		{
			Dist = (p.y*fCosSPhi - p.x*fSinSPhi);

			if ( Dist < halfCarTolerance )
			{
				sd = Dist/Comp;

				if (sd < snxt)
				{
					if ( sd < 0 )	{ sd = 0.0; }
					zi = p.z + sd*v.z;
					if ( std::fabs(zi) <= tolODz )
					{
						xi	 = p.x + sd*v.x;
						yi	 = p.y + sd*v.y;
						rho2 = xi*xi + yi*yi;

						if ( ( (rho2 >= tolIRMin2) && (rho2 <= tolIRMax2) )
							|| ( (rho2 >	tolORMin2) && (rho2 <	tolIRMin2)
							&& ( v.y*fCosSPhi - v.x*fSinSPhi >	0 )
							&& ( v.x*fCosSPhi + v.y*fSinSPhi >= 0 )		 )
							|| ( (rho2 > tolIRMax2) && (rho2 < tolORMax2)
							&& (v.y*fCosSPhi - v.x*fSinSPhi > 0)
							&& (v.x*fCosSPhi + v.y*fSinSPhi < 0) )		)
						{
							// z and r intersections good
							// - check intersecting with correct half-plane
							//
							if ((yi*fCosCPhi-xi*fSinCPhi) <= halfCarTolerance) { snxt = sd; }
						}
					}
				}
			}		
		}

		// Second phi surface (Ending phi)

		Comp		= -(v.x*fSinEPhi - v.y*fCosEPhi);

		if (Comp < 0 )	// Component in outwards normal dirn
		{
			Dist = -(p.y*fCosEPhi - p.x*fSinEPhi);

			if ( Dist < halfCarTolerance )
			{
				sd = Dist/Comp;

				if (sd < snxt)
				{
					if ( sd < 0 )	{ sd = 0; }
					zi = p.z + sd*v.z;
					if ( std::fabs(zi) <= tolODz )
					{
						xi	 = p.x + sd*v.x;
						yi	 = p.y + sd*v.y;
						rho2 = xi*xi + yi*yi;
						if ( ( (rho2 >= tolIRMin2) && (rho2 <= tolIRMax2) )
							|| ( (rho2 > tolORMin2)	&& (rho2 < tolIRMin2)
							&& (v.x*fSinEPhi - v.y*fCosEPhi >	0)
							&& (v.x*fCosEPhi + v.y*fSinEPhi >= 0) )
							|| ( (rho2 > tolIRMax2) && (rho2 < tolORMax2)
							&& (v.x*fSinEPhi - v.y*fCosEPhi > 0)
							&& (v.x*fCosEPhi + v.y*fSinEPhi < 0) ) )
						{
							// z and r intersections good
							// - check intersecting with correct half-plane
							//
							if ( (yi*fCosCPhi-xi*fSinCPhi) >= 0 ) { snxt = sd; }
						}												 //?? >=-halfCarTolerance
					}
				}
			}
		}				 //	Comp < 0
	}					 //	!fPhiFullTube 
	if ( snxt<halfCarTolerance )	{ snxt=0; }
	return snxt;
}

//////////////////////////////////////////////////////////////////
//
// Calculate distance to shape from outside, along normalised vector
// - return UUtils::Infinity() if no intersection, or intersection distance <= tolerance
//
// - Compute the intersection with the z planes 
//				- if at valid r, phi, return
//
// -> If point is outer outer radius, compute intersection with rmax
//				- if at valid phi,z return
//
// -> Compute intersection with inner radius, taking largest +ve root
//				- if valid (in z,phi), save intersction
//
//		-> If phi segmented, compute intersections with phi half planes
//				- return smallest of valid phi intersections and
//					inner radius intersection
//
// NOTE:
// - Precalculations for phi trigonometry are Done `just in time'
// - `if valid' implies tolerant checking of intersection points
//	 Calculate distance (<= actual) to closest surface of shape from outside
// - Calculate distance to z, radial planes
// - Only to phi planes if outside phi extent
// - Return 0 if point inside

double UTubs::SafetyFromOutside( const UVector3& p, bool) const
{
	double safe=0.0, rho, safe1, safe2, safe3;
	double safePhi, cosPsi;

	rho	 = std::sqrt(p.x*p.x + p.y*p.y);
	safe1 = fRMin - rho;
	safe2 = rho - fRMax;
	safe3 = std::fabs(p.z) - fDz;

	if ( safe1 > safe2 ) { safe = safe1; }
	else								 { safe = safe2; }
	if ( safe3 > safe )	{ safe = safe3; }

	if ( (!fPhiFullTube) && (rho) )
	{
		// Psi=angle from central phi to point
		//
		cosPsi = (p.x*fCosCPhi + p.y*fSinCPhi)/rho;

		if ( cosPsi < std::cos(fDPhi*0.5) )
		{
			// Point lies outside phi range

			if ( (p.y*fCosCPhi - p.x*fSinCPhi) <= 0 )
			{
				safePhi = std::fabs(p.x*fSinSPhi - p.y*fCosSPhi);
			}
			else
			{
				safePhi = std::fabs(p.x*fSinEPhi - p.y*fCosEPhi);
			}
			if ( safePhi > safe )	{ safe = safePhi; }
		}
	}
	if ( safe < 0 )	{ safe = 0; }
	return safe;
}

//////////////////////////////////////////////////////////////////////////////
//
// Calculate distance to surface of shape from `inside', allowing for tolerance
// - Only Calc rmax intersection if no valid rmin intersection

// double UTubs::DistanceToOut( const UVector3& p, const UVector3& v, const bool calcNorm, bool *validNorm, UVector3 *n		) const
double UTubs::DistanceToOut(const UVector3& p, const UVector3& v, UVector3 &n, bool &validNorm, double) const
{	
	ESide side=kNull , sider=kNull, sidephi=kNull;
	double snxt, srd=UUtils::Infinity(), sphi=UUtils::Infinity(), pdist;
	double deltaR, t1, t2, t3, b, c, d2, roMin2;

	static const double halfCarTolerance = VUSolid::Tolerance()*0.5;
	static const double halfAngTolerance = kAngTolerance*0.5;

	// Vars for phi intersection:

	double pDistS, compS, pDistE, compE, sphi2, xi, yi, vphi, roi2;

	// Z plane intersection

	if (v.z > 0 )
	{
		pdist = fDz - p.z;
		if ( pdist > halfCarTolerance )
		{
			snxt = pdist/v.z;
			side = kPZ;
		}
		else
		{
			//			if (calcNorm)
			{
				n				 = UVector3(0,0,1);
				validNorm = true;
			}
			return snxt = 0;
		}
	}
	else if ( v.z < 0 )
	{
		pdist = fDz + p.z;

		if ( pdist > halfCarTolerance )
		{
			snxt = -pdist/v.z;
			side = kMZ;
		}
		else
		{
			//			if (calcNorm)
			{
				n				 = UVector3(0,0,-1);
				validNorm = true;
			}
			return snxt = 0.0;
		}
	}
	else
	{
		snxt = UUtils::Infinity();		// Travel perpendicular to z axis
		side = kNull;
	}

	// Radial Intersections
	//
	// Find intersection with cylinders at rmax/rmin
	// Intersection point (xi,yi,zi) on line x=p.x+t*v.x etc.
	//
	// Intersects with x^2+y^2=R^2
	//
	// Hence (v.x^2+v.y^2)t^2+ 2t(p.x*v.x+p.y*v.y)+p.x^2+p.y^2-R^2=0
	//
	//						t1								t2										t3

	t1	 = 1.0 - v.z*v.z;			// since v normalised
	t2	 = p.x*v.x + p.y*v.y;
	t3	 = p.x*p.x + p.y*p.y;

	if ( snxt > 10*(fDz+fRMax) )	{ roi2 = 2*fRMax*fRMax; }
	else	{ roi2 = snxt*snxt*t1 + 2*snxt*t2 + t3; }				// radius^2 on +-fDz

	if ( t1 > 0 ) // Check not parallel
	{
		// Calculate srd, r exit distance

		if ( (t2 >= 0.0) && (roi2 > fRMax*(fRMax + kRadTolerance)) )
		{
			// Delta r not negative => leaving via rmax

			deltaR = t3 - fRMax*fRMax;

			// NOTE: Should use rho-fRMax<-kRadTolerance*0.5
			// - avoid sqrt for efficiency

			if ( deltaR < -kRadTolerance*fRMax )
			{
				b		 = t2/t1;
				c		 = deltaR/t1;
				d2		= b*b-c;
				if( d2 >= 0 ) { srd = c/( -b - std::sqrt(d2)); }
				else					{ srd = 0.; }
				sider = kRMax;
			}
			else
			{
				// On tolerant boundary & heading outwards (or perpendicular to)
				// outer radial surface -> leaving immediately

				//				if ( calcNorm ) 
				{
					n				 = UVector3(p.x/fRMax,p.y/fRMax,0);
					validNorm = true;
				}
				return snxt = 0; // Leaving by rmax immediately
			}
		}						 
		else if ( t2 < 0. ) // i.e.	t2 < 0; Possible rmin intersection
		{
			roMin2 = t3 - t2*t2/t1; // min ro2 of the plane of movement 

			if ( fRMin && (roMin2 < fRMin*(fRMin - kRadTolerance)) )
			{
				deltaR = t3 - fRMin*fRMin;
				b			= t2/t1;
				c			= deltaR/t1;
				d2		 = b*b - c;

				if ( d2 >= 0 )	 // Leaving via rmin
				{
					// NOTE: SHould use rho-rmin>kRadTolerance*0.5
					// - avoid sqrt for efficiency

					if (deltaR > kRadTolerance*fRMin)
					{
						srd = c/(-b+std::sqrt(d2)); 
						sider = kRMin;
					}
					else
					{
						//if ( calcNorm ) 
						{ validNorm = false; }	// Concave side
						return snxt = 0.0;
					}
				}
				else		// No rmin intersect -> must be rmax intersect
				{
					deltaR = t3 - fRMax*fRMax;
					c		 = deltaR/t1;
					d2		= b*b-c;
					if( d2 >=0. )
					{
						srd		 = -b + std::sqrt(d2);
						sider	= kRMax;
					}
					else // Case: On the border+t2<kRadTolerance
						//			 (v is perpendicular to the surface)
					{
						// if (calcNorm)
						{
							n = UVector3(p.x/fRMax,p.y/fRMax,0);
							validNorm = true;
						}
						return snxt = 0.0;
					}
				}
			}
			else if ( roi2 > fRMax*(fRMax + kRadTolerance) )
				// No rmin intersect -> must be rmax intersect
			{
				deltaR = t3 - fRMax*fRMax;
				b			= t2/t1;
				c			= deltaR/t1;
				d2		 = b*b-c;
				if( d2 >= 0 )
				{
					srd		 = -b + std::sqrt(d2);
					sider	= kRMax;
				}
				else // Case: On the border+t2<kRadTolerance
					//			 (v is perpendicular to the surface)
				{
					//					if (calcNorm)
					{
						n = UVector3(p.x/fRMax,p.y/fRMax,0);
						validNorm = true;
					}
					return snxt = 0.0;
				}
			}
		}

		// Phi Intersection

		if ( !fPhiFullTube )
		{
			// add angle calculation with correction 
			// of the difference in domain of atan2 and Sphi
			//
			vphi = std::atan2(v.y,v.x);

			if ( vphi < fSPhi - halfAngTolerance	)						 { vphi += 2*UUtils::kPi; }
			else if ( vphi > fSPhi + fDPhi + halfAngTolerance ) { vphi -= 2*UUtils::kPi; }


			if ( p.x || p.y )	// Check if on z axis (rho not needed later)
			{
				// pDist -ve when inside

				pDistS = p.x*fSinSPhi - p.y*fCosSPhi;
				pDistE = -p.x*fSinEPhi + p.y*fCosEPhi;

				// Comp -ve when in direction of outwards normal

				compS	 = -fSinSPhi*v.x + fCosSPhi*v.y;
				compE	 =	fSinEPhi*v.x - fCosEPhi*v.y;

				sidephi = kNull;

				if( ( (fDPhi <= UUtils::kPi) && ( (pDistS <= halfCarTolerance)
					&& (pDistE <= halfCarTolerance) ) )
					|| ( (fDPhi >	UUtils::kPi) && !((pDistS >	halfCarTolerance)
					&& (pDistE >	halfCarTolerance) ) )	)
				{
					// Inside both phi *full* planes

					if ( compS < 0 )
					{
						sphi = pDistS/compS;

						if (sphi >= -halfCarTolerance)
						{
							xi = p.x + sphi*v.x;
							yi = p.y + sphi*v.y;

							// Check intersecting with correct half-plane
							// (if not -> no intersect)
							//
							if( (std::fabs(xi)<=VUSolid::Tolerance())&&(std::fabs(yi)<=VUSolid::Tolerance()) )
							{
								sidephi = kSPhi;
								if (((fSPhi-halfAngTolerance)<=vphi)
									&&((fSPhi+fDPhi+halfAngTolerance)>=vphi))
								{
									sphi = UUtils::Infinity();
								}
							}
							else if ( yi*fCosCPhi-xi*fSinCPhi >=0 )
							{
								sphi = UUtils::Infinity();
							}
							else
							{
								sidephi = kSPhi;
								if ( pDistS > -halfCarTolerance )
								{
									sphi = 0.0; // Leave by sphi immediately
								}		
							}			 
						}
						else
						{
							sphi = UUtils::Infinity();
						}
					}
					else
					{
						sphi = UUtils::Infinity();
					}

					if ( compE < 0 )
					{
						sphi2 = pDistE/compE;

						// Only check further if < starting phi intersection
						//
						if ( (sphi2 > -halfCarTolerance) && (sphi2 < sphi) )
						{
							xi = p.x + sphi2*v.x;
							yi = p.y + sphi2*v.y;

							if ((std::fabs(xi)<=VUSolid::Tolerance())&&(std::fabs(yi)<=VUSolid::Tolerance()))
							{
								// Leaving via ending phi
								//
								if( !((fSPhi-halfAngTolerance <= vphi)
									&&(fSPhi+fDPhi+halfAngTolerance >= vphi)) )
								{
									sidephi = kEPhi;
									if ( pDistE <= -halfCarTolerance )	{ sphi = sphi2; }
									else																{ sphi = 0.0;	 }
								}
							} 
							else		// Check intersecting with correct half-plane 

								if ( (yi*fCosCPhi-xi*fSinCPhi) >= 0)
								{
									// Leaving via ending phi
									//
									sidephi = kEPhi;
									if ( pDistE <= -halfCarTolerance ) { sphi = sphi2; }
									else															 { sphi = 0.0;	 }
								}
						}
					}
				}
				else
				{
					sphi = UUtils::Infinity();
				}
			}
			else
			{
				// On z axis + travel not || to z axis -> if phi of vector direction
				// within phi of shape, Step limited by rmax, else Step =0

				if ( (fSPhi - halfAngTolerance <= vphi)
					&& (vphi <= fSPhi + fDPhi + halfAngTolerance ) )
				{
					sphi = UUtils::Infinity();
				}
				else
				{
					sidephi = kSPhi; // arbitrary 
					sphi		= 0.0;
				}
			}
			if (sphi < snxt)	// Order intersecttions
			{
				snxt = sphi;
				side = sidephi;
			}
		}
		if (srd < snxt)	// Order intersections
		{
			snxt = srd;
			side = sider;
		}
	}
	//	if (calcNorm)
	{
		switch(side)
		{
		case kRMax:
			// Note: returned vector not normalised
			// (divide by fRMax for Unit vector)
			//
			xi = p.x + snxt*v.x;
			yi = p.y + snxt*v.y;
			n = UVector3(xi/fRMax,yi/fRMax,0);
			validNorm = true;
			break;

		case kRMin:
			validNorm = false;	// Rmin is inconvex
			break;

		case kSPhi:
			if ( fDPhi <= UUtils::kPi )
			{
				n				 = UVector3(fSinSPhi,-fCosSPhi,0);
				validNorm = true;
			}
			else
			{
				validNorm = false;
			}
			break;

		case kEPhi:
			if (fDPhi <= UUtils::kPi)
			{
				n = UVector3(-fSinEPhi,fCosEPhi,0);
				validNorm = true;
			}
			else
			{
				validNorm = false;
			}
			break;

		case kPZ:
			n				 = UVector3(0,0,1);
			validNorm = true;
			break;

		case kMZ:
			n				 = UVector3(0,0,-1);
			validNorm = true;
			break;

		default:
			cout << std::endl;
			// DumpInfo();
			std::ostringstream message;
			int oldprc = message.precision(16);
			message << "Undefined side for valid surface normal to solid."
				<< std::endl
				<< "Position:"	<< std::endl << std::endl
				<< "p.x = "	 << p.x << " mm" << std::endl
				<< "p.y = "	 << p.y << " mm" << std::endl
				<< "p.z = "	 << p.z << " mm" << std::endl << std::endl
				<< "Direction:" << std::endl << std::endl
				<< "v.x = "	 << v.x << std::endl
				<< "v.y = "	 << v.y << std::endl
				<< "v.z = "	 << v.z << std::endl << std::endl
				<< "Proposed distance :" << std::endl << std::endl
				<< "snxt = "		<< snxt << " mm" << std::endl;
			message.precision(oldprc);
			// UException("UTubs::DistanceToOut(p,v,..)", "GeomSolids1002",
			//						JustWarning, message);
			break;
		}
	}
	if ( snxt<halfCarTolerance )	{ snxt=0; }

	return snxt;
}

//////////////////////////////////////////////////////////////////////////
//
// Calculate distance (<=actual) to closest surface of shape from inside

double UTubs::SafetyFromInside( const UVector3& p, bool) const
{
	double safe=0.0, rho, safeR1, safeR2, safeZ, safePhi;
	rho = std::sqrt(p.x*p.x + p.y*p.y);

#ifdef UCSGDEBUG
	if( Inside(p) == eOutside )
	{
		int oldprc = cout.precision(16);
		cout << std::endl;
		DumpInfo();
		cout << "Position:"	<< std::endl << std::endl;
		cout << "p.x = "	 << p.x << " mm" << std::endl;
		cout << "p.y = "	 << p.y << " mm" << std::endl;
		cout << "p.z = "	 << p.z << " mm" << std::endl << std::endl;
		cout.precision(oldprc);
		// UException("UTubs::DistanceToOut(p)", "GeomSolids1002",
		JustWarning, "Point p is outside !?");
	}
#endif

	if ( fRMin )
	{
		safeR1 = rho	 - fRMin;
		safeR2 = fRMax - rho;

		if ( safeR1 < safeR2 ) { safe = safeR1; }
		else									 { safe = safeR2; }
	}
	else
	{
		safe = fRMax - rho;
	}
	safeZ = fDz - std::fabs(p.z);

	if ( safeZ < safe )	{ safe = safeZ; }

	// Check if phi divided, Calc distances closest phi plane
	//
	if ( !fPhiFullTube )
	{
		if ( p.y*fCosCPhi-p.x*fSinCPhi <= 0 )
		{
			safePhi = -(p.x*fSinCPhi - p.y*fCosSPhi);
		}
		else
		{
			safePhi = (p.x*fSinEPhi - p.y*fCosEPhi);
		}
		if (safePhi < safe)	{ safe = safePhi; }
	}
	if ( safe < 0 )	{ safe = 0; }

	return safe;	
}

/////////////////////////////////////////////////////////////////////////
//
// Create a List containing the transformed vertices
// Ordering [0-3] -fDz Cross section
//					[4-7] +fDz Cross section such that [0] is below [4],
//																						 [1] below [5] etc.
// Note:
//	Caller has deletion resposibility
//	Potential improvement: For last slice, use actual ending angle
//												 to avoid rounding error problems.

/*
UVector3List*
UTubs::CreateRotatedVertices( const UAffineTransform& pTransform ) const
{
UVector3List* vertices;
UVector3 vertex0, vertex1, vertex2, vertex3;
double meshAngle, meshRMax, crossAngle,
cosCrossAngle, sinCrossAngle, sAngle;
double rMaxX, rMaxY, rMinX, rMinY, meshRMin;
int crossSection, noCrossSections;

// Compute no of Cross-sections necessary to mesh tube
//
noCrossSections = int(fDPhi/kMeshAngleDefault) + 1;

if ( noCrossSections < kMinMeshSections )
{
noCrossSections = kMinMeshSections;
}
else if (noCrossSections>kMaxMeshSections)
{
noCrossSections = kMaxMeshSections;
}
// noCrossSections = 4;

meshAngle = fDPhi/(noCrossSections - 1);
// meshAngle = fDPhi/(noCrossSections);

meshRMax	= (fRMax+100*VUSolid::Tolerance())/std::cos(meshAngle*0.5);
meshRMin = fRMin - 100*VUSolid::Tolerance(); 

// If complete in phi, Set start angle such that mesh will be at fRMax
// on the x axis. Will give better extent calculations when not rotated.

if (fPhiFullTube && (fSPhi == 0) )	{ sAngle = -meshAngle*0.5; }
else																{ sAngle =	fSPhi; }

vertices = new UVector3List();

if ( vertices )
{
vertices->reserve(noCrossSections*4);
for (crossSection = 0; crossSection < noCrossSections; crossSection++ )
{
// Compute coordinates of Cross section at section crossSection

crossAngle		= sAngle + crossSection*meshAngle;
cosCrossAngle = std::cos(crossAngle);
sinCrossAngle = std::sin(crossAngle);

rMaxX = meshRMax*cosCrossAngle;
rMaxY = meshRMax*sinCrossAngle;

if(meshRMin <= 0.0)
{
rMinX = 0.0;
rMinY = 0.0;
}
else
{
rMinX = meshRMin*cosCrossAngle;
rMinY = meshRMin*sinCrossAngle;
}
vertex0 = UVector3(rMinX,rMinY,-fDz);
vertex1 = UVector3(rMaxX,rMaxY,-fDz);
vertex2 = UVector3(rMaxX,rMaxY,+fDz);
vertex3 = UVector3(rMinX,rMinY,+fDz);

vertices->push_back(pTransform.TransformPoint(vertex0));
vertices->push_back(pTransform.TransformPoint(vertex1));
vertices->push_back(pTransform.TransformPoint(vertex2));
vertices->push_back(pTransform.TransformPoint(vertex3));
}
}
else
{
// DumpInfo();
// UException("UTubs::CreateRotatedVertices()",
//						"GeomSolids0003", FatalException,
//						"Error in allocation of vertices. Out of memory !");
}
return vertices;
}
*/

//////////////////////////////////////////////////////////////////////////
//
// Stream object contents to an output stream

UGeometryType UTubs::GetEntityType() const
{
	return std::string("UTubs");
}

//////////////////////////////////////////////////////////////////////////
//
// Make a clone of the object
//
VUSolid* UTubs::Clone() const
{
	return new UTubs(*this);
}

//////////////////////////////////////////////////////////////////////////
//
// Stream object contents to an output stream

std::ostream& UTubs::StreamInfo( std::ostream& os ) const
{
	int oldprc = os.precision(16);
	os << "-----------------------------------------------------------\n"
		<< "		*** Dump for solid - " << GetName() << " ***\n"
		<< "		===================================================\n"
		<< " Solid type: UTubs\n"
		<< " Parameters: \n"
		<< "		inner radius : " << fRMin << " mm \n"
		<< "		outer radius : " << fRMax << " mm \n"
		<< "		half length Z: " << fDz << " mm \n"
		<< "		starting phi : " << fSPhi/(UUtils::kPi/180.0) << " degrees \n"
		<< "		delta phi		: " << fDPhi/(UUtils::kPi/180.0) << " degrees \n"
		<< "-----------------------------------------------------------\n";
	os.precision(oldprc);

	return os;
}

/////////////////////////////////////////////////////////////////////////
//
// GetPointOnSurface

UVector3 UTubs::GetPointOnSurface() const
{
	double xRand, yRand, zRand, phi, cosphi, sinphi, chose,
		aOne, aTwo, aThr, aFou;
	double rRand;

	aOne = 2.*fDz*fDPhi*fRMax;
	aTwo = 2.*fDz*fDPhi*fRMin;
	aThr = 0.5*fDPhi*(fRMax*fRMax-fRMin*fRMin);
	aFou = 2.*fDz*(fRMax-fRMin);

	phi		= UUtils::Random(fSPhi, fSPhi+fDPhi);
	cosphi = std::cos(phi);
	sinphi = std::sin(phi);

	rRand	= UUtils::GetRadiusInRing(fRMin,fRMax);

	if( (fSPhi == 0) && (fDPhi == 2*UUtils::kPi) ) { aFou = 0; }

	chose	= UUtils::Random(0.,aOne+aTwo+2.*aThr+2.*aFou);

	if( (chose >=0) && (chose < aOne) )
	{
		xRand = fRMax*cosphi;
		yRand = fRMax*sinphi;
		zRand = UUtils::Random(-1.*fDz,fDz);
		return UVector3	(xRand, yRand, zRand);
	}
	else if( (chose >= aOne) && (chose < aOne + aTwo) )
	{
		xRand = fRMin*cosphi;
		yRand = fRMin*sinphi;
		zRand = UUtils::Random(-1.*fDz,fDz);
		return UVector3	(xRand, yRand, zRand);
	}
	else if( (chose >= aOne + aTwo) && (chose < aOne + aTwo + aThr) )
	{
		xRand = rRand*cosphi;
		yRand = rRand*sinphi;
		zRand = fDz;
		return UVector3	(xRand, yRand, zRand);
	}
	else if( (chose >= aOne + aTwo + aThr) && (chose < aOne + aTwo + 2.*aThr) )
	{
		xRand = rRand*cosphi;
		yRand = rRand*sinphi;
		zRand = -1.*fDz;
		return UVector3	(xRand, yRand, zRand);
	}
	else if( (chose >= aOne + aTwo + 2.*aThr)
		&& (chose < aOne + aTwo + 2.*aThr + aFou) )
	{
		xRand = rRand*std::cos(fSPhi);
		yRand = rRand*std::sin(fSPhi);
		zRand = UUtils::Random(-1.*fDz,fDz);
		return UVector3	(xRand, yRand, zRand);
	}
	else
	{
		xRand = rRand*std::cos(fSPhi+fDPhi);
		yRand = rRand*std::sin(fSPhi+fDPhi);
		zRand = UUtils::Random(-1.*fDz,fDz);
		return UVector3	(xRand, yRand, zRand);
	}
}

///////////////////////////////////////////////////////////////////////////
//
// Methods for visualisation

/*
void UTubs::DescribeYourselfTo ( UVGraphicsScene& scene ) const 
{
scene.AddSolid (*this);
}
*/

UPolyhedron* UTubs::CreatePolyhedron () const 
{
	return new UPolyhedronTubs (fRMin, fRMax, fDz, fSPhi, fDPhi);
}

void UTubs::Extent (UVector3 &aMin, UVector3 &aMax) const
{
	aMin = UVector3(-fRMax, -fRMax, -fDz);
	aMax = UVector3(fRMax, fRMax, fDz);
}
