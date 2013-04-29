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
// $Id: UPolycone.cc 66241 2012-12-13 18:34:42Z gunter $
//
// 
// --------------------------------------------------------------------
// GEANT 4 class source file
//
//
// UPolycone.cc
//
// Implementation of a CSG polycone
//
// --------------------------------------------------------------------

#include "UUtils.hh"
#include <string>
#include <cmath>
#include <sstream>
#include "UPolycone.hh"

#include "UPolyconeSide.hh"
#include "UPolyPhiFace.hh"

 


#include "UEnclosingCylinder.hh"
#include "UReduciblePolygon.hh"


using namespace std;

//
// Constructor (GEANT3 style parameters)
//	
UPolycone::UPolycone( const std::string& name, 
															double phiStart,
															double phiTotal,
															int numZPlanes,
												const double zPlane[],
												const double rInner[],
												const double rOuter[]	)
	: UVCSGfaceted( name ), genericPcon(false)
{
	//
	// Some historical ugliness
	//
	original_parameters = new UPolyconeHistorical();
	
	original_parameters->Start_angle = phiStart;
	original_parameters->Opening_angle = phiTotal;
	original_parameters->Num_z_planes = numZPlanes;
	original_parameters->Z_values = new double[numZPlanes];
	original_parameters->Rmin = new double[numZPlanes];
	original_parameters->Rmax = new double[numZPlanes];

	int i;
	for (i=0; i<numZPlanes; i++)
	{
		if (( i < numZPlanes-1) && ( zPlane[i] == zPlane[i+1] ))
		{
			if( (rInner[i]	 > rOuter[i+1])
				||(rInner[i+1] > rOuter[i])	 )
			{
				
				std::ostringstream message;
				message << "Cannot create a Polycone with no contiguous segments."
								<< std::endl
								<< "				Segments are not contiguous !" << std::endl
								<< "				rMin[" << i << "] = " << rInner[i]
								<< " -- rMax[" << i+1 << "] = " << rOuter[i+1] << std::endl
								<< "				rMin[" << i+1 << "] = " << rInner[i+1]
								<< " -- rMax[" << i << "] = " << rOuter[i];
				// UException("UPolycone::UPolycone()", "GeomSolids0002",
				//						FatalErrorInArgument, message);
			}
		} 
		original_parameters->Z_values[i] = zPlane[i];
		original_parameters->Rmin[i] = rInner[i];
		original_parameters->Rmax[i] = rOuter[i];
	}

	//
	// Build RZ polygon using special PCON/PGON GEANT3 constructor
	//
	UReduciblePolygon *rz =
		new UReduciblePolygon( rInner, rOuter, zPlane, numZPlanes );
	
	//
	// Do the real work
	//
	Create( phiStart, phiTotal, rz );
	
	delete rz;
}


//
// Constructor (generic parameters)
//
UPolycone::UPolycone( const std::string& name, 
															double phiStart,
															double phiTotal,
															int		numRZ,
												const double r[],
												const double z[]	 )
	: UVCSGfaceted( name ), genericPcon(true)
{
	UReduciblePolygon *rz = new UReduciblePolygon( r, z, numRZ );
	
	Create( phiStart, phiTotal, rz );
	
	// Set original_parameters struct for consistency
	//
	SetOriginalParameters();
	
	delete rz;
}


//
// Create
//
// Generic create routine, called by each constructor after
// conversion of arguments
//
void UPolycone::Create( double phiStart,
												 double phiTotal,
												 UReduciblePolygon *rz		)
{
	//
	// Perform checks of rz values
	//
	if (rz->Amin() < 0.0)
	{
		std::ostringstream message;
		message << "Illegal input parameters - " << GetName() << std::endl
						<< "				All R values must be >= 0 !";
		// UException("UPolycone::Create()", "GeomSolids0002",
		//						FatalErrorInArgument, message);
	}
		
	double rzArea = rz->Area();
	if (rzArea < -VUSolid::Tolerance())
		rz->ReverseOrder();

	else if (rzArea < -VUSolid::Tolerance())
	{
		std::ostringstream message;
		message << "Illegal input parameters - " << GetName() << std::endl
						<< "				R/Z Cross section is zero or near zero: " << rzArea;
		// UException("UPolycone::Create()", "GeomSolids0002",
		//						FatalErrorInArgument, message);
	}
		
	if ( (!rz->RemoveDuplicateVertices( VUSolid::Tolerance() ))
		|| (!rz->RemoveRedundantVertices( VUSolid::Tolerance() ))		 ) 
	{
		std::ostringstream message;
		message << "Illegal input parameters - " << GetName() << std::endl
						<< "				Too few unique R/Z values !";
		// UException("UPolycone::Create()", "GeomSolids0002",
		//						FatalErrorInArgument, message);
	}

	if (rz->CrossesItself(1/UUtils::Infinity())) 
	{
		std::ostringstream message;
		message << "Illegal input parameters - " << GetName() << std::endl
						<< "				R/Z segments Cross !";
		// UException("UPolycone::Create()", "GeomSolids0002",
		//						FatalErrorInArgument, message);
	}

	numCorner = rz->NumVertices();

	//
	// Phi opening? Account for some possible roundoff, and interpret
	// nonsense value as representing no phi opening
	//
	if (phiTotal <= 0 || phiTotal > 2*UUtils::kPi-1E-10)
	{
		phiIsOpen = false;
		startPhi = 0;
		endPhi = 2*UUtils::kPi;
	}
	else
	{
		phiIsOpen = true;
		
		//
		// Convert phi into our convention
		//
		startPhi = phiStart;
		while( startPhi < 0 ) startPhi += 2*UUtils::kPi;
		
		endPhi = phiStart+phiTotal;
		while( endPhi < startPhi ) endPhi += 2*UUtils::kPi;
	}
	
	//
	// Allocate corner array. 
	//
	corners = new UPolyconeSideRZ[numCorner];

	//
	// Copy corners
	//
	UReduciblePolygonIterator iterRZ(rz);
	
	UPolyconeSideRZ *next = corners;
	iterRZ.Begin();
	do
	{
		next->r = iterRZ.GetA();
		next->z = iterRZ.GetB();
	} while( ++next, iterRZ.Next() );
	
	//
	// Allocate face pointer array
	//
	numFace = phiIsOpen ? numCorner+2 : numCorner;
	faces = new UVCSGface*[numFace];
	
	//
	// Construct conical faces
	//
	// But! Don't construct a face if both points are at zero radius!
	//
	UPolyconeSideRZ *corner = corners,
									 *prev = corners + numCorner-1,
									 *nextNext;
	UVCSGface	**face = faces;
	do
	{
		next = corner+1;
		if (next >= corners+numCorner) next = corners;
		nextNext = next+1;
		if (nextNext >= corners+numCorner) nextNext = corners;
		
		if (corner->r < 1/UUtils::Infinity() && next->r < 1/UUtils::Infinity()) continue;
		
		//
		// We must decide here if we can dare declare one of our faces
		// as having a "valid" normal (i.e. allBehind = true). This
		// is never possible if the face faces "inward" in r.
		//
		bool allBehind;
		if (corner->z > next->z)
		{
			allBehind = false;
		}
		else
		{
			//
			// Otherwise, it is only true if the line passing
			// through the two points of the segment do not
			// split the r/z Cross section
			//
			allBehind = !rz->BisectedBy( corner->r, corner->z,
								 next->r, next->z, VUSolid::Tolerance() );
		}
		
		*face++ = new UPolyconeSide( prev, corner, next, nextNext,
								startPhi, endPhi-startPhi, phiIsOpen, allBehind );
	} while( prev=corner, corner=next, corner > corners );
	
	if (phiIsOpen)
	{
		//
		// Construct phi open edges
		//
		*face++ = new UPolyPhiFace( rz, startPhi, 0, endPhi	);
		*face++ = new UPolyPhiFace( rz, endPhi,	 0, startPhi );
	}
	
	//
	// We might have dropped a face or two: recalculate numFace
	//
	numFace = face-faces;
	
	//
	// Make enclosingCylinder
	//
	enclosingCylinder =
		new UEnclosingCylinder( rz, phiIsOpen, phiStart, phiTotal );
}


//
// Fake default constructor - sets only member data and allocates memory
//														for usage restricted to object persistency.
//

/*
UPolycone::UPolycone( __void__& a )
	: UVCSGfaceted(a), startPhi(0.),	endPhi(0.), phiIsOpen(false),
		genericPcon(false), numCorner(0), corners(0),
		original_parameters(0), enclosingCylinder(0)
{
}
*/


//
// Destructor
//
UPolycone::~UPolycone()
{
	delete [] corners;
	delete original_parameters;
	delete enclosingCylinder;
}


//
// Copy constructor
//
UPolycone::UPolycone( const UPolycone &source )
	: UVCSGfaceted( source )
{
	CopyStuff( source );
}


//
// Assignment operator
//
const UPolycone &UPolycone::operator=( const UPolycone &source )
{
	if (this == &source) return *this;
	
	UVCSGfaceted::operator=( source );
	
	delete [] corners;
	if (original_parameters) delete original_parameters;
	
	delete enclosingCylinder;
	
	CopyStuff( source );
	
	return *this;
}


//
// CopyStuff
//
void UPolycone::CopyStuff( const UPolycone &source )
{
	//
	// Simple stuff
	//
	startPhi	= source.startPhi;
	endPhi		= source.endPhi;
	phiIsOpen	= source.phiIsOpen;
	numCorner	= source.numCorner;
	genericPcon= source.genericPcon;

	//
	// The corner array
	//
	corners = new UPolyconeSideRZ[numCorner];
	
	UPolyconeSideRZ	*corn = corners,
				*sourceCorn = source.corners;
	do
	{
		*corn = *sourceCorn;
	} while( ++sourceCorn, ++corn < corners+numCorner );
	
	//
	// Original parameters
	//
	if (source.original_parameters)
	{
		original_parameters =
			new UPolyconeHistorical( *source.original_parameters );
	}
	
	//
	// Enclosing cylinder
	//
	enclosingCylinder = new UEnclosingCylinder( *source.enclosingCylinder );
}


//
// Reset
//
bool UPolycone::Reset()
{
	if (genericPcon)
	{
		std::ostringstream message;
		message << "Solid " << GetName() << " built using generic construct."
						<< std::endl << "Not applicable to the generic construct !";
		// UException("UPolycone::Reset(,,)", "GeomSolids1001",
		//						JustWarning, message, "Parameters NOT resetted.");
		return 1;
	}

	//
	// Clear old setup
	//
	UVCSGfaceted::DeleteStuff();
	delete [] corners;
	delete enclosingCylinder;

	//
	// Rebuild polycone
	//
	UReduciblePolygon *rz =
		new UReduciblePolygon( original_parameters->Rmin,
														original_parameters->Rmax,
														original_parameters->Z_values,
														original_parameters->Num_z_planes );
	Create( original_parameters->Start_angle,
					original_parameters->Opening_angle, rz );
	delete rz;

	return 0;
}


//
// Inside
//
// This is an override of UVCSGfaceted::Inside, created in order
// to speed things up by first checking with UEnclosingCylinder.
//
VUSolid::EnumInside UPolycone::Inside( const UVector3 &p ) const
{
	//
	// Quick test
	//
	if (enclosingCylinder->MustBeOutside(p)) return eOutside;

	//
	// Long answer
	//
	return UVCSGfaceted::Inside(p);
}


//
// DistanceToIn
//
// This is an override of UVCSGfaceted::Inside, created in order
// to speed things up by first checking with UEnclosingCylinder.
//
double UPolycone::DistanceToIn( const UVector3 &p,
																	 const UVector3 &v ) const
{
	//
	// Quick test
	//
	if (enclosingCylinder->ShouldMiss(p,v))
		return UUtils::Infinity();
	
	//
	// Long answer
	//
	return UVCSGfaceted::DistanceToIn( p, v );
}


//
// DistanceToIn
//
double UPolycone::SafetyFromOutside( const UVector3 &p, bool aAccurate) const
{
	return UVCSGfaceted::SafetyFromOutside(p);
}


/*
//
// ComputeDimensions
//
void UPolycone::ComputeDimensions(			 UVPVParameterisation* p,
																		const int n,
																		const UVPhysicalVolume* pRep )
{
	p->ComputeDimensions(*this,n,pRep);
}
*/

//
// GetEntityType
//
UGeometryType	UPolycone::GetEntityType() const
{
	return std::string("UPolycone");
}

//
// Make a clone of the object
//
VUSolid* UPolycone::Clone() const
{
	return new UPolycone(*this);
}

//
// Stream object contents to an output stream
//
std::ostream& UPolycone::StreamInfo( std::ostream& os ) const
{
	int oldprc = os.precision(16);
	os << "-----------------------------------------------------------\n"
		 << "		*** Dump for solid - " << GetName() << " ***\n"
		 << "		===================================================\n"
		 << " Solid type: UPolycone\n"
		 << " Parameters: \n"
		 << "		starting phi angle : " << startPhi/(UUtils::kPi/180.0) << " degrees \n"
		 << "		ending phi angle	 : " << endPhi/(UUtils::kPi/180.0) << " degrees \n";
	int i=0;
	if (!genericPcon)
	{
		int numPlanes = original_parameters->Num_z_planes;
		os << "		number of Z planes: " << numPlanes << "\n"
			 << "							Z values: \n";
		for (i=0; i<numPlanes; i++)
		{
			os << "							Z plane " << i << ": "
				 << original_parameters->Z_values[i] << "\n";
		}
		os << "							Tangent distances to inner surface (Rmin): \n";
		for (i=0; i<numPlanes; i++)
		{
			os << "							Z plane " << i << ": "
				 << original_parameters->Rmin[i] << "\n";
		}
		os << "							Tangent distances to outer surface (Rmax): \n";
		for (i=0; i<numPlanes; i++)
		{
			os << "							Z plane " << i << ": "
				 << original_parameters->Rmax[i] << "\n";
		}
	}
	os << "		number of RZ points: " << numCorner << "\n"
		 << "							RZ values (corners): \n";
		 for (i=0; i<numCorner; i++)
		 {
			 os << "												 "
					<< corners[i].r << ", " << corners[i].z << "\n";
		 }
	os << "-----------------------------------------------------------\n";
	os.precision(oldprc);

	return os;
}


//
// GetPointOnCone
//
// Auxiliary method for Get Point On Surface
//
UVector3 UPolycone::GetPointOnCone(double fRmin1, double fRmax1,
																				 double fRmin2, double fRmax2,
																				 double zOne,	 double zTwo,
																				 double& totArea) const
{ 
	// declare working variables
	//
	double Aone, Atwo, Afive, phi, zRand, fDPhi, cosu, sinu;
	double rRand1, rmin, rmax, chose, rone, rtwo, qone, qtwo,
					 fDz = std::fabs((zTwo-zOne)/2.);
	UVector3 point, offset;
	offset = UVector3(0.,0.,0.5*(zTwo+zOne));
	fDPhi = endPhi - startPhi;
	rone = (fRmax1-fRmax2)/(2.*fDz); 
	rtwo = (fRmin1-fRmin2)/(2.*fDz);
	if(fRmax1==fRmax2){qone=0.;}
	else{
		qone = fDz*(fRmax1+fRmax2)/(fRmax1-fRmax2);
	}
	if(fRmin1==fRmin2){qtwo=0.;}
	else{
		qtwo = fDz*(fRmin1+fRmin2)/(fRmin1-fRmin2);
	 }
	Aone	 = 0.5*fDPhi*(fRmax2 + fRmax1)*(UUtils::sqr(fRmin1-fRmin2)+UUtils::sqr(zTwo-zOne));			 
	Atwo	 = 0.5*fDPhi*(fRmin2 + fRmin1)*(UUtils::sqr(fRmax1-fRmax2)+UUtils::sqr(zTwo-zOne));
	Afive	= fDz*(fRmax1-fRmin1+fRmax2-fRmin2);
	totArea = Aone+Atwo+2.*Afive;
	
	phi	= UUtils::Random(startPhi,endPhi);
	cosu = std::cos(phi);
	sinu = std::sin(phi);


	if( (startPhi == 0) && (endPhi == 2*UUtils::kPi) ) { Afive = 0; }
	chose = UUtils::Random(0.,Aone+Atwo+2.*Afive);
	if( (chose >= 0) && (chose < Aone) )
	{
		if(fRmax1 != fRmax2)
		{
			zRand = UUtils::Random(-1.*fDz,fDz); 
			point = UVector3 (rone*cosu*(qone-zRand),
														 rone*sinu*(qone-zRand), zRand);
			
		 
		}
		else
		{
			point = UVector3(fRmax1*cosu, fRmax1*sinu,
														UUtils::Random(-1.*fDz,fDz));
		 
		}
	}
	else if(chose >= Aone && chose < Aone + Atwo)
	{
		if(fRmin1 != fRmin2)
			{ 
			zRand = UUtils::Random(-1.*fDz,fDz); 
			point = UVector3 (rtwo*cosu*(qtwo-zRand),
														 rtwo*sinu*(qtwo-zRand), zRand);
			
		}
		else
		{
			point = UVector3(fRmin1*cosu, fRmin1*sinu,
														UUtils::Random(-1.*fDz,fDz));
		 
		}
	}
	else if( (chose >= Aone + Atwo + Afive) && (chose < Aone + Atwo + 2.*Afive) )
	{
		zRand	= UUtils::Random(-1.*fDz,fDz);
		rmin	 = fRmin2-((zRand-fDz)/(2.*fDz))*(fRmin1-fRmin2);
		rmax	 = fRmax2-((zRand-fDz)/(2.*fDz))*(fRmax1-fRmax2);
		rRand1 = std::sqrt(UUtils::Random()*(UUtils::sqr(rmax)-UUtils::sqr(rmin))+UUtils::sqr(rmin));
		point =	UVector3 (rRand1*std::cos(startPhi),
														rRand1*std::sin(startPhi), zRand);
	}
	else
	{ 
		zRand	= UUtils::Random(-1.*fDz,fDz); 
		rmin	 = fRmin2-((zRand-fDz)/(2.*fDz))*(fRmin1-fRmin2);
		rmax	 = fRmax2-((zRand-fDz)/(2.*fDz))*(fRmax1-fRmax2);
		rRand1 = std::sqrt(UUtils::Random()*(UUtils::sqr(rmax)-UUtils::sqr(rmin))+UUtils::sqr(rmin));
		point	= UVector3 (rRand1*std::cos(endPhi),
														rRand1*std::sin(endPhi), zRand);
	 
	}
	return point+offset;
}


//
// GetPointOnTubs
//
// Auxiliary method for GetPoint On Surface
//
UVector3 UPolycone::GetPointOnTubs(double fRMin, double fRMax,
																				 double zOne,	double zTwo,
																				 double& totArea) const
{ 
	double xRand,yRand,zRand,phi,cosphi,sinphi,chose,
					 aOne,aTwo,aFou,rRand,fDz,fSPhi,fDPhi;
	fDz = std::fabs(0.5*(zTwo-zOne));
	fSPhi = startPhi;
	fDPhi = endPhi-startPhi;
	
	aOne = 2.*fDz*fDPhi*fRMax;
	aTwo = 2.*fDz*fDPhi*fRMin;
	aFou = 2.*fDz*(fRMax-fRMin);
	totArea = aOne+aTwo+2.*aFou;
	phi		= UUtils::Random(startPhi,endPhi);
	cosphi = std::cos(phi);
	sinphi = std::sin(phi);
	rRand	= fRMin + (fRMax-fRMin)*std::sqrt(UUtils::Random());
 
	if(startPhi == 0 && endPhi == 2*UUtils::kPi) 
		aFou = 0;
	
	chose	= UUtils::Random(0.,aOne+aTwo+2.*aFou);
	if( (chose >= 0) && (chose < aOne) )
	{
		xRand = fRMax*cosphi;
		yRand = fRMax*sinphi;
		zRand = UUtils::Random(-1.*fDz,fDz);
		return UVector3(xRand, yRand, zRand+0.5*(zTwo+zOne));
	}
	else if( (chose >= aOne) && (chose < aOne + aTwo) )
	{
		xRand = fRMin*cosphi;
		yRand = fRMin*sinphi;
		zRand = UUtils::Random(-1.*fDz,fDz);
		return UVector3(xRand, yRand, zRand+0.5*(zTwo+zOne));
	}
	else if( (chose >= aOne+aTwo) && (chose <aOne+aTwo+aFou) )
	{
		xRand = rRand*std::cos(fSPhi+fDPhi);
		yRand = rRand*std::sin(fSPhi+fDPhi);
		zRand = UUtils::Random(-1.*fDz,fDz);
		return UVector3(xRand, yRand, zRand+0.5*(zTwo+zOne));
	}

	// else

	xRand = rRand*std::cos(fSPhi+fDPhi);
	yRand = rRand*std::sin(fSPhi+fDPhi);
	zRand = UUtils::Random(-1.*fDz,fDz);
	return UVector3(xRand, yRand, zRand+0.5*(zTwo+zOne));
}


//
// GetPointOnRing
//
// Auxiliary method for GetPoint On Surface
//
UVector3 UPolycone::GetPointOnRing(double fRMin1, double fRMax1,
																				 double fRMin2,double fRMax2,
																				 double zOne) const
{
	double xRand,yRand,phi,cosphi,sinphi,rRand1,rRand2,A1,Atot,rCh;
	
	phi		= UUtils::Random(startPhi,endPhi);
	cosphi = std::cos(phi);
	sinphi = std::sin(phi);

	if(fRMin1==fRMin2)
	{
		rRand1 = fRMin1; A1=0.;
	}
	else
	{
		rRand1 = UUtils::Random(fRMin1,fRMin2);
		A1=std::fabs(fRMin2*fRMin2-fRMin1*fRMin1);
	}
	if(fRMax1==fRMax2)
	{
		rRand2=fRMax1; Atot=A1;
	}
	else
	{
		rRand2 = UUtils::Random(fRMax1,fRMax2);
		Atot	 = A1+std::fabs(fRMax2*fRMax2-fRMax1*fRMax1);
	}
	rCh	 = UUtils::Random(0.,Atot);
 
	if(rCh>A1) { rRand1=rRand2; }
	
	xRand = rRand1*cosphi;
	yRand = rRand1*sinphi;

	return UVector3(xRand, yRand, zOne);
}


//
// GetPointOnCut
//
// Auxiliary method for Get Point On Surface
//
UVector3 UPolycone::GetPointOnCut(double fRMin1, double fRMax1,
																				double fRMin2, double fRMax2,
																				double zOne,	double zTwo,
																				double& totArea) const
{	 if(zOne==zTwo)
		{
			return GetPointOnRing(fRMin1, fRMax1,fRMin2,fRMax2,zOne);
		}
		if( (fRMin1 == fRMin2) && (fRMax1 == fRMax2) )
		{
			return GetPointOnTubs(fRMin1, fRMax1,zOne,zTwo,totArea);
		}
		return GetPointOnCone(fRMin1,fRMax1,fRMin2,fRMax2,zOne,zTwo,totArea);
}


//
// GetPointOnSurface
//
UVector3 UPolycone::GetPointOnSurface() const
{
	if (!genericPcon)	// Polycone by faces
	{
		double Area=0,totArea=0,Achose1=0,Achose2=0,phi,cosphi,sinphi,rRand;
		int i=0;
		int numPlanes = original_parameters->Num_z_planes;
	
		phi = UUtils::Random(startPhi,endPhi);
		cosphi = std::cos(phi);
		sinphi = std::sin(phi);

		rRand = original_parameters->Rmin[0] +
			( (original_parameters->Rmax[0]-original_parameters->Rmin[0])
	* std::sqrt(UUtils::Random()) );

		std::vector<double> areas;			 // (numPlanes+1);
		std::vector<UVector3> points; // (numPlanes-1);
	
		areas.push_back(UUtils::kPi*(UUtils::sqr(original_parameters->Rmax[0])
											 -UUtils::sqr(original_parameters->Rmin[0])));

		for(i=0; i<numPlanes-1; i++)
		{
			Area = (original_parameters->Rmin[i]+original_parameters->Rmin[i+1])
					 * std::sqrt(UUtils::sqr(original_parameters->Rmin[i]
											-original_parameters->Rmin[i+1])+
											 UUtils::sqr(original_parameters->Z_values[i+1]
											-original_parameters->Z_values[i]));

			Area += (original_parameters->Rmax[i]+original_parameters->Rmax[i+1])
						* std::sqrt(UUtils::sqr(original_parameters->Rmax[i]
											 -original_parameters->Rmax[i+1])+
												UUtils::sqr(original_parameters->Z_values[i+1]
											 -original_parameters->Z_values[i]));

			Area *= 0.5*(endPhi-startPhi);
		
			if(startPhi==0.&& endPhi == 2*UUtils::kPi)
			{
				Area += std::fabs(original_parameters->Z_values[i+1]
												 -original_parameters->Z_values[i])*
												 (original_parameters->Rmax[i]
												 +original_parameters->Rmax[i+1]
												 -original_parameters->Rmin[i]
												 -original_parameters->Rmin[i+1]);
			}
			areas.push_back(Area);
			totArea += Area;
		}
	
		areas.push_back(UUtils::kPi*(UUtils::sqr(original_parameters->Rmax[numPlanes-1])-
												UUtils::sqr(original_parameters->Rmin[numPlanes-1])));
	
		totArea += (areas[0]+areas[numPlanes]);
		double chose = UUtils::Random(0.,totArea);

		if( (chose>=0.) && (chose<areas[0]) )
		{
			return UVector3(rRand*cosphi, rRand*sinphi,
													 original_parameters->Z_values[0]);
		}
	
		for (i=0; i<numPlanes-1; i++)
		{
			Achose1 += areas[i];
			Achose2 = (Achose1+areas[i+1]);
			if(chose>=Achose1 && chose<Achose2)
			{
				return GetPointOnCut(original_parameters->Rmin[i],
														 original_parameters->Rmax[i],
														 original_parameters->Rmin[i+1],
														 original_parameters->Rmax[i+1],
														 original_parameters->Z_values[i],
														 original_parameters->Z_values[i+1], Area);
			}
		}

		rRand = original_parameters->Rmin[numPlanes-1] +
			( (original_parameters->Rmax[numPlanes-1]-original_parameters->Rmin[numPlanes-1])
	* std::sqrt(UUtils::Random()) );

	
		return UVector3(rRand*cosphi,rRand*sinphi,
												 original_parameters->Z_values[numPlanes-1]);	

	}
	else	// Generic Polycone
	{
		return GetPointOnSurfaceGeneric();	
	}
}

//
// CreatePolyhedron
//
UPolyhedron* UPolycone::CreatePolyhedron() const
{ 
	//
	// This has to be fixed in visualization. Fake it for the moment.
	// 
	if (!genericPcon)
	{
		return new UPolyhedronPcon( original_parameters->Start_angle,
																 original_parameters->Opening_angle,
																 original_parameters->Num_z_planes,
																 original_parameters->Z_values,
																 original_parameters->Rmin,
																 original_parameters->Rmax );
	}
	else
	{
		// The following code prepares for:
		// HepPolyhedron::createPolyhedron(int Nnodes, int Nfaces,
		//																	const double xyz[][3],
		//																	const int faces_vec[][4])
		// Here is an extract from the header file HepPolyhedron.h:
		/**
		 * Creates user defined polyhedron.
		 * This function allows to the user to define arbitrary polyhedron.
		 * The faces of the polyhedron should be either triangles or planar
		 * quadrilateral. Nodes of a face are defined by indexes pointing to
		 * the elements in the xyz array. Numeration of the elements in the
		 * array starts from 1 (like in fortran). The indexes can be positive
		 * or negative. Negative sign means that the corresponding edge is
		 * invisible. The normal of the face should be directed to exterior
		 * of the polyhedron. 
		 * 
		 * @param	Nnodes number of nodes
		 * @param	Nfaces number of faces
		 * @param	xyz		nodes
		 * @param	faces_vec	faces (quadrilaterals or triangles)
		 * @return status of the operation - is non-zero in case of problem
		 */
		const int numSide =
					int(UPolyhedron::GetNumberOfRotationSteps()
								* (endPhi - startPhi) / 2*UUtils::kPi) + 1;
		int nNodes;
		int nFaces;
		typedef double double3[3];
		double3* xyz;
		typedef int int4[4];
		int4* faces_vec;
		if (phiIsOpen)
		{
			// Triangulate open ends. Simple ear-chopping algorithm...
			// I'm not sure how robust this algorithm is (J.Allison).
			//
			std::vector<bool> chopped(numCorner, false);
			std::vector<int*> triQuads;
			int remaining = numCorner;
			int iStarter = 0;
			while (remaining >= 3)
			{
				// Find unchopped corners...
				//
				int A = -1, B = -1, C = -1;
				int iStepper = iStarter;
				do
				{
					if (A < 0)			{ A = iStepper; }
					else if (B < 0) { B = iStepper; }
					else if (C < 0) { C = iStepper; }
					do
					{
						if (++iStepper >= numCorner) { iStepper = 0; }
					}
					while (chopped[iStepper]);
				}
				while (C < 0 && iStepper != iStarter);

				// Check triangle at B is pointing outward (an "ear").
				// Sign of z Cross product determines...
				//
				double BAr = corners[A].r - corners[B].r;
				double BAz = corners[A].z - corners[B].z;
				double BCr = corners[C].r - corners[B].r;
				double BCz = corners[C].z - corners[B].z;
				if (BAr * BCz - BAz * BCr < VUSolid::Tolerance())
				{
					int* tq = new int[3];
					tq[0] = A + 1;
					tq[1] = B + 1;
					tq[2] = C + 1;
					triQuads.push_back(tq);
					chopped[B] = true;
					--remaining;
				}
				else
				{
					do
					{
						if (++iStarter >= numCorner) { iStarter = 0; }
					}
					while (chopped[iStarter]);
				}
			}
			// Transfer to faces...
			//
			nNodes = (numSide + 1) * numCorner;
			nFaces = numSide * numCorner + 2 * triQuads.size();
			faces_vec = new int4[nFaces];
			int iface = 0;
			int addition = numCorner * numSide;
			int d = numCorner - 1;
			for (int iEnd = 0; iEnd < 2; ++iEnd)
			{
				for (size_t i = 0; i < triQuads.size(); ++i)
				{
					// Negative for soft/auxiliary/normally invisible edges...
					//
					int a, b, c;
					if (iEnd == 0)
					{
						a = triQuads[i][0];
						b = triQuads[i][1];
						c = triQuads[i][2];
					}
					else
					{
						a = triQuads[i][0] + addition;
						b = triQuads[i][2] + addition;
						c = triQuads[i][1] + addition;
					}
					int ab = std::abs(b - a);
					int bc = std::abs(c - b);
					int ca = std::abs(a - c);
					faces_vec[iface][0] = (ab == 1 || ab == d)? a: -a;
					faces_vec[iface][1] = (bc == 1 || bc == d)? b: -b;
					faces_vec[iface][2] = (ca == 1 || ca == d)? c: -c;
					faces_vec[iface][3] = 0;
					++iface;
				}
			}

			// Continue with sides...

			xyz = new double3[nNodes];
			const double dPhi = (endPhi - startPhi) / numSide;
			double phi = startPhi;
			int ixyz = 0;
			for (int iSide = 0; iSide < numSide; ++iSide)
			{
				for (int iCorner = 0; iCorner < numCorner; ++iCorner)
				{
					xyz[ixyz][0] = corners[iCorner].r * std::cos(phi);
					xyz[ixyz][1] = corners[iCorner].r * std::sin(phi);
					xyz[ixyz][2] = corners[iCorner].z;
					if (iSide == 0)	 // startPhi
					{
						if (iCorner < numCorner - 1)
						{
							faces_vec[iface][0] = ixyz + 1;
							faces_vec[iface][1] = -(ixyz + numCorner + 1);
							faces_vec[iface][2] = ixyz + numCorner + 2;
							faces_vec[iface][3] = ixyz + 2;
						}
						else
						{
							faces_vec[iface][0] = ixyz + 1;
							faces_vec[iface][1] = -(ixyz + numCorner + 1);
							faces_vec[iface][2] = ixyz + 2;
							faces_vec[iface][3] = ixyz - numCorner + 2;
						}
					}
					else if (iSide == numSide - 1)	 // endPhi
					{
						if (iCorner < numCorner - 1)
							{
								faces_vec[iface][0] = ixyz + 1;
								faces_vec[iface][1] = ixyz + numCorner + 1;
								faces_vec[iface][2] = ixyz + numCorner + 2;
								faces_vec[iface][3] = -(ixyz + 2);
							}
						else
							{
								faces_vec[iface][0] = ixyz + 1;
								faces_vec[iface][1] = ixyz + numCorner + 1;
								faces_vec[iface][2] = ixyz + 2;
								faces_vec[iface][3] = -(ixyz - numCorner + 2);
							}
					}
					else
					{
						if (iCorner < numCorner - 1)
							{
								faces_vec[iface][0] = ixyz + 1;
								faces_vec[iface][1] = -(ixyz + numCorner + 1);
								faces_vec[iface][2] = ixyz + numCorner + 2;
								faces_vec[iface][3] = -(ixyz + 2);
							}
							else
							{
								faces_vec[iface][0] = ixyz + 1;
								faces_vec[iface][1] = -(ixyz + numCorner + 1);
								faces_vec[iface][2] = ixyz + 2;
								faces_vec[iface][3] = -(ixyz - numCorner + 2);
							}
						}
						++iface;
						++ixyz;
				}
				phi += dPhi;
			}

			// Last corners...

			for (int iCorner = 0; iCorner < numCorner; ++iCorner)
			{
				xyz[ixyz][0] = corners[iCorner].r * std::cos(phi);
				xyz[ixyz][1] = corners[iCorner].r * std::sin(phi);
				xyz[ixyz][2] = corners[iCorner].z;
				++ixyz;
			}
		}
		else	// !phiIsOpen - i.e., a complete 360 degrees.
		{
			nNodes = numSide * numCorner;
			nFaces = numSide * numCorner;;
			xyz = new double3[nNodes];
			faces_vec = new int4[nFaces];
			const double dPhi = (endPhi - startPhi) / numSide;
			double phi = startPhi;
			int ixyz = 0, iface = 0;
			for (int iSide = 0; iSide < numSide; ++iSide)
			{
				for (int iCorner = 0; iCorner < numCorner; ++iCorner)
				{
					xyz[ixyz][0] = corners[iCorner].r * std::cos(phi);
					xyz[ixyz][1] = corners[iCorner].r * std::sin(phi);
					xyz[ixyz][2] = corners[iCorner].z;

					if (iSide < numSide - 1)
					{
						if (iCorner < numCorner - 1)
						{
							faces_vec[iface][0] = ixyz + 1;
							faces_vec[iface][1] = -(ixyz + numCorner + 1);
							faces_vec[iface][2] = ixyz + numCorner + 2;
							faces_vec[iface][3] = -(ixyz + 2);
						}
						else
						{
							faces_vec[iface][0] = ixyz + 1;
							faces_vec[iface][1] = -(ixyz + numCorner + 1);
							faces_vec[iface][2] = ixyz + 2;
							faces_vec[iface][3] = -(ixyz - numCorner + 2);
						}
					}
					else	 // Last side joins ends...
					{
						if (iCorner < numCorner - 1)
						{
							faces_vec[iface][0] = ixyz + 1;
							faces_vec[iface][1] = -(ixyz + numCorner - nFaces + 1);
							faces_vec[iface][2] = ixyz + numCorner - nFaces + 2;
							faces_vec[iface][3] = -(ixyz + 2);
						}
						else
						{
							faces_vec[iface][0] = ixyz + 1;
							faces_vec[iface][1] = -(ixyz - nFaces + numCorner + 1);
							faces_vec[iface][2] = ixyz - nFaces + 2;
							faces_vec[iface][3] = -(ixyz - numCorner + 2);
						}
					}
					++ixyz;
					++iface;
				}
				phi += dPhi;
			}
		}
		UPolyhedron* polyhedron = new UPolyhedron;
		int problem = polyhedron->createPolyhedron(nNodes, nFaces, xyz, faces_vec);
		delete [] faces_vec;
		delete [] xyz;
		if (problem)
		{
			std::ostringstream message;
			message << "Problem creating UPolyhedron for: " << GetName();
			// UException("UPolycone::CreatePolyhedron()", "GeomSolids1002",
			//						JustWarning, message);
			delete polyhedron;
			return 0;
		}
		else
		{
			return polyhedron;
		}
	}
}


//
// UPolyconeHistorical stuff
//

UPolyconeHistorical::UPolyconeHistorical()
	: Start_angle(0.), Opening_angle(0.), Num_z_planes(0),
		Z_values(0), Rmin(0), Rmax(0)
{
}

UPolyconeHistorical::~UPolyconeHistorical()
{
	delete [] Z_values;
	delete [] Rmin;
	delete [] Rmax;
}

UPolyconeHistorical::
UPolyconeHistorical( const UPolyconeHistorical &source )
{
	Start_angle	 = source.Start_angle;
	Opening_angle = source.Opening_angle;
	Num_z_planes	= source.Num_z_planes;
	
	Z_values	= new double[Num_z_planes];
	Rmin			= new double[Num_z_planes];
	Rmax			= new double[Num_z_planes];
	
	for( int i = 0; i < Num_z_planes; i++)
	{
		Z_values[i] = source.Z_values[i];
		Rmin[i]		 = source.Rmin[i];
		Rmax[i]		 = source.Rmax[i];
	}
}

UPolyconeHistorical&
UPolyconeHistorical::operator=( const UPolyconeHistorical& right )
{
	if ( &right == this ) return *this;

	if (&right)
	{
		Start_angle	 = right.Start_angle;
		Opening_angle = right.Opening_angle;
		Num_z_planes	= right.Num_z_planes;
	
		delete [] Z_values;
		delete [] Rmin;
		delete [] Rmax;
		Z_values	= new double[Num_z_planes];
		Rmin			= new double[Num_z_planes];
		Rmax			= new double[Num_z_planes];
	
		for( int i = 0; i < Num_z_planes; i++)
		{
			Z_values[i] = right.Z_values[i];
			Rmin[i]		 = right.Rmin[i];
			Rmax[i]		 = right.Rmax[i];
		}
	}
	return *this;
}

void UPolycone::Extent (UVector3 &aMin, UVector3 &aMax) const
{
  enclosingCylinder->Extent(aMin, aMax);
}

