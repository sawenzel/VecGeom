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
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: UVCSGfaceted.cc 66241 2012-12-13 18:34:42Z gunter $
//
// 
// --------------------------------------------------------------------
// GEANT 4 class source file
//
//
// UVCSGfaceted.cc
//
// Implementation of the virtual class of a CSG type shape that is built
// entirely out of UVCSGface faces.
//
// --------------------------------------------------------------------

#include "UUtils.hh"
#include <string>
#include <cmath>
#include <sstream>
#include "UVCSGfaceted.hh"
#include "UVCSGface.hh"
#include "UVoxelizer.hh"

//#include "USolidExtentList.hh"






	 

//#include "UVisExtent.hh"

//
// Constructor
//
UVCSGfaceted::UVCSGfaceted( const std::string& name )
	: VUSolid(name),
		numFace(0), faces(0), fCubicVolume(0.), fSurfaceArea(0.), fpPolyhedron(0),
		fStatistics(1000000), fCubVolEpsilon(0.001), fAreaAccuracy(-1.)
{
}



//
// Destructor
//
UVCSGfaceted::~UVCSGfaceted()
{
	DeleteStuff();
	delete fpPolyhedron;
}


//
// Copy constructor
//
UVCSGfaceted::UVCSGfaceted( const UVCSGfaceted &source )
	: VUSolid( source )
{
	fStatistics = source.fStatistics;
	fCubVolEpsilon = source.fCubVolEpsilon;
	fAreaAccuracy = source.fAreaAccuracy;

	CopyStuff( source );
}


//
// Assignment operator
//
const UVCSGfaceted &UVCSGfaceted::operator=( const UVCSGfaceted &source )
{
	if (&source == this) { return *this; }
	
	// Copy base class data
	//
	VUSolid::operator=(source);

	// Copy data
	//
	fStatistics = source.fStatistics;
	fCubVolEpsilon = source.fCubVolEpsilon;
	fAreaAccuracy = source.fAreaAccuracy;

	DeleteStuff();
	CopyStuff( source );
	
	return *this;
}


//
// CopyStuff (protected)
//
// Copy the contents of source
//
void UVCSGfaceted::CopyStuff( const UVCSGfaceted &source )
{
	numFace = source.numFace;
	if (numFace == 0) { return; }		// odd, but permissable?
	
	faces = new UVCSGface*[numFace];
	
	UVCSGface **face = faces,
			 **sourceFace = source.faces;
	do
	{
		*face = (*sourceFace)->Clone();
	} while( ++sourceFace, ++face < faces+numFace );
	fCubicVolume = source.fCubicVolume;
	fSurfaceArea = source.fSurfaceArea;
	fpPolyhedron = 0;
}


//
// DeleteStuff (protected)
//
// Delete all allocated objects
//
void UVCSGfaceted::DeleteStuff()
{
	if (numFace)
	{
		UVCSGface **face = faces;
		do
		{
			delete *face;
		} while( ++face < faces + numFace );

		delete [] faces;
	}
}


//
// CalculateExtent
//

/*
bool UVCSGfaceted::CalculateExtent( const EAxisType axis,
																			 const UVoxelLimits &voxelLimit,
																			 const UAffineTransform &transform,
																						 double &min,
																						 double &max ) const
{
	USolidExtentList	extentList( axis, voxelLimit );

	//
	// Loop over all faces, checking min/max extent as we go.
	//
	UVCSGface **face = faces;
	do
	{
		(*face)->CalculateExtent( axis, voxelLimit, transform, extentList );
	} while( ++face < faces + numFace );
	
	//
	// Return min/max value
	//
	return extentList.GetExtent( min, max );
}
*/

//
// Inside
//
// It could be a good idea to override this virtual
// member to add first a simple test (such as spherical
// test or whatnot) and to call this version only if
// the simplier test fails.
//

/*
VUSolid::EnumInside UVCSGfaceted::Inside( const UVector3 &p ) const
{
	VUSolid::EnumInside answer=eOutside;
	UVCSGface **face = faces;
	double best = UUtils::Infinity();
	do
	{
		double distance;
		VUSolid::EnumInside result = (*face)->Inside( p, fgTolerance*0.5, &distance );
		if (result == eSurface) { return eSurface; }
		if (distance < best)
		{
			best = distance;
			answer = result;
		}
	} while( ++face < faces + numFace );

	return answer;
}
*/


//#include "UPolyconeSide.hh"

static std::vector<double> ranges;
static std::vector<std::vector<int> > candidates;

//
// Inside
//
// It could be a good idea to override this virtual
// member to add first a simple test (such as spherical
// test or whatnot) and to call this version only if
// the simplier test fails.
//
VUSolid::EnumInside UVCSGfaceted::Inside( const UVector3 &p ) const
{
  VUSolid::EnumInside answer=eOutside;
  UVCSGface **face = faces;
  double best = kInfinity;
  do
  {
    double distance;
    VUSolid::EnumInside result = (*face)->Inside( p, fgTolerance*0.5, &distance);
    if (result == eSurface) { return eSurface; }
    if (distance < best)
    { 
      best = distance;
      answer = result;
    }
  } while( ++face < faces + numFace );

  return answer;

  return InsideOptimized(p);
}


VUSolid::EnumInside UVCSGfaceted::InsideOptimized( const UVector3 &p ) const
{
  if (ranges.size() == 0)
  {
    ranges.resize(numFace+1);
    candidates.resize(numFace);
    for (int i = 0; i < numFace; i++)
    {
      UVCSGface *face = faces[i];
      double minZ = -face->Extent(UVector3(0, 0, -1));
      double maxZ = face->Extent(UVector3(0, 0, 1));
      if (i == 0) ranges[i] = minZ;
      ranges[i+1] = maxZ;
      candidates[i].push_back(i);
    }
  }
  //	do

  int index = UVoxelizer::BinarySearch(ranges, p.z);
  //	if (index < 0 || index >= ranges.size()) return eOutside;

  VUSolid::EnumInside answer=eOutside;
  UVCSGface **face = faces;
  double best = UUtils::Infinity();

  std::vector<int> &focus = candidates[index];
  int size = focus.size();

  for (int i = 0; i < size; i++)
  {
    index = focus[i];
    UVCSGface *face = (UVCSGface *) faces[index];

    double distance;
    VUSolid::EnumInside result = (*face).Inside( p, fgTolerance*0.5, &distance );
    if (result == eSurface) { return eSurface; }
    if (distance <= best)
    {
      best = distance;
      answer = result;
    }
  } // while( ++face < faces + numFace );

  return answer;
}

//
// SurfaceNormal
//

bool UVCSGfaceted::Normal( const UVector3 &p, UVector3 &n) const
{
	UVector3 answer;
	UVCSGface **face = faces;
	double best = UUtils::Infinity();
	do
	{
		double distance;
		UVector3 normal = (*face)->Normal( p, &distance );
		if (distance < best)
		{
			best = distance;
			answer = normal;
		}
	} while( ++face < faces + numFace );

	n = answer;

  return true;
}


//
// DistanceToIn(p,v)
//

double UVCSGfaceted::DistanceToIn( const UVector3& p,
  const UVector3& v, double aPstep) const
{
	double distance = UUtils::Infinity();
	double distFromSurface = UUtils::Infinity();
	UVCSGface **face = faces;
	UVCSGface *bestFace = *face;
	do
	{
		double	 faceDistance,
							 faceDistFromSurface;
		UVector3	 faceNormal;
		bool		faceAllBehind;
		if ((*face)->Distance( p, v, false, fgTolerance*0.5,
								faceDistance, faceDistFromSurface,
								faceNormal, faceAllBehind ) )
		{
			//
			// Intersecting face
			//
			if (faceDistance < distance)
			{
				distance = faceDistance;
				distFromSurface = faceDistFromSurface;
				bestFace = *face;
				if (distFromSurface <= 0) { return 0; }
			}
		}
	} while( ++face < faces + numFace );
	
	if (distance < UUtils::Infinity() && distFromSurface<fgTolerance/2)
	{
		if (bestFace->Safety(p,false) < fgTolerance/2)	{ distance = 0; }
	}

	return distance;
}




//
// DistanceToOut(p,v)
//

double UVCSGfaceted::DistanceToOut( const UVector3 &p, const UVector3  &v, UVector3 &n, bool &aConvex, double aPstep) const
{
	bool allBehind = true;
	double distance = UUtils::Infinity();
	double distFromSurface = UUtils::Infinity();
	UVector3 normal;
	
	UVCSGface **face = faces;
	UVCSGface *bestFace = *face;
	do
	{
		double	faceDistance,
							faceDistFromSurface;
		UVector3	faceNormal;
		bool		faceAllBehind;
		if ((*face)->Distance( p, v, true, fgTolerance/2,
								faceDistance, faceDistFromSurface,
								faceNormal, faceAllBehind ) )
		{
			//
			// Intersecting face
			//
			if ( (distance < UUtils::Infinity()) || (!faceAllBehind) )	{ allBehind = false; }
			if (faceDistance < distance)
			{
				distance = faceDistance;
				distFromSurface = faceDistFromSurface;
				normal = faceNormal;
				bestFace = *face;
				if (distFromSurface <= 0)	{ break; }
			}
		}
	} while( ++face < faces + numFace );
	
	if (distance < UUtils::Infinity())
	{
		if (distFromSurface <= 0)
		{
			distance = 0;
		}
		else if (distFromSurface<fgTolerance/2)
		{
			if (bestFace->Safety(p,true) < fgTolerance/2)	{ distance = 0; }
		}

		aConvex = allBehind;
		n = normal;
	}
	else
	{ 
		if (Inside(p) == eSurface)	{ distance = 0; }
		aConvex = false;
	}

	return distance;
}




//
// DistanceTo
//
// Protected routine called by DistanceToIn and DistanceToOut
//
double UVCSGfaceted::DistanceTo( const UVector3 &p,
																		const bool outgoing ) const
{
	UVCSGface **face = faces;
	double best = UUtils::Infinity();
	do
	{
		double distance = (*face)->Safety( p, outgoing );
		if (distance < best)	{ best = distance; }
	} while( ++face < faces + numFace );

	return (best < 0.5*fgTolerance) ? 0 : best;
}


/*
//
// DescribeYourselfTo
//
void UVCSGfaceted::DescribeYourselfTo( UVGraphicsScene& scene ) const
{
	 scene.AddSolid( *this );
}
*/


/*
//
// GetExtent
//
// Define the sides of the box into which our solid instance would fit.
//
UVisExtent UVCSGfaceted::GetExtent() const 
{
	static const UVector3 xMax(1,0,0), xMin(-1,0,0),
														 yMax(0,1,0), yMin(0,-1,0),
														 zMax(0,0,1), zMin(0,0,-1);
	static const UVector3 *axes[6] =
		 { &xMin, &xMax, &yMin, &yMax, &zMin, &zMax };
	
	double answers[6] =
		 {-UUtils::Infinity(), -UUtils::Infinity(), -UUtils::Infinity(), -UUtils::Infinity(), -UUtils::Infinity(), -UUtils::Infinity()};

	UVCSGface **face = faces;
	do
	{		
		const UVector3 **axis = axes+5 ;
		double *answer = answers+5;
		do
		{
			double testFace = (*face)->Extent( **axis );
			if (testFace > *answer)	{ *answer = testFace; }
		}
		while( --axis, --answer >= answers );
		
	} while( ++face < faces + numFace );
	
		return UVisExtent( -answers[0], answers[1], 
												-answers[2], answers[3],
												-answers[4], answers[5]	);
}
*/

//
// GetEntityType
//
UGeometryType UVCSGfaceted::GetEntityType() const
{
	return std::string("UCSGfaceted");
}


//
// Stream object contents to an output stream
//
std::ostream& UVCSGfaceted::StreamInfo( std::ostream& os ) const
{
	os << "-----------------------------------------------------------\n"
		 << "		*** Dump for solid - " << GetName() << " ***\n"
		 << "		===================================================\n"
		 << " Solid type: UVCSGfaceted\n"
		 << " Parameters: \n"
		 << "		number of faces: " << numFace << "\n"
		 << "-----------------------------------------------------------\n";

	return os;
}


//
// GetCubVolStatistics
//
int UVCSGfaceted::GetCubVolStatistics() const
{
	return fStatistics;
}


//
// GetCubVolEpsilon
//
double UVCSGfaceted::GetCubVolEpsilon() const
{
	return fCubVolEpsilon;
}


//
// SetCubVolStatistics
//
void UVCSGfaceted::SetCubVolStatistics(int st)
{
	fCubicVolume=0.;
	fStatistics=st;
}


//
// SetCubVolEpsilon
//
void UVCSGfaceted::SetCubVolEpsilon(double ep)
{
	fCubicVolume=0.;
	fCubVolEpsilon=ep;
}


//
// GetAreaStatistics
//
int UVCSGfaceted::GetAreaStatistics() const
{
	return fStatistics;
}


//
// GetAreaAccuracy
//
double UVCSGfaceted::GetAreaAccuracy() const
{
	return fAreaAccuracy;
}


//
// SetAreaStatistics
//
void UVCSGfaceted::SetAreaStatistics(int st)
{
	fSurfaceArea=0.;
	fStatistics=st;
}


//
// SetAreaAccuracy
//
void UVCSGfaceted::SetAreaAccuracy(double ep)
{
	fSurfaceArea=0.;
	fAreaAccuracy=ep;
}


//
// Capacity
//
double UVCSGfaceted::Capacity()
{
	if(fCubicVolume != 0.) {;}
	else	 { fCubicVolume = EstimateCubicVolume(fStatistics,fCubVolEpsilon); }
	return fCubicVolume;
}


//
// SurfaceArea
//
double UVCSGfaceted::SurfaceArea()
{
	if(fSurfaceArea != 0.) {;}
	else	 { fSurfaceArea = EstimateSurfaceArea(fStatistics,fAreaAccuracy); }
	return fSurfaceArea;
}


//
// GetPolyhedron
//
UPolyhedron* UVCSGfaceted::GetPolyhedron () const
{
	if (!fpPolyhedron /*||
			fpPolyhedron->GetNumberOfRotationStepsAtTimeOfCreation() !=
			fpPolyhedron->GetNumberOfRotationSteps()*/)
	{
		delete fpPolyhedron;
		fpPolyhedron = CreatePolyhedron();
	}
	return fpPolyhedron;
}


//
// GetPointOnSurfaceGeneric proportional to Areas of faces
// in case of GenericPolycone or GenericPolyhedra
//
UVector3 UVCSGfaceted::GetPointOnSurfaceGeneric( ) const
{
	// Preparing variables
	//
	UVector3 answer=UVector3(0.,0.,0.);
	UVCSGface **face = faces;
	double area = 0;
	int i;
	std::vector<double> areas; 

	// First step: calculate surface areas
	//
	do
	{
		double result = (*face)->SurfaceArea( );
		areas.push_back(result);
		area=area+result;
	} while( ++face < faces + numFace );

	// Second Step: choose randomly one surface
	//
	UVCSGface **face1 = faces;
	double chose = area*UUtils::Random();
	double Achose1, Achose2;
	Achose1=0; Achose2=0.; 
	i=0;

	do
	{
		Achose2+=areas[i];
		if(chose>=Achose1 && chose<Achose2)
		{
			UVector3 point;
			point= (*face1)->GetPointOnFace();
			return point;
		}
		i++;
		Achose1=Achose2;
	} while( ++face1 < faces + numFace );

	return answer;
}

//
// DistanceToIn(p)
//
double UVCSGfaceted::SafetyFromOutside( const UVector3 &p, bool ) const
{
  return DistanceTo( p, false );
}

//
// DistanceToOut(p)
//
double UVCSGfaceted::SafetyFromInside( const UVector3 &p, bool ) const
{
  return DistanceTo( p, true );
}
