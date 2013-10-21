
// UGenericPolycone.cc
//
// Implementation of a CSG polycone
//
// --------------------------------------------------------------------

#include "UUtils.hh"
#include <string>
#include <cmath>
#include <sstream>
#include "UGenericPolycone.hh"

#include "UPolyconeSide.hh"
#include "UPolyPhiFace.hh"

 


#include "UEnclosingCylinder.hh"
#include "UReduciblePolygon.hh"


using namespace std;
//
// Constructor (generic parameters)
//
UGenericPolycone::UGenericPolycone( const std::string& name, 
         	double phiStart,
		double phiTotal,
		int		numRZ,
		const double r[],
	        const double z[]	 )
	: UVCSGfaceted( name )
{
	UReduciblePolygon *rz = new UReduciblePolygon( r, z, numRZ );
	
	Create( phiStart, phiTotal, rz );
		
	delete rz;
}


//
// Create
//
// Generic create routine, called by each constructor after
// conversion of arguments
//
void UGenericPolycone::Create( double phiStart,
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
		UUtils::Exception("UGenericPolycone::Create()", "GeomSolids0002",
				  FatalErrorInArguments,1, message.str().c_str());
	}
		
	double rzArea = rz->Area();
	if (rzArea < -VUSolid::Tolerance())
		rz->ReverseOrder();

	else if (rzArea < -VUSolid::Tolerance())
	{
		std::ostringstream message;
		message << "Illegal input parameters - " << GetName() << std::endl
						<< "				R/Z Cross section is zero or near zero: " << rzArea;
		UUtils::Exception("UGenericPolycone::Create()", "GeomSolids0002",
			    FatalErrorInArguments,1, message.str().c_str());
	}
		
	if ( (!rz->RemoveDuplicateVertices( VUSolid::Tolerance() ))
		|| (!rz->RemoveRedundantVertices( VUSolid::Tolerance() ))		 ) 
	{
		std::ostringstream message;
		message << "Illegal input parameters - " << GetName() << std::endl
						<< "				Too few unique R/Z values !";
		UUtils::Exception("UGenericPolycone::Create()", "GeomSolids0002",
				  FatalErrorInArguments,1, message.str().c_str());
	}

	if (rz->CrossesItself(1/UUtils::kInfinity)) 
	{
		std::ostringstream message;
		message << "Illegal input parameters - " << GetName() << std::endl
						<< "				R/Z segments Cross !";
		UUtils::Exception("UGenericPolycone::Create()", "GeomSolids0002",
				  FatalErrorInArguments,1, message.str().c_str());
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
		
		if (corner->r < 1/UUtils::kInfinity && next->r < 1/UUtils::kInfinity) continue;
		
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
		new UEnclosingCylinder(rz->Amax(), rz->Bmax(), rz->Bmin(), phiIsOpen, phiStart, phiTotal );

  InitVoxels(*rz, enclosingCylinder->radius);

  fNoVoxels = fMaxSection < 2;

}


//
// Fake default constructor - sets only member data and allocates memory
//														for usage restricted to object persistency.
//

/*
UGenericPolycone::UGenericPolycone( __void__& a )
	: UVCSGfaceted(a), startPhi(0.),	endPhi(0.), phiIsOpen(false),
		genericPcon(false), numCorner(0), corners(0),
		fOriginalParameters(0), enclosingCylinder(0)
{
}
*/


//
// Destructor
//
UGenericPolycone::~UGenericPolycone()
{
	delete [] corners;
	delete enclosingCylinder;
}


//
// Copy constructor
//
UGenericPolycone::UGenericPolycone( const UGenericPolycone &source )
	: UVCSGfaceted( source )
{
	CopyStuff( source );
}


//
// Assignment operator
//
const UGenericPolycone &UGenericPolycone::operator=( const UGenericPolycone &source )
{
	if (this == &source) return *this;
	
	UVCSGfaceted::operator=( source );
	
	delete [] corners;

	delete enclosingCylinder;
	
	CopyStuff( source );
	
	return *this;
}


//
// CopyStuff
//
void UGenericPolycone::CopyStuff( const UGenericPolycone &source )
{
	//
	// Simple stuff
	//
	startPhi	= source.startPhi;
	endPhi		= source.endPhi;
	phiIsOpen	= source.phiIsOpen;
	numCorner	= source.numCorner;

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
	// Enclosing cylinder
	//
	enclosingCylinder = new UEnclosingCylinder( *source.enclosingCylinder );
}


//
// Reset
//
bool UGenericPolycone::Reset()
{
	
		std::ostringstream message;
		message << "Solid " << GetName() << " built using generic construct."
						<< std::endl << "Not applicable to the generic construct !";
		// UException("UGenericPolycone::Reset(,,)", "GeomSolids1001",
		//						JustWarning, message, "Parameters NOT resetted.");
		return 1;

}


//
// Inside
//
// This is an override of UVCSGfaceted::Inside, created in order
// to speed things up by first checking with UEnclosingCylinder.
//
VUSolid::EnumInside UGenericPolycone::Inside( const UVector3 &p ) const
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
double UGenericPolycone::DistanceToIn( const UVector3 &p,
																	 const UVector3 &v, double aPstep) const
{
	//
	// Quick test
	//
	if (enclosingCylinder->ShouldMiss(p,v))
		return UUtils::kInfinity;
	
	//
	// Long answer
	//
	return UVCSGfaceted::DistanceToIn( p, v, aPstep);
}

//
// GetEntityType
//
UGeometryType	UGenericPolycone::GetEntityType() const
{
	return std::string("UGenericPolycone");
}

//
// Make a clone of the object
//
VUSolid* UGenericPolycone::Clone() const
{
	return new UGenericPolycone(*this);
}

//
// Stream object contents to an output stream
//
std::ostream& UGenericPolycone::StreamInfo( std::ostream& os ) const
{
	int oldprc = os.precision(16);
	os << "-----------------------------------------------------------\n"
		 << "		*** Dump for solid - " << GetName() << " ***\n"
		 << "		===================================================\n"
		 << " Solid type: UGenericPolycone\n"
		 << " Parameters: \n"
		 << "		starting phi angle : " << startPhi/(UUtils::kPi/180.0) << " degrees \n"
		 << "		ending phi angle	 : " << endPhi/(UUtils::kPi/180.0) << " degrees \n";
	int i=0;

        os << "    number of RZ points: " << numCorner << "\n"
           << "              RZ values (corners): \n";
        for (i=0; i<numCorner; i++)
        {
         os << "                         "
          << corners[i].r << ", " << corners[i].z << "\n";
         }

	os << "-----------------------------------------------------------\n";
	os.precision(oldprc);

	return os;
}

//
// GetPointOnSurface
//
UVector3 UGenericPolycone::GetPointOnSurface() const
{
  return GetPointOnSurfaceGeneric();	

}

//
// CreatePolyhedron
//
UPolyhedron* UGenericPolycone::CreatePolyhedron() const
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
			UUtils::Exception("UGenericPolycone::CreatePolyhedron()", "GeomSolids1002",
				    Warning,1, message.str().c_str());
			delete polyhedron;
			return 0;
		}
		else
		{
			return polyhedron;
		}
	
}



void UGenericPolycone::Extent (UVector3 &aMin, UVector3 &aMax) const
{
  enclosingCylinder->Extent(aMin, aMax);
}

