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
// $Id: UPolycone2.cc 66241 2012-12-13 18:34:42Z gunter $
//
// 
// --------------------------------------------------------------------
// GEANT 4 class source file
//
//
// UPolycone2.cc
//
// Implementation of a CSG polycone
//
// --------------------------------------------------------------------

#include "UUtils.hh"
#include <string>
#include <cmath>
#include <sstream>
#include "UPolycone2.hh"
#include "UPolycone.hh"
#include "UMultiUnion.hh"

#include "UEnclosingCylinder.hh"
#include "UReduciblePolygon.hh"

#include "UTubs.hh"
#include "UCons.hh"
#include "UTransform3D.hh"

using namespace std;

//
// Constructor (GEANT3 style parameters)
//	
UPolycone2::UPolycone2( const std::string& name, 
															double phiStart,
															double phiTotal,
															int numZPlanes,
												const double zPlane[],
												const double rInner[],
												const double rOuter[]	)
	: UMultiUnion( name )
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

  double prevZ, prevRmax, prevRmin;

//  int curSolid = 0;

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
				// UException("UPolycone2::UPolycone2()", "GeomSolids0002",
				//						FatalErrorInArgument, message);
			}
		} 

    double rMin = rInner[i];
    double rMax = rOuter[i];
    double z = zPlane[i];

    if (i > 0)
    {
      if (z > prevZ)
      {
        VUSolid *solid;
        double dz = (z - prevZ)/2;
        UTransform3D *trans = new UTransform3D(0, 0, prevZ + dz);

        if (rMin == prevRmin && prevRmax == rMax)
          solid = new UTubs("", rMin, rMax, dz, phiStart, phiTotal);
        else
          solid = new UCons("", prevRmin, prevRmax, rMin, rMax, dz, phiStart, phiTotal);

        this->AddNode(*solid, *trans);
      }
    }

		original_parameters->Z_values[i] = zPlane[i];
		original_parameters->Rmin[i] = rInner[i];
		original_parameters->Rmax[i] = rOuter[i];

    prevZ = z;
    prevRmin = rMin;
    prevRmax = rMax;
	}

  this->Voxelize();

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
UPolycone2::UPolycone2( const std::string& name, 
															double phiStart,
															double phiTotal,
															int		numRZ,
												const double r[],
												const double z[]	 )
	: UMultiUnion( name )
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
void UPolycone2::Create( double phiStart,
												 double phiTotal,
												 UReduciblePolygon *rz		)
{
  return;

}


//
// Destructor
//
UPolycone2::~UPolycone2()
{
	//delete [] corners;
	//delete original_parameters;
}

//
// Stream object contents to an output stream
//
std::ostream& UPolycone2::StreamInfo( std::ostream& os ) const
{
	int oldprc = os.precision(16);
	os << "-----------------------------------------------------------\n"
		 << "		*** Dump for solid - " << GetName() << " ***\n"
		 << "		===================================================\n"
		 << " Solid type: UPolycone2\n"
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
