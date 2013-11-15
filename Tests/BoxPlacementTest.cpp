/*
 * BoxPlacementTest.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: swenzel
 */

// verify matrix and box implementation against the ROOT implementation

// from root we need
#include "TGeoMatrix.h"
#include "TGeoBBox.h"
#include "TGeoManager.h"

// from FastGeom we need
#include "../GeoManager.h"
#include "../PhysicalVolume.h"
#include "../PhysicalBox.h"
#include "../Vector3D.h"

// for the rotations
const double[][3] EulerAngles = {{0.,0.,0.},
										{30.,0, 0.},
										{0, 45., 0.},
										{0, 0, 67.5},
										{0, 0, 180.},
										{30, 48, 0.},
										{78.2, 81.0, -10.}};


const double[][3] TransCases = {{0,0,0}
								{10, 0, 0},
								{10, -10, 0},
							    }


int main()
{
//	TGeoManager *geom = new TGeoManager();
//	TGeoVolume * box = geom->MakeBox(100,100,100); ....

	BoxParameters *bp = new BoxParameters(15,20,25);

	// particle position and particle direction
	Vector3D pPos( -90 , -90 , -90 );
	Vector3D pDir(1./sqrt(3.),1./sqrt(3.),1./sqrt(3.)); // fly towards origin

	// testparticle is in universe (in some corner and flies towards the origin)

	// create root shape and placement
	for(int t=0; t<TN; ++t) // translation cases
	{
		for(int r=0; r<RN; ++r) // rotation cases
		{

			// also try the other approach with a Node placement
			TGeoMatrix * matrix = new TGeoCombinedMatrix ();
			// transform coordinates
			Vector3D pPos2;
			Vector3D pDir2;
			//
			matrix->MasterToLocal(&pPos, &pPos2);
			matrix->MasterToLocalVec(&pDir, &pDir2);
			double dRoot = box->DistFromOutside(pPos, pDir, ...);

			PhysicalVolume *fastbox = GeoManager::MakePlacedBox(bp,
								new TransformationMatrix(TransCases[t][0], TransCases[t][1], TransCases[t][2],
														 EulerAngles[t][0], EulerAngles[t][1], EulerAngles[t][2]));
			double dFastGeom = fastbox->DistanceToIn(pPos, pDir);


		// delete pointer
			delete matrix;
			delete fastbox;
		}
	}
}
