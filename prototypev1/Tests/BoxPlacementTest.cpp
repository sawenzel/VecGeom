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
#include "TGeoVolume.h"

// from FastGeom we need
#include "../GeoManager.h"
#include "../PhysicalVolume.h"
#include "../TransformationMatrix.h"
#include "../PhysicalBox.h"
#include "../Vector3D.h"

#include <iostream>

#include <vector>
// for the rotations ( C++11 feature )
const std::vector<std::vector<double>> EulerAngles  {{0.,0.,0.},
										{30.,0, 0.},
										{0, 45., 0.},
										{0, 0, 67.5},
										{0, 0, 180.},
										{30, 48, 0.},
										{78.2, 81.0, -10.}};


const std::vector<std::vector<double>> TransCases  {{0,0,0},
			   				   {10, 0, 0},
								{10, -10, 0},
								{10,-100,100}};


int main()
{
	TGeoManager *geom = new TGeoManager("","");
	TGeoVolume * vol = geom->MakeBox("abox",0,15,20,25);
	TGeoShape *	box=vol->GetShape();
	box->InspectShape();

	BoxParameters *bp = new BoxParameters(15,20,25);

	// particle position and particle direction
	//Vector3D pPos( -90 , 0 , 0 );
	//Vector3D pDir(1,0,0); // fly towards origin
	Vector3D pPos( -90 , -90 , -90 );
	Vector3D zeroPos(0.,0.,0.);
	Vector3D pDir(1./sqrt(3), 1./sqrt(3), 1./sqrt(3)); // fly towards origin

	// testparticle is in universe (in some corner and flies towards the origin)

	// create root shape and placement
	for( int r=0; r< EulerAngles.size(); ++r ) // rotation cases
		for( int t=0; t<TransCases.size(); ++t ) // translation cases
		{
			// also try the other approach with a Node placement
			TGeoMatrix * matrix = new TGeoCombiTrans(TransCases[t][0], TransCases[t][1], TransCases[t][2],
								  	  	  	  	  	  new TGeoRotation("rot1", EulerAngles[r][0], EulerAngles[r][1], EulerAngles[r][2]));

			// transform coordinates
			Vector3D pPos2;
			Vector3D pDir2;
			//
			matrix->MasterToLocal(&pPos.x, &pPos2.x);
			matrix->MasterToLocalVect(&pDir.x, &pDir2.x);
			double dRoot = box->DistFromOutside(&pPos2.x, &pDir2.x, 3, 1E30, 0 );

			PhysicalVolume *fastbox = GeoManager::MakePlacedBox(bp,
								new TransformationMatrix(TransCases[t][0], TransCases[t][1], TransCases[t][2],
														 EulerAngles[r][0], EulerAngles[r][1], EulerAngles[r][2])
								);
			double dFastGeom = fastbox->DistanceToIn(pPos, pDir,1E30);

			std::cout << r << "\t" << t << "\t" << dRoot << "\t" << dFastGeom << std::endl;
			std::cout << fastbox->Contains(zeroPos) << " " << fastbox->Contains(pPos) << std::endl;

			// delete pointer
			delete matrix;
			delete fastbox;
		}
	return 1;
}
