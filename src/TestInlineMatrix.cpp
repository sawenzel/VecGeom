/*
 * TestInlineMatrix.cpp
 *
 *  Created on: Nov 11, 2013
 *      Author: swenzel
 */

#include "../TransformationMatrix.h"
#include "../Vector3D.h"
#include "../PhysicalBox.h"
#include "../GeoManager.h"
#include <cstdlib>
#include <iostream>

int main()
{
	Vector3D x,dir;
	//double r[9]={1,2,3,4,5,6,7,8,9};

	double r[9]={0,0,0,0,0,0,0,0,0};

	double t[3]={100,1,1};

	x.x=20;//+rand();
	x.y=0; //rand();
	x.z=0; //rand();

	dir.x=-1.;
	dir.y=0.;
	dir.z=0.;


	PhysicalVolume * box2 = GeoManager::MakePlacedBox(new BoxParameters(10,10,10), new TransformationMatrix(t,r));

	volatile double c=box2->DistanceToIn(x,dir,1E30);
	std::cerr << c << std::endl;

	return 1;
}



