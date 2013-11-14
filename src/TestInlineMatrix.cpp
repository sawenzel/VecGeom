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

#include "Vc/vector.h"

int main()
{
	Vector3D x,dir;
	//double r[9]={1,2,3,4,5,6,7,8,9};
	double r[9] = {0,0,0,0,0,0,0,0,0};
	double t[3] = {100,1,1};

	x.x=20;//+rand();
	x.y=0; //rand();
	x.z=0; //rand();

	dir.x=-1.;
	dir.y=0.;
	dir.z=0.;

	// a box placed and rotated 45 degree around z-axis
	TransformationMatrix *tm = new TransformationMatrix(t[0],t[1],t[2],-45.,0.,0.);
	PhysicalVolume * box2 = GeoManager::MakePlacedBox(new BoxParameters(10,10,10), tm);

	volatile double c=box2->DistanceToIn(x,dir,1E30);

	Vc::double_v ax(1.);
	Vc::double_v ay(1.);
	Vc::double_v az(1.);

	Vc::double_v bx(1.);
	Vc::double_v by(1.);
	Vc::double_v bz(1.);

	tm->MasterToLocal<1,-1>(ax,ay,ax,bx,by,bz);

	std::cerr << bx.sum()+by.sum()+bz.sum() << std::endl;

	std::cerr << c << std::endl;

	return 1;
}



