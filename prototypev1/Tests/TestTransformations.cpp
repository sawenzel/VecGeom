/*
 * TestTransformations.cpp
 *
 *  Created on: Feb 4, 2014
 *      Author: swenzel
 */


#include "TGeoMatrix.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoShape.h"
#include "../TransformationMatrix.h"
#include <iostream>

int main()
{
	TGeoRotation * r1 = new TGeoRotation("rot1",30,0,0);
	TGeoRotation * r2 = new TGeoRotation("rot2",30,0,23.5);
	TGeoMatrix   * m1 = new TGeoCombiTrans("c1",0,10,10,r1);
	TGeoMatrix   * m2 = new TGeoCombiTrans("c2",20,20,0,r2);
	TGeoIdentity * i  = new TGeoIdentity();

	TGeoHMatrix * globalm = new TGeoHMatrix();

	globalm->CopyFrom( i );
	globalm->Multiply( m1 );
	globalm->Multiply( m2 );

	m1->Print();
	m2->Print();
	globalm->Print();

	// do some transformation tests;
	double testpoint[3] = {1.23, -5, -4.5};
	double intermediate[3], final[3];
	double cmp[3];

	m1->MasterToLocal(testpoint, intermediate);
	m2->MasterToLocal(intermediate, final);

	globalm->LocalToMaster(final, cmp);

	std::cerr << testpoint[0] << " " << testpoint[1] << " " << testpoint[2] << std::endl;
	std::cerr << final[0] << " " << final[1] << " " << final[2] << std::endl;
	std::cerr << cmp[0] << " " << cmp[1] << " " << cmp[2] << std::endl;


	// some with new stuff
	TransformationMatrix const * m1_new = new TransformationMatrix(0,10,10,30,0,0);
	TransformationMatrix const * m2_new = new TransformationMatrix(20,20,0,30,0,23.5);

	TransformationMatrix * global_new = new TransformationMatrix(0,0,0,0,0,0);
	global_new->Multiply( m1_new );
	global_new->Multiply( m2_new );

	Vector3D testpoint_new(1.23,-5,-4.5);
	Vector3D intermediate_new(0,0,0);
	Vector3D final_new(0,0,0);
	Vector3D cmp_new(0,0,0);

	m1_new->MasterToLocal<1,-1>(testpoint_new, intermediate_new);
	m2_new->MasterToLocal<1,-1>(intermediate_new, final_new);
	testpoint_new.print();
	final_new.print();

	global_new->print();
	global_new->LocalToMaster(final_new, cmp_new);
	cmp_new.print();

	return 1;
}
