/*
 * TransformationMatrix.cpp
 *
 *  Created on: Nov 14, 2013
 *      Author: swenzel
 */
#include "TransformationMatrix.h"
#include <cmath>

void TransformationMatrix::setAngles( double phi, double theta, double psi )
{
	//	Set matrix elements according to Euler angles
	double degrad = M_PI/180.;
	double sinphi = sin(degrad*phi);
	double cosphi = cos(degrad*phi);
	double sinthe = sin(degrad*theta);
	double costhe = cos(degrad*theta);
	double sinpsi = sin(degrad*psi);
	double cospsi = cos(degrad*psi);

	rot[0] =  cospsi*cosphi - costhe*sinphi*sinpsi;
	rot[1] = -sinpsi*cosphi - costhe*sinphi*cospsi;
	rot[2] =  sinthe*sinphi;
	rot[3] =  cospsi*sinphi + costhe*cosphi*sinpsi;
	rot[4] = -sinpsi*sinphi + costhe*cosphi*cospsi;
	rot[5] = -sinthe*cosphi;
	rot[6] =  sinpsi*sinthe;
	rot[7] =  cospsi*sinthe;
	rot[8] =  costhe;
}



//void TransformationMatrix :
