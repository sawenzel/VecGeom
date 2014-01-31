/*
 * TransformationMatrix.cpp
 *
 *  Created on: Nov 14, 2013
 *      Author: swenzel
 */
#include "TransformationMatrix.h"
#include <cmath>
#include <cstdio>

#include "TGeoMatrix.h"

// constructor
TransformationMatrix::TransformationMatrix(double const *t, double const *r) : rootmatrix(0)
{
	trans[0]=t[0];
	trans[1]=t[1];
	trans[2]=t[2];
	for(auto i=0;i<9;i++)
		rot[i]=r[i];
	// we need to check more stuff ( for instance that product along diagonal is +1)
	setProperties();

}

// more general constructor ala ROOT ( with Euler Angles )
TransformationMatrix::TransformationMatrix(double tx, double ty, double tz, double phi, double theta, double psi)
{
	trans[0]=tx;
	trans[1]=ty;
	trans[2]=tz;
	setAngles(phi, theta, psi);
	setProperties();

	InitEquivalentTGeoMatrix(phi, theta, psi);
}


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

// function that returns the translation and rotation type of a transformation specified by tx,ty,tz and Euler angles
void
TransformationMatrix::classifyMatrix( double tx, double ty, double tz, double phi, double theta, double psi, int & tid, int & rid)
{
	//	Set matrix elements according to Euler angles
		double degrad = M_PI/180.;
		double sinphi = sin(degrad*phi);
		double cosphi = cos(degrad*phi);
		double sinthe = sin(degrad*theta);
		double costhe = cos(degrad*theta);
		double sinpsi = sin(degrad*psi);
		double cospsi = cos(degrad*psi);

		double rotm[9];
		rotm[0] =  cospsi*cosphi - costhe*sinphi*sinpsi;
		rotm[1] = -sinpsi*cosphi - costhe*sinphi*cospsi;
		rotm[2] =  sinthe*sinphi;
		rotm[3] =  cospsi*sinphi + costhe*cosphi*sinpsi;
		rotm[4] = -sinpsi*sinphi + costhe*cosphi*cospsi;
		rotm[5] = -sinthe*cosphi;
		rotm[6] =  sinpsi*sinthe;
		rotm[7] =  cospsi*sinthe;
		rotm[8] =  costhe;

		double transtmp[3];
		transtmp[0]=tx;
		transtmp[1]=ty;
		transtmp[2]=tz;

		rid=TransformationMatrix::getRotationFootprintS(rotm);
		tid=TransformationMatrix::GetTranslationIdTypeS(transtmp);
}


void TransformationMatrix::setProperties()
{
	hasTranslation = (this->GetTranslationIdType() == 0)? false : true;
	hasRotation = (this->getRotationFootprint() == 1296)? false : true;
	identity = !hasTranslation && !hasRotation;
}


void TransformationMatrix::print() const
{
// print the matrix in 4x4 format
   printf("matrix %p - tr=%d  rot=%d \n", this, (int)hasTranslation, (int)hasRotation);
   printf("number of rotation zeros: %d and type %d\n", this->getNumberOfZeroEntries(), this->getRotationFootprint());
   printf("%12.11f\t%12.11f\t%12.11f    Tx = %10.6f\n", rot[0], rot[1], rot[2], trans[0]);
   printf("%12.11f\t%12.11f\t%12.11f    Ty = %10.6f\n", rot[3], rot[4], rot[5], trans[1]);
   printf("%12.11f\t%12.11f\t%12.11f    Tz = %10.6f\n", rot[6], rot[7], rot[8], trans[2]);
 }

unsigned int TransformationMatrix::getNumberOfZeroEntries() const
	{
		unsigned int count=0;
		for(auto i=0;i<9;++i)
			{
				if(std::fabs(rot[i]) < 1E-12)	count++;
			}
		return count;
	}

void TransformationMatrix::InitEquivalentTGeoMatrix(double phi, double theta, double psi)
{
	if( identity ){
		rootmatrix = new TGeoIdentity();
	}
	else if ( hasRotation && ! hasTranslation )
	{
		rootmatrix = new TGeoRotation("internalrot",phi, theta, psi);
	}
	else if ( hasTranslation && ! hasRotation )
	{
		rootmatrix = new TGeoTranslation(trans[0],trans[1],trans[2]);
	}
	else
	{
		rootmatrix = new TGeoCombiTrans(trans[0],trans[1],trans[2], new TGeoRotation("internalrot",phi,theta,psi));
	}
	// for the moment we are not treating the case of a scale matrix

}


TransformationMatrix const *
TransformationMatrix::createSpecializedMatrix( double tx, double ty, double tz, double phi, double theta, double psi  )
{
	int tid=0, rid=0;
	TransformationMatrix::classifyMatrix(tx,ty,tz,phi,theta,psi,tid,rid);

	// following piece of code is script generated
	if(tid == 0 && rid == 1296) return new SpecializedTransformation<0,1296>(tx,ty,tz,phi, theta,psi); // identity
	if(tid == 1 && rid == 1296) return new SpecializedTransformation<1,1296>(tx,ty,tz,phi,theta,psi); // only translation
	if(tid == 0 && rid == 252) return new SpecializedTransformation<0,252>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 252) return new SpecializedTransformation<1,252>(tx,ty,tz,phi,theta,psi);
	if(tid == 0 && rid == 405) return new SpecializedTransformation<0,405>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 405) return new SpecializedTransformation<1,405>(tx,ty,tz,phi,theta,psi);
	if(tid == 0 && rid == 882) return new SpecializedTransformation<0,882>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 882) return new SpecializedTransformation<1,882>(tx,ty,tz,phi,theta,psi);
	if(tid == 0 && rid == 415) return new SpecializedTransformation<0,415>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 415) return new SpecializedTransformation<1,415>(tx,ty,tz,phi,theta,psi);
	if(tid == 0 && rid == 496) return new SpecializedTransformation<0,496>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 496) return new SpecializedTransformation<1,496>(tx,ty,tz,phi,theta,psi);
	if(tid == 0 && rid == 793) return new SpecializedTransformation<0,793>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 793) return new SpecializedTransformation<1,793>(tx,ty,tz,phi,theta,psi);
	if(tid == 0 && rid == 638) return new SpecializedTransformation<0,638>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 638) return new SpecializedTransformation<1,638>(tx,ty,tz,phi,theta,psi);
	if(tid == 0 && rid == 611) return new SpecializedTransformation<0,611>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 611) return new SpecializedTransformation<1,611>(tx,ty,tz,phi,theta,psi);
	if(tid == 0 && rid == 692) return new SpecializedTransformation<0,692>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 692) return new SpecializedTransformation<1,692>(tx,ty,tz,phi,theta,psi);
	if(tid == 0 && rid == 720) return new SpecializedTransformation<0,720>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 720) return new SpecializedTransformation<1,720>(tx,ty,tz,phi,theta,psi);
	if(tid == 0 && rid == 828) return new SpecializedTransformation<0,828>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 828) return new SpecializedTransformation<1,828>(tx,ty,tz,phi,theta,psi);
	if(tid == 0 && rid == 756) return new SpecializedTransformation<0,756>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 756) return new SpecializedTransformation<1,756>(tx,ty,tz,phi,theta,psi);
	if(tid == 0 && rid == 918) return new SpecializedTransformation<0,918>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 918) return new SpecializedTransformation<1,918>(tx,ty,tz,phi,theta,psi);
	if(tid == 0 && rid == 954) return new SpecializedTransformation<0,954>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 954) return new SpecializedTransformation<1,954>(tx,ty,tz,phi,theta,psi);
	if(tid == 0 && rid == 1008) return new SpecializedTransformation<0,1008>(tx,ty,tz,phi,theta,psi);
	if(tid == 1 && rid == 1008) return new SpecializedTransformation<1,1008>(tx,ty,tz,phi,theta,psi);

	// fallback case
	return new SpecializedTransformation<1,-1>(tx,ty,tz,phi,theta,psi);
}

