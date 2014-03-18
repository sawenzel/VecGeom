/*
 * TestPolyconeContains.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: swenzel
 */

// testing matrix transformations on vectors
#include "../TransformationMatrix.h"
#include "TGeoMatrix.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoShape.h"
#include "../Utils.h"
#include <iostream>
#include "mm_malloc.h"
#include "../GlobalDefs.h"
#include "../GeoManager.h"
#include "../PhysicalTube.h"
#include "../TestShapeContainer.h"
#include "../SimpleVecNavigator.h"

// in order to compare to USolids
#include "VUSolid.hh"
#include "UTubs.hh"


static void cmpresults(double * a1, double * a2, int np,
		PhysicalVolume const * vol, std::vector<Vector3D> const & points, std::vector<Vector3D> const & dirs)
{
	int counter=0;
	for( auto i=0; i<np; ++i )
	{
		if( std::abs( a1[i] - a2[i] ) > Utils::GetCarTolerance() )
		{
			counter++;
#ifdef SHOWDIFFERENCES
			std::cerr << i << " " << a1[i] << " " << a2[i] << std::endl;
			vol->DebugPointAndDirDistanceToIn( points[i], dirs[i] );
#endif
		}
	}
	std::cerr << " have " << counter << " differences " << std::endl;
}


int main()
{
	Vectors3DSOA points, intermediatepoints;
	StructOfCoord rpoints, rintermediatepoints;

	int np = 102400;
	int NREPS = 1000;

	points.alloc(np);
	intermediatepoints.alloc(np);

	rpoints.alloc(np);
	rintermediatepoints.alloc(np);

	double *plainpointarray = (double *) _mm_malloc(3*np*sizeof(double), ALIGNMENT_BOUNDARY);

	GlobalTypes::SurfaceEnumType * results = (GlobalTypes::SurfaceEnumType * ) malloc( sizeof(GlobalTypes::SurfaceEnumType)*np);
	VUSolid::EnumInside * resultsUsolid = (VUSolid::EnumInside * ) malloc( sizeof(VUSolid::EnumInside)*np);
	bool * resultsROOT = (bool *) malloc(sizeof(bool)*np);

	int RN=100000;
	StopWatch timer;
	double * randomnumber = (double*) malloc(sizeof(double)*RN);

    // generate benchmark cases
	TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);

	// the world volume is a tube
	double worldrmax = 100.;
	double worldrmin = 0.;
	double worldz = 200.;
	PhysicalVolume * world = GeoManager::MakePlacedTube( new TubeParameters<>(worldrmin, worldrmax, worldz, 0, M_PI ), IdentityTransformationMatrix );

	/*
		// a small polycone
		double zplanes[] = {-80, -20, 20, 80 };
		double rinner[] = { 0, 0, 0, 0 };
		double router[] = { 70, 70, 100, 100 };
		int Nz=4;
	*/


	// a medium polycone

/*
	double zplanes[] = {-90, -70, -50, -30, -10, 0, 10, 30, 50, 70, 90 };
	double rinner[] =  { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };
	double router[] =  { 70, 70, 100, 100, 70, 70, 70, 100, 100, 70, 70 };
	const int Nz=11;
*/

	// a large polycone
//	for(auto k=5;k<100;k+=5)
//	{
		const int Nz=10;
		double zplanes[Nz];
		double rinner[Nz];
		double router[Nz];
		double scale = 100;
		for(auto i = 0;i< Nz; i++)
		{
			zplanes[i]=-90 + i*(180./(Nz-1));
			rinner[i]=0.05*scale;
			if(i%4==0) router[i]=scale*0.7;
			if(i%4==1) router[i]=scale*0.7;
			if(i%4==2) router[i]=scale;
			if(i%4==3) router[i]=scale;
		}

		for(auto k=0;k<RN;k++)
			randomnumber[k] = zplanes[0] + (zplanes[Nz-1]-zplanes[0])/RN *k;

		/*
		// a minimal polycone
		double zplanes[] = {-80, 80 };
		double rinner[] = { 0, 10, };
		double router[] = { 50, 80, };
		const int Nz=2;
		 */

		PolyconeParameters<> * pconp = new PolyconeParameters<>( 0, M_PI, Nz, zplanes, rinner, router );

		/*
		timer.Start();
		int s=0;
		for(auto i=0;i<RN;i++){
			s+=pconp->FindZSectionDaniel( randomnumber[i] );}
		timer.Stop();
		double t1=timer.getDeltaSecs();
		//std::cerr << k << " " << t1 << std::endl;

		timer.Start();
		int s2=0;
		for(auto i=0;i<RN;i++){
			s2+=pconp->FindZSection( randomnumber[i] );}
		timer.Stop();
		double t2=timer.getDeltaSecs();
		//std::cerr << k << " " << t2 << std::endl;

		timer.Start();
		int s3=0;
		for(auto i=0;i<RN;i++){
			s3+=pconp->FindZSectionBS( randomnumber[i] );}
		timer.Stop();
		double t3=timer.getDeltaSecs();
		std::cerr << k << " " << t1 << " " << t2 << " " << t3 << " " << s << " " << s2 << " " << s3 << std::endl;
		std::cerr << pconp->FindZSectionDaniel( 11.34 ) << std::endl;
		std::cerr << pconp->FindZSection( 11.34 ) << std::endl;
	*/
	//}
	//exit(1);

	//PolyconeParameters<> * pconp;
	PhysicalVolume * pcon = GeoManager::MakePlacedPolycone(pconp, IdentityTransformationMatrix, true );

	// fill points with no daughter inside; then place daughter
	world->fillWithRandomPoints(points,np);
	world->AddDaughter( pcon );
	points.toPlainArray(plainpointarray,np);

	timer.Start();
	for(int reps=0 ;reps < NREPS; reps++ )
	{
		for( auto particle = 0; particle < np; ++particle )
			results[particle] = pcon->Contains_WithSurface( points.getAsVector(particle) );
	}
	timer.Stop();
	double t0 = timer.getDeltaSecs();
	std::cerr << t0 << std::endl;

	timer.Start();
	for( int reps=0 ;reps < NREPS; reps++ )
	{
		for( auto particle = 0; particle < np; ++particle )
			{
				Vector3D pos = points.getAsVector(particle);
				// resultsUsolid[particle] = pcon->GetAsUnplacedUSolid()->Inside( reinterpret_cast<UVector3 const &>( pos ));
				resultsUsolid[particle] = pcon->GetAsUnplacedUSolid()->Inside( (UVector3 const &) pos );
			}
		}
	timer.Stop();
	double t1 = timer.getDeltaSecs();
	std::cerr << t1 << std::endl;

	timer.Start();
	for( int reps=0 ;reps < NREPS; reps++ )
	{
		for( auto particle = 0; particle < np; ++particle )
		{
			Vector3D pos = points.getAsVector(particle);
			resultsROOT[particle] = pcon->GetAsUnplacedROOTSolid()->Contains( &pos.x );
		}
	}
	timer.Stop();
	double t2 = timer.getDeltaSecs();
	std::cerr << t2 << std::endl;


	int counter[3]={0,0,0};
	int counterUSolid[3]={0,0,0};
	int counterROOT[3]={0,0,0};
	for( auto particle=0; particle<np; ++particle )
	{
	  counter[results[particle]]++;
	  counterUSolid[resultsUsolid[particle]]++;
	  counterROOT[(resultsROOT[particle])? 0 : 2]++;
	}
	for( auto j=0; j < 3; j++)
	{
	  std::cerr << counter[j] << " " << counterUSolid[j] << " " << counterROOT[j] << std::endl;
	}

	return 1;

}
