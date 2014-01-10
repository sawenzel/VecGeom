// testing matrix transformations on vectors
#include "../TransformationMatrix.h"
#include "../Utils.h"
#include <iostream>
#include "mm_malloc.h"
#include "../GlobalDefs.h"
#include "../GeoManager.h"
#include "../PhysicalPolycone.h"
#include "../TestShapeContainer.h"

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

	int np = 1024000;
	int NREPS = 10;

	points.alloc(np);
	intermediatepoints.alloc(np);

	rpoints.alloc(np);
	rintermediatepoints.alloc(np);

	double *plainpointarray = (double *) _mm_malloc(3*np*sizeof(double), ALIGNMENT_BOUNDARY);

	GlobalTypes::SurfaceEnumType * results = ( GlobalTypes::SurfaceEnumType * ) malloc( sizeof(GlobalTypes::SurfaceEnumType)*np );
	VUSolid::EnumInside * resultsUsolid = ( VUSolid::EnumInside * ) malloc( sizeof(VUSolid::EnumInside)*np );

	StopWatch timer;


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

	/*
		// a medium polycone
		double zplanes[] = {-90, -70, -50, -30, -10, 0, 10, 30, 50, 70, 90 };
		double rinner[] =  { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };
		double router[] =  { 70, 70, 100, 100, 70, 70, 70, 100, 100, 70, 70 };
		const int Nz=11;
	*/

	/*
	// a large polycone
		const int Nz=101;
		for(auto i = 0;i< Nz; i++)
		{
			zplanes[i]=-90 + i*(180./(Nz-1));
			rinner[i]=5;
			if(i%4==0) router[i]=70;
		 	if(i%4==1) router[i]=70;
		 	if(i%4==2) router[i]=100;
		 	if(i%4==3) router[i]=100;
		}
    */

	/*
	// a minimal polycone
	double zplanes[] = {-80, 80 };
	double rinner[] = { 0, 10, };
	double router[] = { 50, 80, };
	const int Nz=2;
	 */

	PolyconeParameters<> * pconp = new PolyconeParameters<>( 0, 2.*M_PI, Nz, zplanes, rinner, router );
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
				resultsUsolid[particle] = pcon->GetAsUnplacedUSolid()->Inside( reinterpret_cast<UVector3 const &>( pos  ) );
			}
		}
	timer.Stop();
	double t1 = timer.getDeltaSecs();
	std::cerr << t1 << std::endl;

	int counter[3]={0,0,0};
	int counterUSolid[3]={0,0,0};
	for( auto particle=0; particle<np; ++particle )
	{
		counter[results[particle]]++;
		counterUSolid[resultsUsolid[particle]]++;
	}
	for( auto j=0; j < 3; j++)
	{
		std::cerr << counter[j] << " " << counterUSolid[j] << std::endl;

	}

    return 1;
}
