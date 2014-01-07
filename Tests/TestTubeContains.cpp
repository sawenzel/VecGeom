/*
 * TestVectorizedPlacedTube.cpp
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

	int np = 1024;
	int NREPS = 1000;

	points.alloc(np);
	intermediatepoints.alloc(np);

	rpoints.alloc(np);
	rintermediatepoints.alloc(np);

	double *plainpointarray = (double *) _mm_malloc(3*np*sizeof(double), ALIGNMENT_BOUNDARY);

	GlobalTypes::SurfaceEnumType * results = ( GlobalTypes::SurfaceEnumType * ) malloc( sizeof(GlobalTypes::SurfaceEnumType)*np );
	VUSolid::EnumInside * resultsUsolid = ( VUSolid::EnumInside * ) malloc( sizeof(VUSolid::EnumInside)*np );

	StopWatch timer;

    // generate benchmark cases
	TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);

	// the world volume is a tube
	double worldrmax = 100.;
	double worldrmin = 0.;
	double worldz = 200.;
	PhysicalVolume * world = GeoManager::MakePlacedTube( new TubeParameters<>(worldrmin, worldrmax, worldz, 0, 2.*M_PI), identity );

	PhysicalVolume * shield = GeoManager::MakePlacedTube( new TubeParameters<>(0, 5*worldrmax/10, 8*worldz/10, 0, 2. * M_PI ), identity );

	// fill points with no daughter inside; then place daughter
	world->fillWithRandomPoints(points,np);

	world->AddDaughter( shield );

	points.toPlainArray(plainpointarray,np);


	timer.Start();
	for(int reps=0 ;reps < NREPS; reps++ )
	{
		for(auto particle = 0; particle < np; ++particle )
			results[particle] = shield->Contains_WithSurface( points.getAsVector(particle) );
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
				resultsUsolid[particle] = shield->GetAsUnplacedUSolid()->Inside( reinterpret_cast<UVector3 const &>( pos  ) );
			}
		}
	timer.Stop();
	double t1 = timer.getDeltaSecs();
	std::cerr << t1 << std::endl;

	for( auto particle = 0; particle < np; ++particle )
	{
		std::cerr << results[particle] << " " << resultsUsolid[particle] << std::endl;
	}

    return 1;
}
