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
	Vectors3DSOA points, dirs, intermediatepoints, intermediatedirs;
	StructOfCoord rpoints, rintermediatepoints, rdirs, rintermediatedirs;


	int np=1024;
	int NREPS = 1000;

	points.alloc(np);
	dirs.alloc(np);
	intermediatepoints.alloc(np);
	intermediatedirs.alloc(np);

	rpoints.alloc(np);
	rdirs.alloc(np);
	rintermediatepoints.alloc(np);
	rintermediatedirs.alloc(np);

	double *distances = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
	double *distancesROOTSCALAR = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
	double *distancesUSOLIDSCALAR = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
	double *distances2 = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
	double *steps = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
	for(auto i=0;i<np;++i) steps[i] = Utils::kInfinity;

	double *plainpointarray = (double *) _mm_malloc(3*np*sizeof(double), ALIGNMENT_BOUNDARY);
	double *plaindirtarray = (double *) _mm_malloc(3*np*sizeof(double), ALIGNMENT_BOUNDARY);


	StopWatch timer;

    // generate benchmark cases
	TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);

	// the world volume is a tube
	double worldrmax = 100.;
	double worldrmin = 0.;
	double worldz = 200.;
	PhysicalVolume * world = GeoManager::MakePlacedTube( new TubeParameters<>(worldrmin, worldrmax, worldz, 0, 2.*M_PI), identity );
	PhysicalVolume * beampipe = GeoManager::MakePlacedTube( new TubeParameters<>(worldrmax/40., worldrmax/20., worldz), identity );
	world->AddDaughter( beampipe );

	BoxParameters * plateparams = new BoxParameters(30,5.,2.*worldz/3.);
	PhysicalVolume * plate1 = GeoManager::MakePlacedBox( plateparams, new TransformationMatrix(50, 0, 0, 35, 0, 10) );
	PhysicalVolume * plate2 = GeoManager::MakePlacedBox( plateparams, new TransformationMatrix(-50, 0, 0, 35, 0, 10) );
	PhysicalVolume * plate3 = GeoManager::MakePlacedBox( plateparams, new TransformationMatrix(0, 50, 0, -35, 0, 10) );
	PhysicalVolume * plate4 = GeoManager::MakePlacedBox( plateparams, new TransformationMatrix(0, -50, 0, -35, 0, 10) );
	//PhysicalVolume * plate1 = GeoManager::MakePlacedBox( plateparams, TransformationMatrix::createSpecializedMatrix(50, 0, 0, 35, 0, 10) );
	//PhysicalVolume * plate2 = GeoManager::MakePlacedBox( plateparams, TransformationMatrix::createSpecializedMatrix(-50, 0, 0, 35, 0, 10) );
	//PhysicalVolume * plate3 = GeoManager::MakePlacedBox( plateparams, TransformationMatrix::createSpecializedMatrix(0, 50, 0, -35, 0, 10) );
	//PhysicalVolume * plate4 = GeoManager::MakePlacedBox( plateparams, TransformationMatrix::createSpecializedMatrix(0, -50, 0, -35, 0, 10) );
	world->AddDaughter( plate1 );
	world->AddDaughter( plate2 );
	world->AddDaughter( plate3 );
	world->AddDaughter( plate4 );

	PhysicalVolume * shield = GeoManager::MakePlacedTube( new TubeParameters<>(9*worldrmax/11, 9*worldrmax/10, 8*worldz/10), identity );
	world->AddDaughter( shield );

	ConeParameters<double> * endcapparams = new ConeParameters<double>( worldrmax/20., worldrmax,
					worldrmax/20., worldrmax/10., worldz/10., 0, 2.*M_PI );
	PhysicalVolume * endcap1 = GeoManager::MakePlacedCone( endcapparams, new TransformationMatrix(0,0,-9.*worldz/10., 0, 0, 0) );
	PhysicalVolume * endcap2 = GeoManager::MakePlacedCone( endcapparams, new TransformationMatrix(0,0,9*worldz/10, 0, 180, 0) );
	world->AddDaughter( endcap1 );
	world->AddDaughter( endcap2 );

	world->fillWithRandomPoints(points,np);
	world->fillWithBiasedDirections(points, dirs, np, 9/10.);

	points.toPlainArray(plainpointarray,np);
	dirs.toPlainArray(plaindirtarray,np);

	std::cerr << " Number of daughters " << world->GetNumberOfDaughters() << std::endl;

	// time performance for this placement ( we should probably include some random physical steps )

	// do some navigation with a simple Navigator
	SimpleVecNavigator vecnav(np);
	PhysicalVolume ** nextvolumes  = ( PhysicalVolume ** ) _mm_malloc(sizeof(PhysicalVolume *)*np, ALIGNMENT_BOUNDARY);

	timer.Start();
	for(int reps=0 ;reps < NREPS; reps++ )
	{
		vecnav.DistToNextBoundary( world, points, dirs, steps, distances, nextvolumes , np );
	}
	timer.Stop();
	double t0 = timer.getDeltaSecs();
	std::cerr << t0 << std::endl;
	// give out hit pointers
	double d0=0.;
	for(auto k=0;k<np;k++)
	{
		d0+=distances[k];
		distances[k]=Utils::kInfinity;
	}


	timer.Start();
	for(int reps=0 ;reps < NREPS; reps++ )
	{
		vecnav.DistToNextBoundaryUsingUnplacedVolumes( world, points, dirs, steps, distances, nextvolumes , np );
	}
	timer.Stop();
	double t1= timer.getDeltaSecs();

	std::cerr << t1 << std::endl;

	double d1;
	for(auto k=0;k<np;k++)
	{
		d1+=distances[k];
		distances[k]=Utils::kInfinity;

	}

	// now using the ROOT Geometry library (scalar version)
	timer.Start();
	for(int reps=0;reps < NREPS; reps ++ )
	{
		vecnav.DistToNextBoundaryUsingROOT( world, plainpointarray, plaindirtarray, steps, distances, nextvolumes, np );
	}
	timer.Stop();
	double t3 = timer.getDeltaSecs();

	std::cerr << t3 << std::endl;
	double d3;
	for(auto k=0;k<np;k++)
		{
			d3+=distances[k];
		}
	std::cerr << d0 << " " << d1 << " " << d3 << std::endl;


	//vecnav.DistToNextBoundaryUsingUnplacedVolumes( world, points, dirs, steps, distances, nextvolumes , np );
	//( world, points, dirs,  );


	// give out hit pointers
	/*
	for(auto k=0;k<np;k++)
	{
		if( nextvolumes[k] !=0 )
		{
			nextvolumes[k]->printInfo();
		}
		else
		{
			std::cerr << "hitting boundary of world"  << std::endl;
		}
	}
*/
    _mm_free(distances);
    return 1;
}
