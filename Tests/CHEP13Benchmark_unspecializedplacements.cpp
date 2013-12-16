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

	double *distances1 = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
	double *distances2 = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
	double *distances3 = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
	double *distances4 = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
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
	PhysicalVolume * world = GeoManager::MakePlacedTube( new TubeParameters<>(worldrmin, worldrmax, worldz, 0, 2.*M_PI), identity, false );
	PhysicalVolume * beampipe = GeoManager::MakePlacedTube( new TubeParameters<>(worldrmax/40., worldrmax/20., worldz), identity, false );
	world->AddDaughter( beampipe );
	BoxParameters * plateparams = new BoxParameters(30,5.,2.*worldz/3.);

	PhysicalVolume * plate1 = GeoManager::MakePlacedBox( plateparams, new TransformationMatrix(50, 0, 0, 35, 0, 10), false );
	PhysicalVolume * plate2 = GeoManager::MakePlacedBox( plateparams, new TransformationMatrix(-50, 0, 0, 35, 0, 10), false );
	PhysicalVolume * plate3 = GeoManager::MakePlacedBox( plateparams, new TransformationMatrix(0, 50, 0, -35, 0, 10), false );
	PhysicalVolume * plate4 = GeoManager::MakePlacedBox( plateparams, new TransformationMatrix(0, -50, 0, -35, 0, 10), false );
	//PhysicalVolume * plate1 = GeoManager::MakePlacedBox( plateparams, TransformationMatrix::createSpecializedMatrix(50, 0, 0, 35, 0, 10) );
	//PhysicalVolume * plate2 = GeoManager::MakePlacedBox( plateparams, TransformationMatrix::createSpecializedMatrix(-50, 0, 0, 35, 0, 10) );
	//PhysicalVolume * plate3 = GeoManager::MakePlacedBox( plateparams, TransformationMatrix::createSpecializedMatrix(0, 50, 0, -35, 0, 10) );
	//PhysicalVolume * plate4 = GeoManager::MakePlacedBox( plateparams, TransformationMatrix::createSpecializedMatrix(0, -50, 0, -35, 0, 10) );


	world->AddDaughter( plate1 );
	world->AddDaughter( plate2 );
	world->AddDaughter( plate3 );
	world->AddDaughter( plate4 );

	PhysicalVolume * shield = GeoManager::MakePlacedTube( new TubeParameters<>(9*worldrmax/11, 9*worldrmax/10, 8*worldz/10), identity, false );
	world->AddDaughter( shield );

	ConeParameters<double> * endcapparams = new ConeParameters<double>( worldrmax/20., worldrmax,
					worldrmax/20., worldrmax/10., worldz/10., 0, 2.*M_PI );
	PhysicalVolume * endcap1 = GeoManager::MakePlacedCone( endcapparams, new TransformationMatrix(0,0,-9.*worldz/10., 0, 0, 0), false );
	PhysicalVolume * endcap2 = GeoManager::MakePlacedCone( endcapparams, new TransformationMatrix(0,0,9*worldz/10, 0, 180, 0), false );
	//world->AddDaughter( endcap1 );
	//world->AddDaughter( endcap2 );

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
		vecnav.DistToNextBoundary( world, points, dirs, steps, distances1, nextvolumes , np );
	}
	timer.Stop();
	double t0 = timer.getDeltaSecs();
	std::cerr << t0 << std::endl;
	// give out hit pointers
	double d0=0.;
	for(auto k=0;k<np;k++)
	{
		d0+=distances1[k];
	}


	timer.Start();
	for(int reps=0 ;reps < NREPS; reps++ )
	{
		vecnav.DistToNextBoundaryUsingUnplacedVolumes( world, points, dirs, steps, distances2, nextvolumes , np );
	}
	timer.Stop();
	double t1= timer.getDeltaSecs();

	std::cerr << t1 << std::endl;

	double d1=0.;
	for(auto k=0;k<np;k++)
	{
		d1+=distances2[k];
	}

	// now using the ROOT Geometry library (scalar version)
	timer.Start();
	for(int reps=0;reps < NREPS; reps ++ )
	{
		vecnav.DistToNextBoundaryUsingROOT( world, plainpointarray, plaindirtarray, steps, distances3, nextvolumes, np );
	}
	timer.Stop();
	double t3 = timer.getDeltaSecs();

	std::cerr << t3 << std::endl;
	double d3=0.;
	for(auto k=0;k<np;k++)
		{
			d3+=distances3[k];
		}

	timer.Start();
	for(int reps=0;reps < NREPS; reps ++ )
	{
		vecnav.DistToNextBoundaryUsingUSOLIDS( world, points, dirs, steps, distances4, nextvolumes, np );
	}
	timer.Stop();
	double t4 = timer.getDeltaSecs();

	std::cerr << t4 << std::endl;
	double d4=0.;
	for(auto k=0;k<np;k++)
	{
		d4+=distances4[k];
	}

	std::cerr << d0 << " " << d1 << " " << d3 << " " << d4 << std::endl;

	for(auto k=0;k<np;k++)
	{
//		if(std::abs(distances3[k]-distances2[k])>1e-9)
		{
			std::cerr << k << " " << distances1[k] << " " << distances2[k] << " " << distances3[k] << " " << " " << distances4[k] << " " << nextvolumes << std::endl;
			world->PrintDistToEachDaughter( points.getAsVector(k), dirs.getAsVector(k) );
			world->PrintDistToEachDaughterROOT( points.getAsVector(k), dirs.getAsVector(k) );
			world->PrintDistToEachDaughterUSOLID( points.getAsVector(k), dirs.getAsVector(k) );
		}
	}


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
	_mm_free(distances1);
	_mm_free(distances2);
	_mm_free(distances3);
	return 1;
}
