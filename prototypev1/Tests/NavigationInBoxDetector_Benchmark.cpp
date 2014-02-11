/*
 * BuildBoxDetector.cpp
 *
 *  Created on: Feb 3, 2014
 *      Author: swenzel
 */

#include "../TransformationMatrix.h"
#include "../Utils.h"
#include <iostream>
#include "mm_malloc.h"
#include "../GlobalDefs.h"
#include "../GeoManager.h"
#include "../PhysicalBox.h"
#include "../SimpleVecNavigator.h"
#include <map>
#include <cassert>


void foo( double s, Vector3D const &a, Vector3D const &b, Vector3D &c )
{
	c=a+s*b;
}

int main()
{
	Vectors3DSOA points, dirs, intermediatepoints, intermediatedirs;

	int np=1024;
	int NREPS = 1000;

	points.alloc(np);
	dirs.alloc(np);

	// generate benchmark cases
	TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);

	double L = 10.;
	double Lz = 10.;
	const double Sqrt2 = sqrt(2.);

	BoxParameters *	worldp =  new BoxParameters(L, L, Lz );
	PhysicalVolume * world = GeoManager::MakePlacedBox( worldp , identity );
	double volworld = worldp->GetVolume();

	BoxParameters * boxlevel2 = new BoxParameters( Sqrt2*L/2./2., Sqrt2*L/2./2., Lz );
	BoxParameters * boxlevel3 = new BoxParameters( L/2./2. ,L/2./2., Lz);
	BoxParameters * boxlevel1 = new BoxParameters( L/2., L/2., Lz );

	PhysicalVolume * box2 = GeoManager::MakePlacedBox(boxlevel2, new TransformationMatrix(0,0,0,0,0,45));
	PhysicalVolume * box3 = GeoManager::MakePlacedBox( boxlevel3, new TransformationMatrix(0,0,0,0,0,-45));
	box2->AddDaughter( box3 ); // rotated 45 degree around z axis

	PhysicalVolume * box1 = GeoManager::MakePlacedBox(boxlevel1, identity);
	box1->AddDaughter( box2 );

	PhysicalVolume * box1left  = world->PlaceDaughter(GeoManager::MakePlacedBox(boxlevel1, new TransformationMatrix(-L/2.,0.,0.,0.,0.,0)), box1->GetDaughterList());
	PhysicalVolume * box1right = world->PlaceDaughter(GeoManager::MakePlacedBox(boxlevel1, new TransformationMatrix(+L/2.,0.,0.,0.,0.,0)), box1->GetDaughterList());


    // perform basic tests
	SimpleVecNavigator nav(1, world);

	StopWatch timer;
	// some object which are expensive to create
	TransformationMatrix *m = new TransformationMatrix();
	VolumePath path(4);
	VolumePath newpath(4);

	timer.Start();
	int stepsdone=0;
	for(int i=0;i<100000;i++)
	// testing the NavigationAndStepInterface
	{
		Vector3D p;
		PhysicalVolume::samplePoint( p, worldp->GetDX(), worldp->GetDY(), worldp->GetDZ(), 1. );
		// setup point in world
		Vector3D d(1,0,0);
		Vector3D resultpoint;

		m->SetToIdentity();
		PhysicalVolume const * vol;
		vol = nav.LocatePoint( world, p, resultpoint, path, m );
		while( vol!=NULL )
		{
			stepsdone++;
			// do one step
			double step;
			nav.FindNextBoundaryAndStep(m, p, d, path, newpath, resultpoint, step);
			// go on with navigation
			p = resultpoint;
			path=newpath;
			vol=path.Top();
		}
	}
	timer.Stop();
	std::cout << " time for 100000 particles " << timer.getDeltaSecs( ) << std::endl;
	std::cout << " average steps done " << stepsdone / 100000. << std::endl;
	delete m;
}


