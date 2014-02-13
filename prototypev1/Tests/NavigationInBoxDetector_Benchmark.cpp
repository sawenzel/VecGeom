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

#define ITERATIVE
//#define DEBUG

void foo( double s, Vector3D const &a, Vector3D const &b, Vector3D &c )
{
	c=a+s*b;
}

int main(int argc, char * argv[])
{
	bool iterative=true;
	if(argc>1) iterative=false;
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

	PhysicalVolume const * box1left  = world->PlaceDaughter(GeoManager::MakePlacedBox(boxlevel1, new TransformationMatrix(-L/2.,0.,0.,0.,0.,0)), box1->GetDaughters());
	PhysicalVolume const * box1right = world->PlaceDaughter(GeoManager::MakePlacedBox(boxlevel1, new TransformationMatrix(+L/2.,0.,0.,0.,0.,0)), box1->GetDaughters());


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
		int localstepsdone=0;
		double distancetravelled=0.;
		Vector3D p;
		PhysicalVolume::samplePoint( p, worldp->GetDX(), worldp->GetDY(), worldp->GetDZ(), 1. );
#ifdef DEBUG
		std::cerr << p << " " << worldp->GetDX()-p.GetX() << " ";
#endif

		// setup point in world
		Vector3D d(1,0,0);
		Vector3D resultpoint;

		m->SetToIdentity();
		PhysicalVolume const * vol;
#ifdef ITERATIVE
			vol = nav.LocatePoint_iterative( world, p, resultpoint, path, m );
#else
			vol = nav.LocatePoint( world, p, resultpoint, path, m );
#endif
			while( vol!=NULL )
		{
			localstepsdone++;
			// do one step
			double step;
#ifdef ITERATIVE
			nav.FindNextBoundaryAndStep_iterative(m, p, d, path, newpath, resultpoint, step);
#else
			nav.FindNextBoundaryAndStep(m, p, d, path, newpath, resultpoint, step);
#endif
			distancetravelled+=step;
#ifdef DEBUG
			std::cerr << "  proposed step: " << step << std::endl;
			std::cerr << "  next point " << resultpoint << std::endl;
			std::cerr << "  in vol " << newpath.Top() << std::endl;
#endif

			// go on with navigation
			p = resultpoint;
			path=newpath;
			vol=path.Top();
		}
#ifdef DEBUG
			std::cerr << localstepsdone << " " << distancetravelled << std::endl;
#endif
			stepsdone+=localstepsdone;
	}
	timer.Stop();
	std::cout << " time for 100000 particles " << timer.getDeltaSecs( ) << std::endl;
	std::cout << " average steps done " << stepsdone / 100000. << std::endl;
	std::cout << " time per step " << timer.getDeltaSecs()/stepsdone << std::endl;

	delete m;
}


