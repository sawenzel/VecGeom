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
	PhysicalVolume * box3 = GeoManager::MakePlacedBox( boxlevel3, new TransformationMatrix(0,0,0,0,0,45));
	box2->AddDaughter( box3 ); // rotated 45 degree around z axis

	PhysicalVolume * box1 = GeoManager::MakePlacedBox(boxlevel1, identity);
	box1->AddDaughter( box2 );

	PhysicalVolume * box1left  = world->PlaceDaughter(GeoManager::MakePlacedBox(boxlevel1, new TransformationMatrix(-L/2.,0.,0.,0.,0.,0)), box1->GetDaughterList());
	PhysicalVolume * box1right = world->PlaceDaughter(GeoManager::MakePlacedBox(boxlevel1, new TransformationMatrix(+L/2.,0.,0.,0.,0.,0)), box1->GetDaughterList());

	std::cerr << " Number of daughters " << world->GetNumberOfDaughters() << std::endl;

    // perform basic tests
	SimpleVecNavigator nav(1, world);
	Vector3D result;
	VolumePath path(4);
	PhysicalVolume const * vol;

	{
	// point should be in world
	Vector3D p1(0, 9*L/10., 0); path.Clear();
	vol=nav.LocatePoint( world, p1, result, path );
	assert(vol==world);
	}

	{
	// outside world check
	Vector3D p2(-2*L, 9*L/10., 0); path.Clear();
	vol=nav.LocatePoint( world, p2, result, path );
	assert(vol==NULL);
	}

	{
	// inside box3 check
	Vector3D p3(-L/2., 0., 0.); path.Clear();
	vol=nav.LocatePoint( world, p3, result, path );
	assert(vol==box3);
	std::cerr << path.GetCurrentLevel() << std::endl;
	assert(path.GetCurrentLevel( ) == 4);
	assert(result == Vector3D(0.,0.,0));
	}

	{
	// inside box3 check ( but second box )
	Vector3D p3(L/2., 0., 0.); path.Clear();
	vol=nav.LocatePoint( world, p3, result, path );
	assert(vol==box3);
	std::cerr << path.GetCurrentLevel() << std::endl;
	assert(path.GetCurrentLevel( ) == 4);
	assert(result == Vector3D(0.,0.,0));
	}

	{
	// inside box2 check
	Vector3D p4(-L/2., 9*L/2./10., 0.); path.Clear();
	vol=nav.LocatePoint( world, p4, result, path );
	assert(vol==box2);
	}

	{
	// inside box2 check ( on other side )
	Vector3D p4(L/2., 9*L/2./10., 0.); path.Clear();
	vol=nav.LocatePoint( world, p4, result, path );
	assert(vol==box2);
	}

	{
	// inside box1 check
	Vector3D p5(-9.*L/10., 9*L/2./10., 0.); path.Clear();
	vol=nav.LocatePoint( world, p5, result, path );
	assert(vol == box1left );
	}

	{
	// inside box1 check
	Vector3D p6(9.*L/10., 9*L/2./10., 0.); path.Clear();
	vol=nav.LocatePoint( world, p6, result, path );
	assert(vol == box1right );
	assert(path.GetCurrentLevel( ) == 2); // this means actuall "next" free level
	}


	// now do location and transportation
	{
	  Vector3D p3(-L/2., 0., 0.); path.Clear();
	  Vector3D newpoint;
	  Vector3D d(9*L/2./10.,0,0);
	  TransformationMatrix * m=new TransformationMatrix();

	  vol=nav.LocatePoint( world, p3, result, path );
	  // move point in local reference frame
	  Vector3D p = result+d;
	  vol=nav.LocateLocalPointFromPath_Relative( p, newpoint, path, m );
	  // LocateLocalPointFromPath_Relative(Vector3D const & point, Vector3D & localpoint, VolumePath & path, TransformationMatrix * ) const;
	  assert( vol==box2 );
	  assert( path.GetCurrentLevel() == 3 );
	}



	// now do location and transportation
	{
		Vector3D p3(-L/2., 0., 0.); path.Clear();
		Vector3D newpoint;
		Vector3D d(0.1,0.,0.);
		TransformationMatrix * m=new TransformationMatrix();
		vol=nav.LocatePoint( world, p3, result, path );
		// move point in local reference frame
		Vector3D p = result+d;
		vol=nav.LocateLocalPointFromPath_Relative( p, newpoint, path, m );
		// LocateLocalPointFromPath_Relative(Vector3D const & point, Vector3D & localpoint, VolumePath & path, TransformationMatrix * ) const;
		assert( vol==box3 );
		assert( path.GetCurrentLevel() == 4 );
	}


	// now do location and transportation
	{
		Vector3D p3(-L/2., 0., 0.); path.Clear();
		Vector3D newpoint;
		Vector3D d(Sqrt2*L/4.+0.1,Sqrt2*L/4.+0.1,0);
		TransformationMatrix * m=new TransformationMatrix();
		vol=nav.LocatePoint( world, p3, result, path );
		// move point in local reference frame
		Vector3D p = result+d;
		vol=nav.LocateLocalPointFromPath_Relative( p, newpoint, path, m );
		// LocateLocalPointFromPath_Relative(Vector3D const & point, Vector3D & localpoint, VolumePath & path, TransformationMatrix * ) const;
		assert( vol==box1left );
		assert( path.GetCurrentLevel() == 2 );
	}


	// now do location and transportation
/*
	{
		Vector3D p3(-L/2., 0., 0.); path.Clear();
		Vector3D newpoint;
		Vector3D d(Sqrt2*L/2.+0.1,Sqrt2*L/2.+0.1,0);
		TransformationMatrix * m=new TransformationMatrix();
		vol=nav.LocatePoint( world, p3, result, path );
		// move point in local reference frame
		Vector3D p = result+d;
		vol=nav.LocateLocalPointFromPath_Relative( p, newpoint, path, m );

		// LocateLocalPointFromPath_Relative(Vector3D const & point, Vector3D & localpoint, VolumePath & path, TransformationMatrix * ) const;
		path.Print();
		assert( vol==world );
		assert( path.GetCurrentLevel() == 1 );
	}
*/

	// now do location and transportation
	{
			Vector3D p3(-L/2., 0., 0.); path.Clear();
			Vector3D newpoint;
			Vector3D d(4*L,4*L,0);
			TransformationMatrix * m=new TransformationMatrix();
			vol=nav.LocatePoint( world, p3, result, path );
			// move point in local reference frame
			Vector3D p = result+d;
			vol=nav.LocateLocalPointFromPath_Relative( p, newpoint, path, m );
			// LocateLocalPointFromPath_Relative(Vector3D const & point, Vector3D & localpoint, VolumePath & path, TransformationMatrix * ) const;
			assert( vol==NULL );
	}


	// now do location and transportation
	{
	  Vector3D p3(-L/2., 0., 0.); path.Clear();
	  Vector3D newpoint;
	  Vector3D d(L,0,0);
	  TransformationMatrix * m=new TransformationMatrix();

	  vol=nav.LocatePoint( world, p3, result, path );
	  assert( vol==box3 );

	  // move point in local reference frame
	  Vector3D p = result-d;
	  vol=nav.LocateLocalPointFromPath_Relative( p, newpoint, path, m );
	  // LocateLocalPointFromPath_Relative(Vector3D const & point, Vector3D & localpoint, VolumePath & path, TransformationMatrix * ) const;
	  newpoint.print();
	  path.Print();
	  assert( vol==box3 );
	  assert( path.GetCurrentLevel() == 4 );
	}


}


