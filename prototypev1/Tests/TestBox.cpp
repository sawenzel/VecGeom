/*
 * TestVectorizedPlacedBox.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: swenzel
 */

// testing matrix transformations on vectors
#include "../TransformationMatrix.h"
#include "../Utils.h"
#include <iostream>
#include "mm_malloc.h"
#include "../GlobalDefs.h"
#include "../GeoManager.h"
#include "../Vector3DFast.h"
#include <cassert>

const std::vector<std::vector<double>> TransCases {{0,0,0},
    {10, 10, 0}};


// for the rotations ( C++11 feature )
const std::vector<std::vector<double> > EulerAngles  {{0.,0.,0.},
    {30.,0, 0.},
   //  {0, 45., 0.},
    //	{0, 0, 67.5}};
    {180, 0, 0},
	//{30, 48, 0.},
	{78.2, 81.0, -10.}};
//const std::vector<std::vector<double> > EulerAngles {{180,0,0}};

static void cmpresults(double * a1, double * a2, int np)
{
	int counter=0;
	for(auto i=0;i<np;++i)
	{
		if( a1[i] != a2[i] ) counter++;
#ifdef SHOWDIFFERENCES
		std::cerr << i << " " << a1[i] << " " << a2[i] << std::endl;
#endif
	}
}


int main()
{
  Vectors3DSOA points, pointsforcontains, dirs;
  int np=1024;

  points.alloc(np);
  dirs.alloc(np);
  pointsforcontains.alloc(np);

  double *distances = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
  double *distances2 = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
  double *steps = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
  for(auto i=0;i<np;++i) steps[i]=1E30;

  std::vector<Vector3D> conventionalpoints(np);
  std::vector<Vector3D> conventionaldirs(np);
  Vector3D * conventionalpoints2 = (Vector3D *) new Vector3D[np];
  Vector3D * conventionaldirs2 = (Vector3D *) new Vector3D[np];

  TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);
  PhysicalVolume * world = GeoManager::MakePlacedBox(new BoxParameters(30,30,30), identity);

  PhysicalVolume * daughter = GeoManager::MakePlacedBox(new BoxParameters(10,15,20), new TransformationMatrix(0,0,0,0,0,0));
  world->AddDaughter(daughter);
  
  world->fillWithRandomPoints(points,np);
  world->fillWithBiasedDirections(points, dirs, np, 4./10);


  // sample points for contains function
  for(int i=0;i<np;i++)
    {
      Vector3D tmp; 
      PhysicalVolume::samplePoint( tmp, 30, 30, 30, 1);
      pointsforcontains.set(i, tmp.x, tmp.y, tmp.z );
    }

  /*
  int counter=0;
  for(int i=0;i<np;i++)
    {
      Vector3D tmp = points.getAsVector(i);
      PhysicalVolume::samplePoint( tmp, 30, 30, 30, 1);
      bool containsold = daughter->Contains( tmp );
      bool containsnew = daughter->Contains( Vector3DFast(tmp.x, tmp.y, tmp.z) );
      assert( containsold == containsnew );
      counter+=containsold;
    }
  std::cerr << " contains test passed; actually inside: " << counter << std::endl;
  

  for(int i=0;i<np;i++)
    {
      Vector3D tmp = points.getAsVector(i);
      Vector3D tmpdir = dirs.getAsVector(i);
      double old = world->DistanceToOut( tmp, tmpdir, 1E30 );
      double n = world->DistanceToOut( Vector3DFast(tmp.x, tmp.y, tmp.z), Vector3DFast(tmpdir.x, tmpdir.y, tmpdir.z), 1E30 );
      assert( Utils::IsSameWithinTolerance(old,n) );
    }
  std::cerr << " distancetoout test passed " << std::endl;
  

	// now testing some matrix stuff;
	for(int i=0;i<np;i++)
	{
		Vector3D tmp = points.getAsVector(i);
		Vector3D tmpnew;
		Vector3DFast tmpf(tmp.x, tmp.y, tmp.z), tmpfnew;
		daughter->getMatrix()->MasterToLocalVec(tmp, tmpnew);
		daughter->getFastMatrix()->MasterToLocalVec<-1>(tmpf, tmpfnew);
		daughter->getMatrix()->print();
		daughter->getFastMatrix()->print();

		Vector3D cmp(tmpfnew[0], tmpfnew[1], tmpfnew[2]);
		//	std::cerr << cmp << std::endl;
		// std::cerr << tmpnew << std::endl;
		//	assert(cmp==tmpnew);
	}

	counter=0;
	for(int i=0;i<np;i++)
	{
		Vector3D tmp = points.getAsVector(i);
		Vector3D tmpdir = dirs.getAsVector(i);
		double old = daughter->DistanceToIn( tmp, tmpdir, 1E30 );
		Vector3DFast x(tmp.x, tmp.y, tmp.z);
		Vector3DFast y(tmpdir.x, tmpdir.y, tmpdir.z);
		double n = daughter->DistanceToIn(x, y, 1E30 );
		assert( Utils::IsSameWithinTolerance(old,n) );
		counter+=( n < 1E30 )? 1 : 0;
	}
	std::cerr << " distancetoin test passed; actually hit " << counter << std::endl;

  */

	// timing results
	StopWatch timer;

	// distance to in method
	double s=0.;
	timer.Start();
	for(int n=0;n<10000;n++)
	  {
	    for(int i=0;i<np;i++)
	      {
		Vector3D tmp = points.getAsVector(i);
		Vector3D tmpdir = dirs.getAsVector(i);
		s+=daughter->DistanceToIn( tmp, tmpdir, 1E30 );
	      }
	  }
	timer.Stop();
	std::cerr << "old time " << timer.getDeltaSecs() << std::endl;

	s=0.;
	timer.Start();
	for(int n=0;n<10000;n++)
	  {	
	    for(int i=0;i<np;i++)
	      {
		Vector3D tmp = points.getAsVector(i);
		Vector3D tmpdir = dirs.getAsVector(i);
		Vector3DFast x(tmp.x, tmp.y, tmp.z);
		Vector3DFast y(tmpdir.x, tmpdir.y, tmpdir.z);
		s+= daughter->DistanceToIn(x, y, 1E30 );
	      }
	  }
	timer.Stop();
	std::cerr << "new time " << timer.getDeltaSecs() << std::endl;

	// distance to in method
	s=0.;
	timer.Start();
	for(int n=0;n<10000;n++)
	  {
	    for(int i=0;i<np;i++)
	      {
		Vector3D tmp = points.getAsVector(i);
		Vector3D tmpdir = dirs.getAsVector(i);
		s+=world->DistanceToOut( tmp, tmpdir, 1E30 );
	      }
	  }
	timer.Stop();
	std::cerr << "old time " << timer.getDeltaSecs() << std::endl;

	// distance to in method
	s=0.;
	timer.Start();
	for(int n=0;n<10000;n++)
	  {
	    for(int i=0;i<np;i++)
	      {
		Vector3D tmp = points.getAsVector(i);
		Vector3D tmpdir = dirs.getAsVector(i);

		Vector3DFast tmpf(tmp.x,tmp.y,tmp.z);
		Vector3DFast tmpfdir(tmpdir.x,tmpdir.y,tmpdir.z);
		s+=world->DistanceToOut( tmpf, tmpfdir, 1E30 );
	      }
	  }
	timer.Stop();
	std::cerr << "new time disttoout " << timer.getDeltaSecs() << std::endl;


	s=0.;
	timer.Start();
	for(int n=0;n<10000;n++)
	  {	
	    for(int i=0;i<np;i++)
	      {
		Vector3D tmp = pointsforcontains.getAsVector(i);
		s+= daughter->Contains(tmp);
	      }
	  }
	timer.Stop();
	std::cerr << "old time contains " << timer.getDeltaSecs() << std::endl;



	s=0.;
	timer.Start();
	for(int n=0;n<10000;n++)
	  {	
	    for(int i=0;i<np;i++)
	      {
		Vector3D tmp = pointsforcontains.getAsVector(i);
		Vector3DFast x(tmp.x, tmp.y, tmp.z);
		s+= daughter->Contains(x);
	      }
	  }
	timer.Stop();
	std::cerr << "new time " << timer.getDeltaSecs() << std::endl;

	_mm_free(distances);
	return 1;
}
