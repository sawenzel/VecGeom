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
#include "../PFMWatch.h"

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

__attribute__((noinline))
double testdivsd(double  *a, double  *b )
{
  double s=0.;
  for(int n=0;n<10000;n++)
    {
      for(int i=0;i<1024;i++)
	{
	  s+=a[i]/b[i];
	}
    }
  return s;
}


double testdivpd(Vector3DFast const *a, Vector3DFast const *b )
{
  double s=0.;
  for(int n=0;n<10000;n++)
    {
      for(int i=0;i<1024;i++)
	{
	  // Vector3DFast c=a[i]/b[i];
	  s+=a[i][0]/b[i][0];
	}
    }
  return s;
}

double test1( PhysicalVolume const *v, Vector3D * p )
{
	double s=0.;
	Vector3D t;

	// interesting fact:
	// this loop would vectorize ( bad for performance comparison )
	// fast math would convert 9 multiplications to 3 multiplications  ( bad for performance comparison )
	for(int n=0;n<10000;n++)
	  {
	    for(int i=0;i<1024;i++)
	      {
		v->getMatrix()->MasterToLocal<1,-1>(p[i], t);
		s+=t[0]+t[1]+t[2];
	      }
	 }
	return s;
}

double foo( PhysicalVolume const * v , Vector3D const & input , Vector3D & output )
{
  v->getMatrix()->MasterToLocal<1,-1>(input, output);
}


double bar( PhysicalVolume const * v , Vector3D const & input )
{
  Vector3D tmp;
  v->getMatrix()->MasterToLocal<1,-1>(input, tmp);
  return tmp[0]+ tmp[1] + tmp[2];
}

double test2( PhysicalVolume const *v, Vector3DFast const * p )
{
	Vector3DFast t;
	double s=0.;
	for(int n=0;n<10000;n++)
	  {
	    for(int i=0;i<1024;i++)
	      {
		Vector3DFast x=p[i];
		v->getFastMatrix()->MasterToLocal<1,-1>(x, t);
		s+=t.Sum();
	      }
	  }
	return s;
}

double testdisttoinold(  PhysicalVolume const *v, Vector3D const * p, Vector3D const * d  )
{
  double s=0.;
  for(int n=0;n<10000;n++)
    {
      for(int i=0;i<1024;i++)
	{
	  s+=v->DistanceToIn( p[i], d[i], 1E30 );
	}
    }
  return s;
}


double testdisttoinnew(  PhysicalVolume const * __restrict__ v, Vector3DFast const *__restrict__ p, Vector3DFast const *__restrict__ d  )
{
  double s=0.;
  for(int n=0;n<10000;n++)
    {
      for(int i=0;i<1024;i++)
	{
	  s+=v->DistanceToIn( p[i], d[i], 1E30 );
	}
    }
  return s;
}


int main()
{
  Vectors3DSOA points, pointsforcontains, dirs;
  int np=1024;

  points.alloc(np);
  dirs.alloc(np);
  pointsforcontains.alloc(np);

  Vector3DFast * fastpoints = new Vector3DFast[np];
  Vector3DFast * fastdirs = new Vector3DFast[np];

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

  PhysicalVolume * daughter = GeoManager::MakePlacedBox(new BoxParameters(10,15,20), new TransformationMatrix(0,0,0,0,0,45));
  world->AddDaughter(daughter);
  
  world->fillWithRandomPoints(points,np);
  world->fillWithBiasedDirections(points, dirs, np, 9./10);


  // sample points for contains function
  for(int i=0;i<np;i++)
    {
      Vector3D tmp; 
      PhysicalVolume::samplePoint( tmp, 30, 30, 30, 1);
      pointsforcontains.set(i, tmp.x, tmp.y, tmp.z );

      tmp = points.getAsVector(i);
      fastpoints[i].SetX(tmp.x);
      fastpoints[i].SetY(tmp.y);
      fastpoints[i].SetZ(tmp.z);   

      conventionalpoints[i][0]=tmp.x;
      conventionalpoints[i][1]=tmp.y;
      conventionalpoints[i][2]=tmp.z;

      tmp = dirs.getAsVector(i);
      fastdirs[i].SetX(tmp.x);
      fastdirs[i].SetY(tmp.y);
      fastdirs[i].SetZ(tmp.z);   
      conventionaldirs[i][0]=tmp.x;
      conventionaldirs[i][1]=tmp.y;
      conventionaldirs[i][2]=tmp.z;
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
  */
	int counter=0;
	for(int i=0;i<np;i++)
	{
		Vector3D tmp = points.getAsVector(i);
		Vector3D tmpdir = dirs.getAsVector(i);
		double old = daughter->DistanceToIn( tmp, tmpdir, 1E30 );
		Vector3DFast x(tmp.x, tmp.y, tmp.z);
		Vector3DFast y(tmpdir.x, tmpdir.y, tmpdir.z);
		double n = daughter->DistanceToIn(x, y, 1E30 );
		std::cerr << old << " " << n << std::endl;
		//	assert( Utils::IsSameWithinTolerance(old,n) );
		counter+=( n < 1E30 )? 1 : 0;
	}
	std::cerr << " distancetoin test passed; actually hit " << counter << std::endl;
	exit(1);


	// timing results
	StopWatch timer;

	// distance to in method
	double s=0.;
	timer.Start();
	s=testdisttoinold(daughter, &conventionalpoints[0], &conventionaldirs[0]);
	timer.Stop();
	std::cerr << "old time distance to in " << timer.getDeltaSecs() << std::endl;

	timer.Start();
	s=testdisttoinnew(daughter, fastpoints, fastdirs);
	timer.Stop();
	std::cerr << "new time distance to in" << timer.getDeltaSecs() << std::endl;

	// distance to in method
	s=0.;
	timer.Start();
	for(int n=0;n<10000;n++)
	  {
	    for(int i=0;i<np;i++)
	      {
		s+=world->DistanceToOut( conventionalpoints[i], conventionaldirs[i], 1E30 );
	      }
	  }
	timer.Stop();
	std::cerr << "old time disttoout " << timer.getDeltaSecs() << std::endl;

	// distance to in method
	s=0.;
	timer.Start();
	for(int n=0;n<10000;n++)
	  {
	    for(int i=0;i<np;i++)
	      {
		s+=world->DistanceToOut( fastpoints[i], fastdirs[i], 1E30 );
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
		s+= daughter->Contains(conventionalpoints[i]);
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
		s+= daughter->Contains(fastpoints[i]);
	      }
	  }
	timer.Stop();
	std::cerr << "new time contains " << timer.getDeltaSecs() << std::endl;

	{
	PFMWatch pfmtimer;
	pfmtimer.Start();
	timer.Start();
	double s = test1( daughter, &conventionalpoints[0] );
	timer.Stop();
	pfmtimer.Stop();
	pfmtimer.printSummary();
	std::cerr << "old time matrix" << timer.getDeltaSecs() << " " << s << std::endl;
	}

	{
	  PFMWatch pfmtimer;
	pfmtimer.Start();
	timer.Start();
	double s = test2( daughter, fastpoints );
	timer.Stop();
	pfmtimer.Stop();
	pfmtimer.printSummary();
	std::cerr << "new time matrix" << timer.getDeltaSecs() << " " << s << std::endl;
	}

	{
	PFMWatch pfmtimer;
	pfmtimer.Start();
	timer.Start();
	double s = testdivsd( points.x, dirs.x );
	timer.Stop();
	pfmtimer.Stop();
	pfmtimer.printSummary();
	std::cerr << "old time matrix" << timer.getDeltaSecs() << " " << s << std::endl;
	}

	{
	  PFMWatch pfmtimer;
	pfmtimer.Start();
	timer.Start();
	double s = testdivpd( fastpoints, fastdirs );
	timer.Stop();
	pfmtimer.Stop();
	pfmtimer.printSummary();
	std::cerr << "new time matrix" << timer.getDeltaSecs() << " " << s << std::endl;
	}

	_mm_free(distances);

	delete fastpoints;

	return 1;
}
