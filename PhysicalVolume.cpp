/*
 * PhysicalVolume.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: swenzel
 */

<<<<<<< HEAD

#include "PhysicalVolume.h"

// checks if point is contained in exactly vol without being in daughters
static bool InExclVolume( TGeoVolume * vol, double * point)
{
	bool contains = vol->GetShape()->Contains( point );
    	if( contains )
    	{
    		for(unsigned int i=0;i<vol->GetNdaughters();++i)
	  {
	    // need to transform point to daughter
	    TGeoMatrix *m = vol->GetNode(i)->GetMatrix();
	    double localpoint[3];
	    m->MasterToLocal( point, localpoint );

	    bool indaughter = vol->GetNode(i)->GetVolume()->GetShape()->Contains(localpoint);
	    if( indaughter ) return false;
	  }
      }
    return contains;
  }



static void samplePoint( double *point, double dx, double dy, double dz, double scale )
{
	point[0]=scale*(1-2.*gRandom->Rndm())*dx;
    point[1]=scale*(1-2.*gRandom->Rndm())*dy;
    point[2]=scale*(1-2.*gRandom->Rndm())*dz;
}

static void samplePoint( double *point, double dx, double dy, double dz, double const * origin, double scale )
{
	point[0]=origin[0]+scale*(1-2.*gRandom->Rndm())*dx;
    point[1]=origin[1]+scale*(1-2.*gRandom->Rndm())*dy;
    point[2]=origin[2]+scale*(1-2.*gRandom->Rndm())*dz;
}


// creates random normalized vectors
static void sampleDir( double * dir )
{
    dir[0]=(1-2.*gRandom->Rndm());
    dir[1]=(1-2.*gRandom->Rndm());
    dir[2]=(1-2.*gRandom->Rndm());
    double inversenorm=1./sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
    dir[0]*=inversenorm;
    dir[1]*=inversenorm;
    dir[2]*=inversenorm;
  }

  static void sampleDir( double * dir, unsigned int np )
  {
    for(unsigned int k=0;k<np;++k)
      {
    	dir[3*k+0]=(1-2.*gRandom->Rndm());
    	dir[3*k+1]=(1-2.*gRandom->Rndm());
    	dir[3*k+2]=(1-2.*gRandom->Rndm());
    	double inversenorm=1./sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
    	dir[3*k+0]*=inversenorm;
    	dir[3*k+1]*=inversenorm;
    	dir[3*k+2]*=inversenorm;
      }
  }

  // here we assume that point IS in the local reference frame
bool
PhysicalVolume::ExclusiveContains( Vector3D const & point ) const
{
	bool contains = this->UnplacedContains( point );
	if( contains )
	{
		// C++11 type iteration over containers
		for(auto i=this->daughters->begin(); i!=daughters->end(); ++i)
		{
		  PhysicalVolume * vol= (*i);
		  if( vol->Contains( point ) ) return false;
		}
	}
	return contains;
}


void
PhysicalVolume::fillWithRandomPoints( Vectors3DSOA &points, Vectors3DSOA & dirs, int number ) const
{
	int counter=0;
	double dx=bbox->GetDX();
	double dy=bbox->GetDY();
	double dz=bbox->GetDZ();

	double const *origin=bbox->GetTrans();
	double point[3];
	// std::cout << dx << " " << dy << " " << dz << std::endl;

	for( auto i=0; i<number; ++i )
	{
		do
		  {
		    counter++;
		    Util::samplePoint(point,dx,dy,dz, origin, 1);
		  }
		while ( ! ExclusiveContains( point ) );
		points[3*i+0]=point[0];
		points[3*i+1]=point[1];
		points[3*i+2]=point[2];
=======
#include "PhysicalVolume.h"
#include "Vector3D.h"

#include <random>
typedef std::mt19937 MyRNG;  // the Mersenne Twister with a popular choice of parameters
uint32_t 1;           // populate somehow

static MyRNG rng(1);                   // e.g. keep one global instance (per thread)

std::uniform_double_distribution<double> udouble_dist(0,1);


static void samplePoint( Vector3D & point, double dx, double dy, double dz, double scale )
{
	point.x=scale*(1-2.*udouble_dist(rng))*dx;
    point.y=scale*(1-2.*udouble_dist(rng))*dy;
    point.z=scale*(1-2.*udouble_dist(rng))*dz;
}

static void samplePoint( Vector3D & point, double dx, double dy, double dz, Vector3D & origin, double scale )
{
	Vector3D tmp;
	samplePoint(tmp, dx, dy, dz, scale);
	point.x=origin.x + tmp.x;
    point.y=origin.y + tmp.y;
    point.z=origin.z + tmp.z;
}


// creates random normalized vectors
static void sampleDir( Vector3D & dir )
{
    dir.x=(1-2.*udouble_dist(rng));
    dir.y=(1-2.*udouble_dist(rng));
    dir.z=(1-2.*udouble_dist(rng));
    double inversenorm=1./sqrt(dir.x*dir.x+dir.y*dir.y+dir.z*dir.z);
    dir.x*=inversenorm;
    dir.y*=inversenorm;
    dir.z*=inversenorm;
  }

// this is a static method
 void PhysicalVolume::fillWithRandomDirections( Vectors3DSOA &dirs, int np )
  {
    for(auto k=0;k<np;++k)
      {
    	Vector3D tmp;
    	sampleDir(tmp);
    	dirs.x[k]=tmp.x;
    	dirs.y[k]=tmp.y;
    	dirs.z[k]=tmp.z;
      }
  }


 void PhysicalVolume::fillWithBiasedDirections( Vectors3DSOA const & points, Vectors3DSOA &dirs, int np, double fraction ) const
 {
	 // idea let's do an importance sampling Monte Carlo method; starting from some initial configuration
	 fillWithRandomDirections( dirs, np );

 }


  // here we assume that point IS in the local reference frame
bool
PhysicalVolume::ExclusiveContains( Vector3D const & point ) const
{
	bool contains = this->UnplacedContains( point );
	if( contains )
	{
		// C++11 type iteration over containers
		for(auto i=this->daughters->begin(); i!=daughters->end(); ++i)
		{
		  PhysicalVolume * vol= (*i);
		  if( vol->Contains( point ) ) return false;
		}
	}
	return contains;
}


void
PhysicalVolume::fillWithRandomPoints( Vectors3DSOA &points, int number) const
{
	int counter=0;
	double dx=bbox->GetDX();
	double dy=bbox->GetDY();
	double dz=bbox->GetDZ();

	Vector3D origin(bbox->getMatrix()->getTrans());
	double point[3];
	for( auto i=0; i<number; ++i )
	{
		do
		  {
		    counter++;
		    samplePoint(point,dx,dy,dz, origin, 1);
		  }
		while ( ! ExclusiveContains(point) );
		points.x[i]=point.x;
		points.y[i]=point.y;
		points.z[i]=point.z;
>>>>>>> branch 'master' of ssh://git@bitbucket.org/swenzel/fastgeom.git
	}
	return counter;
}
/*
 * PhysicalVolume.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: swenzel
 */

#include "PhysicalVolume.h"
#include "Vector3D.h"

#include <random>
typedef std::mt19937 MyRNG;  // the Mersenne Twister with a popular choice of parameters
uint32_t 1;           // populate somehow

static MyRNG rng(1);                   // e.g. keep one global instance (per thread)

std::uniform_double_distribution<double> udouble_dist(0,1);


static void samplePoint( Vector3D & point, double dx, double dy, double dz, double scale )
{
	point.x=scale*(1-2.*udouble_dist(rng))*dx;
    point.y=scale*(1-2.*udouble_dist(rng))*dy;
    point.z=scale*(1-2.*udouble_dist(rng))*dz;
}

static void samplePoint( Vector3D & point, double dx, double dy, double dz, Vector3D & origin, double scale )
{
	Vector3D tmp;
	samplePoint(tmp, dx, dy, dz, scale);
	point.x=origin.x + tmp.x;
    point.y=origin.y + tmp.y;
    point.z=origin.z + tmp.z;
}


// creates random normalized vectors
static void sampleDir( Vector3D & dir )
{
    dir.x=(1-2.*udouble_dist(rng));
    dir.y=(1-2.*udouble_dist(rng));
    dir.z=(1-2.*udouble_dist(rng));
    double inversenorm=1./sqrt(dir.x*dir.x+dir.y*dir.y+dir.z*dir.z);
    dir.x*=inversenorm;
    dir.y*=inversenorm;
    dir.z*=inversenorm;
  }

// this is a static method
 void PhysicalVolume::fillWithRandomDirections( Vectors3DSOA &dirs, int np )
  {
    for(auto k=0;k<np;++k)
      {
    	Vector3D tmp;
    	sampleDir(tmp);
    	dirs.x[k]=tmp.x;
    	dirs.y[k]=tmp.y;
    	dirs.z[k]=tmp.z;
      }
  }


 void PhysicalVolume::fillWithBiasedDirections( Vectors3DSOA const & points, Vectors3DSOA &dirs, int np, double fraction ) const
 {
	 // idea let's do an importance sampling Monte Carlo method; starting from some initial configuration
	 fillWithRandomDirections( dirs, np );

 }


  // here we assume that point IS in the local reference frame
bool
PhysicalVolume::ExclusiveContains( Vector3D const & point ) const
{
	bool contains = this->UnplacedContains( point );
	if( contains )
	{
		// C++11 type iteration over containers
		for(auto i=this->daughters->begin(); i!=daughters->end(); ++i)
		{
		  PhysicalVolume * vol= (*i);
		  if( vol->Contains( point ) ) return false;
		}
	}
	return contains;
}


void
PhysicalVolume::fillWithRandomPoints( Vectors3DSOA &points, int number) const
{
	int counter=0;
	double dx=bbox->GetDX();
	double dy=bbox->GetDY();
	double dz=bbox->GetDZ();

	Vector3D origin(bbox->getMatrix()->getTrans());
	double point[3];
	for( auto i=0; i<number; ++i )
	{
		do
		  {
		    counter++;
		    samplePoint(point,dx,dy,dz, origin, 1);
		  }
		while ( ! ExclusiveContains(point) );
		points.x[i]=point.x;
		points.y[i]=point.y;
		points.z[i]=point.z;
	}
	return counter;
}
