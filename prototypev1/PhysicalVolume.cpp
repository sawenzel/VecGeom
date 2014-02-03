/*
 * PhysicalVolume.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: swenzel
 */

#include "PhysicalVolume.h"
#include "Vector3D.h"
#include <cassert>
#include "PhysicalBox.h"
#include "GlobalDefs.h"

//template <TranslationIdType, RotationIdType> class PlacedBox;
#include "TGeoMatrix.h"
#include "TGeoShape.h"

#include <random>

typedef std::mt19937 MyRNG;  // the Mersenne Twister with a popular choice of parameters
static MyRNG rng(1);                   // e.g. keep one global instance (per thread)
std::uniform_real_distribution<> udouble_dist(0,1);


void PhysicalVolume::samplePoint( Vector3D & point, double dx, double dy, double dz, double scale )
{
	point.x=scale*(1-2.*udouble_dist(rng))*dx;
    point.y=scale*(1-2.*udouble_dist(rng))*dy;
    point.z=scale*(1-2.*udouble_dist(rng))*dz;
}


static void samplePointO( Vector3D & point, double dx, double dy, double dz, Vector3D const & origin, double scale )
{
	Vector3D tmp;
	PhysicalVolume::samplePoint(tmp, dx, dy, dz, scale);
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
	 assert(0.<=fraction<=1.);

	 // idea let's do an importance sampling Monte Carlo method; starting from some initial configuration
	 fillWithRandomDirections( dirs, np );

	 // nothing to do in case no daughter volumes
	 if( ! daughters || daughters->size() == 0 ) return;

	 std::vector<bool> hit(np,false); // to keep track which particle already hits something

	 int hitcounter=0;
	 for(auto i=0;i<np;++i)
	 {
		 Vector3D point,dir;
		 points.getAsVector(i, point);
		 dirs.getAsVector(i, dir);

		 // loop over daughters
		 if( daughters != NULL )
		 {
			 for(auto j=daughters->begin();j!=daughters->end();++j)
			 {
				 PhysicalVolume const * vol = (*j);
				 double d = vol->DistanceToIn( point, dir, Utils::kInfinity );
				 if( d < Utils::kInfinity )
				 {
					 hitcounter++;
					 hit[i]=true;
					 break;
				 }
			 }
		 }
	 }


	 // now first of all remove all directions that hit a daughter
	 for(auto i=0;i<np;++i)
	 {
		 if ( hit[i] )
		 {
			 Vector3D dir;
			 do
			 {
				Vector3D point;
				points.getAsVector( i, point );
				sampleDir( dir );

				hit[i]=false;

				// loop over daughters
				if( daughters != NULL )
				{
					for(auto j=daughters->begin();j!=daughters->end();++j)
					{
						PhysicalVolume const * vol = (*j);
						if( vol->DistanceToIn( point, dir, Utils::kInfinity ) < Utils::kInfinity )
						{
							hit[i]=false;
							break;
						}
					}
				}
			}
			while(hit[i]);
			hitcounter--;
			 // write back the dir
			dirs.setFromVector(i,dir);
		 }
	 }


	 // now gradually reintroduce hitting directions until threshold reached
	 while(hitcounter < np*fraction)
		 {
			 // pick a point randomly
			 int index = udouble_dist(rng)*np;
			 if( ! hit[index] )
			 {
				 do
				  {
					 Vector3D point,dir;
					 points.getAsVector( index, point );
					 sampleDir( dir );

					 // loop over daughters
					 if( daughters!=NULL )
					 {
					 for(auto j=daughters->begin();j!=daughters->end();++j)
					 {
						 PhysicalVolume const * vol = (*j);
						 if( vol->DistanceToIn( point, dir, Utils::kInfinity ) < Utils::kInfinity )
						 {
							hit[index]=true;
						    // write back the dir
							dirs.setFromVector(index,dir);
							break;
						 }
					 }}
				  }
				 while( hit[index]==false );
				 hitcounter++;
			 }
		 }
	 std::cerr << " have " << hitcounter << " points hitting " << std::endl;
}


  // here we assume that point IS in the local reference frame
bool
PhysicalVolume::ExclusiveContains( Vector3D const & point ) const
{
	bool contains = this->UnplacedContains( point );
	if( contains )
	{
		// C++11 type iteration over containers
		if( daughters!=NULL )
							 {

		for(auto i=this->daughters->begin(); i!=daughters->end(); ++i)
		{
		  PhysicalVolume const * vol= (*i);
		  if( vol->Contains( point ) ) return false;
		}}
	}
	return contains;
}


void
PhysicalVolume::fillWithRandomPoints( Vectors3DSOA &points, int number ) const
{
	int counter=0;

	//PlacedBox<1,0> const * box = (PlacedBox<1,0> const *) bbox ;
	PlacedBox<1,0> const * box = static_cast<PlacedBox<1,0> const * >(bbox);

	double dx=box->GetDX();
	double dy=box->GetDY();
	double dz=box->GetDZ();

	Vector3D origin(box->getMatrix()->getTrans());
	Vector3D point;
	for( auto i=0; i<number; ++i )
	{
		do
		  {
		    counter++;
		    samplePointO(point, dx, dy, dz, origin, 1);
		  }
		while ( ! ExclusiveContains(point) );
		points.setFromVector(i, point);
	}
	std::cerr << "needed " << counter << " steps to fill the volume " << std::endl;
	//return counter;
}

void PhysicalVolume::PrintDistToEachDaughter( Vector3D const & point, Vector3D const & dir ) const
{
	// calculate distance to Boundary of current volume
//	volume->DistanceToOut( points, dirs, steps, distance );

	// iterate over all the daughter
	std::list<PhysicalVolume const *> const * daughters = GetDaughterList();
	int counter = 0;
	for( auto iter = daughters->begin(); iter!=daughters->end(); ++iter )
	{
		PhysicalVolume const * daughter = (*iter);
		double dist = daughter->DistanceToIn( point, dir, Utils::kInfinity );
		std::cerr << "\t distance to daughter " << counter << " is " << dist << std::endl;
		counter++;
	}

}

void PhysicalVolume::PrintDistToEachDaughterROOT( Vector3D const & point, Vector3D const & dir ) const
{
	// iterate over all the daughter
	std::list<PhysicalVolume const *> const * daughters = GetDaughterList();
	int counter = 0;
	for( auto iter = daughters->begin(); iter!=daughters->end(); ++iter )
	{
		TGeoShape const * daughter = (*iter)->GetAsUnplacedROOTSolid();
		TGeoMatrix const * rm = (*iter)->getMatrix()->GetAsTGeoMatrix();
		double p[3], d[3];
		rm->MasterToLocal( &point.x, p );
		rm->MasterToLocalVect( &dir.x, d );
		double dist=daughter->DistFromOutside( p, d, 3, Utils::kInfinity, 0 );
		std::cerr << "\t distance to daughter (ROOT) " << counter << " is " << dist << std::endl;
		counter++;
	}
}

void PhysicalVolume::PrintDistToEachDaughterUSOLID( Vector3D const & point, Vector3D const & dir ) const
{
	// iterate over all the daughter
	std::list<PhysicalVolume const *> const * daughters = GetDaughterList();
	int counter = 0;
	for( auto iter = daughters->begin(); iter!=daughters->end(); ++iter )
	{
		VUSolid const * daughter = (*iter)->GetAsUnplacedUSolid();

		Vector3D p, d;
		(*iter)->getMatrix()->MasterToLocal<1,-1>( point, p );
		(*iter)->getMatrix()->MasterToLocalVec<-1>( dir, d );
		double dist=daughter->DistanceToIn( reinterpret_cast<UVector3 const &>(p), reinterpret_cast<UVector3 const &>(d), Utils::kInfinity );
		std::cerr << "\t distance to daughter (USOLID) " << counter << " is " << dist << std::endl;
		counter++;
	}
}

void PhysicalVolume::printInfo() const
{
	std::cerr << "PhysicalVolume with matrix " << matrix << std::endl ;// s " with matrix << std::endl;
}
