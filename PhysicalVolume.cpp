/*
 * PhysicalVolume.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: swenzel
 */


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
	}
	return counter;
}
