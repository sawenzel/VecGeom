/*
 * SimpleVecNavigator.cpp
 *
 *  Created on: Dec 12, 2013
 *      Author: swenzel
 */

#include "SimpleVecNavigator.h"
#include "GlobalDefs.h"
#include "Vc/vector.h"
#include <list>
#include "TGeoShape.h"
#include "TGeoMatrix.h"

SimpleVecNavigator::SimpleVecNavigator(int np) {
	// TODO Auto-generated constructor stub
	workspace = (double * ) _mm_malloc( sizeof(double)*np, ALIGNMENT_BOUNDARY );
	transformeddirs.alloc(np);
	transformedpoints.alloc(np);
}

SimpleVecNavigator::~SimpleVecNavigator() {
	// TODO Auto-generated destructor stub
}

static
inline
void
__attribute__((always_inline))
MIN_v ( double * __restrict__ step, double const * __restrict__ workspace, uint64_t * __restrict__ nextnodepointersasints,
		uint64_t __restrict__ curcandidateVolumePointerasint, unsigned int np )
{
	// this is not vectorizing !
	for(auto k=0;k<np;++k)
    {
      step[k] = (workspace[k]<step[k]) ? workspace[k] : step[k];
      nextnodepointersasints[k] = (workspace[k]< step[k]) ?  curcandidateVolumePointerasint : nextnodepointersasints[k];
    	  //	  std::cout << " hit some node " << curcandidateNode << std::endl;
    }
}

static
inline
void
__attribute__((always_inline))
MIN_v ( double * __restrict__ step, double const * __restrict__ workspace, PhysicalVolume ** __restrict__ nextnodepointersasints,
		PhysicalVolume const * __restrict__ curcandidateVolumePointerasint, unsigned int np )
{
	// this is not vectorizing !
	for(auto k=0;k<np;++k)
    {
		bool cond = workspace[k]<step[k];
		step[k] = (cond)? workspace[k] : step[k];
		nextnodepointersasints[k] = (cond)? curcandidateVolumePointerasint : nextnodepointersasints[k];
    }
}

static
inline
void
__attribute__((always_inline))
MINVc_v ( double * __restrict__ step, double const * __restrict__ workspace, uint64_t * __restrict__ nextnodepointersasints,
		uint64_t  __restrict__ curcandidateVolumePointerasint, unsigned int np )
{

	// this is not vectorizing !
	//for(auto k=0;k<np;++k)
    //{
//      step[k] = (workspace[k]<step[k]) ? workspace[k] : step[k];
      //nextnodepointersasints[k] = (workspace[k]< step[k]) ?  curcandidateVolumePointerasint : nextnodepointersasints[k];
    	  //	  std::cout << " hit some node " << curcandidateNode << std::endl;
    //}
	for(auto k=0;k<np;k+=Vc::double_v::Size)
	{
		Vc::double_v stepvec(&step[k]);
		Vc::double_v pointervec((double *) (&nextnodepointersasints[k]));
		Vc::double_v workspacevec(&workspace[k]);
		Vc::double_m m = stepvec < workspacevec;
		stepvec( m ) = workspacevec;
		pointervec( m ) = curcandidateVolumePointerasint;
		stepvec.store(&step[k]);
		pointervec.store((double *) (&nextnodepointersasints[k]));
	}
}

void SimpleVecNavigator::DistToNextBoundary( PhysicalVolume const * volume, Vectors3DSOA const & points,
											 Vectors3DSOA const & dirs,
											 double const * steps,
											 double * distance,
											 PhysicalVolume ** nextnode, int np ) const
{
// init nextnode ( maybe do a memcpy )
	for(auto k=0;k<np;++k)
	{
		nextnode[k]=0;
	}

	// calculate distance to Boundary of current volume
	volume->DistanceToOut( points, dirs, steps, distance );

	// iterate over all the daughter
	std::list<PhysicalVolume const *> const * daughters = volume->GetDaughterList();
	for( auto iter = daughters->begin(); iter!=daughters->end(); ++iter )
	{
		PhysicalVolume const * daughter = (*iter);
		// previous distance become step estimate, distance to daughter returned in workspace
		daughter->DistanceToIn( points, dirs, distance, workspace );

		// update dist and current nextnode candidate
		// maybe needs to be vectorized manually with Vc
		// MINVc_v(distance, workspace, reinterpret_cast<uint64_t *>(nextnode), reinterpret_cast<uint64_t> (daughter), np);
		// MIN_v(distance, workspace, reinterpret_cast<uint64_t *>(nextnode), reinterpret_cast<uint64_t> (daughter), np);
		MIN_v(distance, workspace, nextnode, daughter, np);
	}
}


void SimpleVecNavigator::DistToNextBoundaryUsingUnplacedVolumes( PhysicalVolume const * volume, Vectors3DSOA const & points,
											 Vectors3DSOA const & dirs,
											 double const * steps,
											 double * distance,
											 PhysicalVolume ** nextnode, int np ) const
{
// init nextnode ( maybe do a memcpy )
	for(auto k=0;k<np;++k)
	{
		nextnode[k]=0;
	}

	// calculate distance to Boundary of current volume
	volume->DistanceToOut( points, dirs, steps, distance );

	// iterate over all the daughter
	std::list<PhysicalVolume const *> const * daughters = volume->GetDaughterList();
	for( auto iter = daughters->begin(); iter!=daughters->end(); ++iter )
	{
		PhysicalVolume const * daughter = (*iter);

		PhysicalVolume const * unplacedvolume = daughter->GetAsUnplacedVolume();

		// previous distance become step estimate, distance to daughter returned in workspace

		// now need to do matrix transformations outside
		TransformationMatrix const * tm = unplacedvolume->getMatrix();
		tm->MasterToLocalCombined( points, const_cast<Vectors3DSOA &>(transformedpoints),
				dirs, const_cast<Vectors3DSOA &>(transformeddirs ));

		unplacedvolume->DistanceToIn( transformedpoints, transformeddirs, distance, workspace );

		// update dist and current nextnode candidate
		// maybe needs to be vectorized manually with Vc
		for( auto k=0; k < np; ++k )
		{
			if( workspace[k] < distance[k] )
			{
				distance[k]=workspace[k];
				nextnode[k]=const_cast<PhysicalVolume *>(daughter);
			}
		}
	}
}


void SimpleVecNavigator::DistToNextBoundaryUsingROOT(  PhysicalVolume const * volume, double const * points,
		 	 	 	 	 	 	 	 	 	 	 	   double const * dirs,
		 	 	 	 	 	 	 	 	 	 	 	   double const * steps,
		 	 	 	 	 	 	 	 	 	 	 	   double * distance,
		 	 	 	 	 	 	 	 	 	 	 	   PhysicalVolume ** nextnode, int np ) const
{
	for(auto k=0;k<np;k++)
	{
		nextnode[k]=0;

		distance[k]=(volume->GetAsUnplacedROOTSolid())->DistFromInside(const_cast<double *>( &points[3*k] ),
																	const_cast<double *>(&dirs[3*k]),
																		3,	steps[k], 0 );

		std::list<PhysicalVolume const *> const * daughters = volume->GetDaughterList();
		for( auto iter = daughters->begin(); iter!=daughters->end(); ++iter )
			{
				PhysicalVolume const * daughter = (*iter);
				TGeoShape const * unplacedvolume = daughter->GetAsUnplacedROOTSolid();

				// previous distance become step estimate, distance to daughter returned in workspace

				// now need to do matrix transformations outside
				TGeoMatrix const * rm = daughter->getMatrix()->GetAsTGeoMatrix();

				double localpoint[3];
				double localdir[3];
				rm->MasterToLocal( &points[3*k], localpoint );
				rm->MasterToLocalVect( &dirs[3*k], localdir );

				double distd = unplacedvolume->DistFromOutside( localpoint, localdir, 3, distance[k], 0 );

				if( distd < distance[k] )
				{
					distance[k] = distd;
					nextnode[k] = const_cast<PhysicalVolume *>(daughter);
				}
			}
	}
}







