/*
 * SimpleVecNavigator.cpp
 *
 *  Created on: Dec 12, 2013
 *      Author: swenzel
 */

#include "SimpleVecNavigator.h"
#include "GlobalDefs.h"
#include <list>

SimpleVecNavigator::SimpleVecNavigator(int np) {
	// TODO Auto-generated constructor stub
	workspace = (double * ) _mm_malloc( sizeof(double)*np, ALIGNMENT_BOUNDARY );
}

SimpleVecNavigator::~SimpleVecNavigator() {
	// TODO Auto-generated destructor stub
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
