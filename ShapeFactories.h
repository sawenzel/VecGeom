/*
 * ShapeFactories.h
 *
 *  Created on: Nov 30, 2013
 *      Author: swenzel
 */

#ifndef SHAPEFACTORIES_H_
#define SHAPEFACTORIES_H_

#include "PhysicalVolume.h"
#include "TubeTraits.h"
#include "PhysicalTube.h"
#include "GlobalDefs.h"

struct ShapeFactories
{
	template<int tid, int rid>
	static
	PhysicalVolume * CreateTube( TubeParameters<> const * tb, TransformationMatrix const * tm )
	{
		if( tb->GetRmin() == 0. )
		{
			if ( tb->dDPhi < UUtils::kTwoPi )
			{
				return new PlacedUSolidsTube<tid,rid,TubeTraits::NonHollowTubeWithPhi>(tb, tm);
			}
			else
			{
				return new PlacedUSolidsTube<tid,rid,TubeTraits::NonHollowTube>(tb, tm);
			}
		}
		else
		{
			if ( tb->dDPhi < UUtils::kTwoPi )
			{
				return new PlacedUSolidsTube<tid,rid,TubeTraits::HollowTubeWithPhi>(tb,tm);
			}
			else
			{
				return new PlacedUSolidsTube<tid,rid,TubeTraits::HollowTube>(tb,tm);
			}
		}
	}
};



};



#endif /* SHAPEFACTORIES_H_ */
