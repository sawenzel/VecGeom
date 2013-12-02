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
#include "PhysicalCone.h"
#include "PhysicalBox.h"

struct TubeFactory
{
	template<int tid, int rid>
	static
	PhysicalVolume * Create( TubeParameters<> const * tp, TransformationMatrix const * tm )
	{
		if( tp->GetRmin() == 0. )
		{
			if ( tp->GetDPhi() < UUtils::kTwoPi )
			{
				return new PlacedUSolidsTube<tid,rid,TubeTraits::NonHollowTubeWithPhi>(tp, tm);
			}
			else
			{
				return new PlacedUSolidsTube<tid,rid,TubeTraits::NonHollowTube>(tp, tm);
			}
		}
		else
		{
			if ( tp->GetDPhi() < UUtils::kTwoPi )
			{
				return new PlacedUSolidsTube<tid,rid,TubeTraits::HollowTubeWithPhi>(tp,tm);
			}
			else
			{
				return new PlacedUSolidsTube<tid,rid,TubeTraits::HollowTube>(tp,tm);
			}
		}
	}
};


struct
ConeFactory
{
	template<int tid, int rid>
	static
	PhysicalVolume * Create( ConeParameters<> const * cp, TransformationMatrix const * tm )
	{
		if( cp->GetRmin1() == 0. && cp->GetRmin2() == 0. )
			{
				if ( cp->GetDPhi() < UUtils::kTwoPi )
				{
					return new PlacedCone<tid,rid,ConeTraits::NonHollowConeWithPhi>( cp, tm );
				}
				else
				{
					return new PlacedCone<tid,rid,ConeTraits::NonHollowCone>( cp, tm );
				}
			}
		else
			{
				if ( cp->GetDPhi() < UUtils::kTwoPi )
				{
					return new PlacedCone<tid,rid,ConeTraits::HollowConeWithPhi>( cp, tm );
				}
				else
				{
					return new PlacedCone<tid,rid,ConeTraits::HollowCone>( cp, tm );
				}
			}
	}
};


struct
BoxFactory
{
	template<int tid, int rid>
	static
	PhysicalVolume * Create( BoxParameters const * cp, TransformationMatrix const * tm )
	{
		return new PlacedBox<tid,rid>( cp, tm );
	}
};

struct
NullFactory
{
	template<int tid, int rid>
	static
	PhysicalVolume * Create( BoxParameters const * cp, TransformationMatrix const * tm )
	{
		return NULL;
	}
};



#endif /* SHAPEFACTORIES_H_ */
