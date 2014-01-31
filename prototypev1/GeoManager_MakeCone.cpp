/*
 * GeoManager_MakeTube.cpp
 *
 *  Created on: Dec 13, 2013
 *      Author: swenzel
 */

#include "GeoManager.h"


PhysicalVolume * GeoManager::MakePlacedCone( ConeParameters<> const * bp, TransformationMatrix const * tm, bool specialize_placement)
{
		return MakePlacedShape<Cone>( bp, tm, specialize_placement);
}
