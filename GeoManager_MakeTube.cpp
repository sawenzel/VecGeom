/*
 * GeoManager_MakeTube.cpp
 *
 *  Created on: Dec 13, 2013
 *      Author: swenzel
 */

#include "GeoManager.h"


PhysicalVolume * GeoManager::MakePlacedTube( TubeParameters<> const * bp, TransformationMatrix const * tm)
{
		return MakePlacedShape<Tube>( bp, tm);
}
