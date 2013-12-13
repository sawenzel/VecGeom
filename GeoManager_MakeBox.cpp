/*
 * GeoManager_MakeBox.cpp
 *
 *  Created on: Dec 13, 2013
 *      Author: swenzel
 */

#include "GeoManager.h"


PhysicalVolume * GeoManager::MakePlacedBox( BoxParameters const * bp, TransformationMatrix const * tm)
{
		return MakePlacedShape<Box>( bp, tm);
}
