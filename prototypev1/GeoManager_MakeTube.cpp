/*
 * GeoManager_MakeTube.cpp
 *
 *  Created on: Dec 13, 2013
 *      Author: swenzel
 */

#include "GeoManager.h"


PhysicalVolume * GeoManager::MakePlacedTube( TubeParameters<> const * bp, TransformationMatrix const * tm, bool specialize_placement)
{
      return MakePlacedShape<Tube>( bp, tm, specialize_placement);
}
