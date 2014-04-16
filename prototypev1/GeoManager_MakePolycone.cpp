/*
 * GeoManager_MakePolycone.cpp
 *
 *  Created on: Jan 8, 2014
 *      Author: swenzel
 */

#include "GeoManager.h"


PhysicalVolume * GeoManager::MakePlacedPolycone( PolyconeParameters<> const * bp, TransformationMatrix const * tm, bool specialize_placement)
{
      return MakePlacedShape<Polycone>( bp, tm, specialize_placement);
}
