/*
 * PlacedPolycone.h
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_PLACEDPOLYCONE_H_
#define VECGEOM_VOLUMES_PLACEDPOLYCONE_H_


#include "base/Global.h"
#include "backend/Backend.h"

#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"


namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedPolycone; )
VECGEOM_DEVICE_DECLARE_CONV( PlacedPolycone );

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedPolycone : public VPlacedVolume {



}; // end of class

} // end inline namespace

} // end global namespace

#endif /* VECGEOM_VOLUMES_PLACEDPOLYCONE_H_ */
