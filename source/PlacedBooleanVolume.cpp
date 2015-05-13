/*
 * PlacedBooleanVolume.cpp
 *
 *  Created on: Nov 7, 2014
 *      Author: swenzel
 */

#include "volumes/PlacedBooleanVolume.h"
#include "volumes/SpecializedBooleanVolume.h"
#include "volumes/UnplacedBooleanVolume.h"
#include "volumes/LogicalVolume.h"

namespace vecgeom {

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN( SpecializedBooleanVolume, kUnion)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN( SpecializedBooleanVolume, kIntersection)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN( SpecializedBooleanVolume, kSubtraction)

#endif // VECGEOM_NVCC

} // End namespace vecgeom
