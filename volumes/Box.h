/// @file Box.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
//
/// Includes all headers related to the box volume.

#ifndef VECGEOM_VOLUMES_BOX_H_
#define VECGEOM_VOLUMES_BOX_H_

#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "base/Global.h"
#include "volumes/PlacedBox.h"
#include "volumes/SpecializedBox.h"
#include "volumes/UnplacedBox.h"

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif

#endif // VECGEOM_VOLUMES_BOX_H_
