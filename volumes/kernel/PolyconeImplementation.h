/*
 * PolyconeImplementation.h
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */


#ifndef VECGEOM_VOLUMES_KERNEL_POLYCONEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_POLYCONEIMPLEMENTATION_H_

#include "base/Global.h"
#include "base/Transformation3D.h"
#include "volumes/UnplacedPolycone.h"

#include <cassert>
#include <stdio.h>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(PolyconeImplementation,
        TranslationCode,transCodeT, RotationCode,rotCodeT)

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedPolycone;

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct PolyconeImplementation {

  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;

  using PlacedShape_t = PlacedPolycone;
  using UnplacedShape_t = UnplacedPolycone;

  // here put all the implementations


}; // end PolyconeImplementation

}} // end namespace


#endif /* POLYCONEIMPLEMENTATION_H_ */
