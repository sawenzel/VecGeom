#ifndef VECGEOM_KERNEL_H_
#define VECGEOM_KERNEL_H_

#include "Global.h"
#include "UnplacedBox.h"
#include "UnplacedTube.h"

template <class Backend>
void BoxInside(UnplacedBox const &box,
               typename Backend::double_v const *const point,
               typename Backend::bool_v &output) {

  typename Backend::bool_v inside[3];
  inside[0] = fabs(point[0]) < box.x();
  inside[1] = fabs(point[1]) < box.y();
  inside[2] = fabs(point[2]) < box.z();

  output = inside[0] && inside[1] && inside[2];
}

template <class Backend, class TubeSpecialization>
void TubeInside(UnplacedTube const &tube,
                typename Backend::double_v const *const point,
                typename Backend::bool_v &output) {
  if (TubeSpecialization::is_fancy) {
    output = true;
  } else {
    output = false;
  }
}

#endif // VECGEOM_KERNEL_H_