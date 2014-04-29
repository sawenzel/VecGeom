#ifndef VECGEOM_KERNEL_H_
#define VECGEOM_KERNEL_H_

#include "Global.h"
#include "UnplacedBox.h"
#include "UnplacedTube.h"

struct BoxImplementation {

  template <class Backend>
  inline
  static void Inside(UnplacedBox const &box,
                     Vector3D<typename Backend::double_v> const &point,
                     typename Backend::bool_v &output) {

    typename Backend::bool_v inside[3];
    inside[0] = fabs(point[0]) < box.x();
    inside[1] = fabs(point[1]) < box.y();
    inside[2] = fabs(point[2]) < box.z();

    output = inside[0] && inside[1] && inside[2];
  }

};

template <class TubeSpecialization>
struct TubeImplementation {

  template <class Backend>
  inline
  static void Inside(UnplacedTube const &tube,
                     Vector3D<typename Backend::double_v> const &point,
                     typename Backend::bool_v &output) {
    if (TubeSpecialization::isFancy) {
      output = typename Backend::bool_v(true);
    } else {
      output = typename Backend::bool_v(false);
    }
  }

};

#endif // VECGEOM_KERNEL_H_