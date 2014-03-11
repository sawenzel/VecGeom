#ifndef VECGEOM_BACKEND_SCALAR_IMPLEMENTATION_H_
#define VECGEOM_BACKEND_SCALAR_IMPLEMENTATION_H_

#include "base/global.h"

#include "base/aos3d.h"
#include "base/soa3d.h"
#include "backend/scalar/backend.h"
#include "volumes/placed_box.h"
#include "volumes/kernel/box_kernel.h"

namespace vecgeom {

template <TranslationCode trans_code, RotationCode rot_code,
          typename ContainerType>
VECGEOM_INLINE
void PlacedBox::InsideBackend(ContainerType const &points,
                              bool *const output) const {
  for (int i = 0; i < points.size(); ++i) {
    output[i] = InsideTemplate<trans_code, rot_code, kScalar>(points[i]);
  }
}

template <TranslationCode trans_code, RotationCode rot_code,
          typename ContainerType>
VECGEOM_INLINE
void PlacedBox::DistanceToInBackend(ContainerType const &positions,
                                    ContainerType const &directions,
                                    Precision const *const step_max,
                                    Precision *const output) const {
  for (int i = 0; i < positions.size(); ++i) {
    output[i] = DistanceToInTemplate<trans_code, rot_code, kScalar>(
      positions[i], directions[i], step_max[i]
    );
  }
}

} // End namespace vecgeom

#endif // VECGEOM_BACKEND_SCALAR_IMPLEMENTATION_H_