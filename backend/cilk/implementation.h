#ifndef VECGEOM_BACKEND_CILK_IMPLEMENTATION_H_
#define VECGEOM_BACKEND_CILK_IMPLEMENTATION_H_

#include "base/global.h"

#include "backend/cilk/backend.h"
#include "base/aos3d.h"
#include "base/soa3d.h"
#include "volumes/placed_box.h"
#include "volumes/kernel/box_kernel.h"

namespace vecgeom {

template <TranslationCode trans_code, RotationCode rot_code,
          typename ContainerType>
VECGEOM_INLINE
void PlacedBox::InsideBackend(ContainerType const &points,
                              bool *const output) const {
  for (int i = 0; i < points.size(); i += kVectorSize) {
    const CilkBool result = InsideTemplate<trans_code, rot_code, kCilk>(
      Vector3D<CilkPrecision>(CilkPrecision(&points.x(i)),
                              CilkPrecision(&points.y(i)),
                              CilkPrecision(&points.z(i)))
    );
    result.store(&output[i]);
  }
}

template <TranslationCode trans_code, RotationCode rot_code,
          typename ContainerType>
VECGEOM_INLINE
void PlacedBox::DistanceToInBackend(ContainerType const &positions,
                                    ContainerType const &directions,
                                    Precision const *const step_max,
                                    Precision *const output) const {
  for (int i = 0; i < positions.size(); i += kVectorSize) {
    const CilkPrecision result = DistanceToInTemplate<trans_code, rot_code,
                                                      kCilk>(
      Vector3D<CilkPrecision>(CilkPrecision(&positions.x(i)),
                              CilkPrecision(&positions.y(i)),
                              CilkPrecision(&positions.z(i))),
      Vector3D<CilkPrecision>(CilkPrecision(&directions.x(i)),
                              CilkPrecision(&directions.y(i)),
                              CilkPrecision(&directions.z(i))),
      CilkPrecision(&step_max[i])
    );
    result.store(&output[i]);
  }
}

} // End namespace vecgeom

#endif // VECGEOM_BACKEND_CILK_IMPLEMENTATION_H_