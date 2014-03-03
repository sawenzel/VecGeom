#include "backend/scalar_backend.h"
#include "volumes/placed_box.h"

namespace vecgeom {

VECGEOM_CUDA_HEADER_BOTH
bool PlacedBox::Inside(Vector3D<Precision> const &point) const {
  return PlacedBox::template InsideTemplate<1, 0, kScalar>(point);
}

VECGEOM_CUDA_HEADER_BOTH
Precision PlacedBox::DistanceToIn(Vector3D<Precision> const &position,
                                  Vector3D<Precision> const &direction,
                                  const Precision step_max) const {
  return PlacedBox::template DistanceToInTemplate<1, 0, kScalar>(position,
                                                                 direction,
                                                                 step_max);
}

VECGEOM_CUDA_HEADER_BOTH
Precision PlacedBox::DistanceToOut(Vector3D<Precision> const &position,
                                   Vector3D<Precision> const &direction) const {

  Vector3D<Precision> const &dim = AsUnplacedBox()->dimensions();

  const Vector3D<Precision> safety_plus  = dim + position;
  const Vector3D<Precision> safety_minus = dim - position;

  Vector3D<Precision> distance = safety_minus;
  const Vector3D<bool> direction_plus = direction < 0.0;
  distance.MaskedAssign(direction_plus, safety_plus);

  distance /= direction;

  const Precision min = distance.Min();
  return (min < 0) ? 0 : min;
}

} // End namespace vecgeom