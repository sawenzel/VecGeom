#include "backend/scalar_backend.h"
#include "volumes/placed_box.h"
#ifdef VECGEOM_CUDA
#include "backend/cuda_backend.cuh"
#endif

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

#ifdef VECGEOM_CUDA

namespace {

__global__
void ConstructOnGpu(LogicalVolume const *const logical_volume,
                    TransformationMatrix const *const matrix,
                    VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) PlacedBox(logical_volume, matrix);
}

} // End anonymous namespace

VPlacedVolume* PlacedBox::CopyToGpu(LogicalVolume const *const logical_volume,
                                    TransformationMatrix const *const matrix,
                                    VPlacedVolume *const gpu_ptr) const {
  ConstructOnGpu<<<1, 1>>>(logical_volume, matrix, gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedBox::CopyToGpu(
    LogicalVolume const *const logical_volume,
    TransformationMatrix const *const matrix) const {
  VPlacedVolume *const gpu_ptr = AllocateOnGpu<PlacedBox>();
  return CopyToGpu(logical_volume, matrix, gpu_ptr);
}

#endif

} // End namespace vecgeom