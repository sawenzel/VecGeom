#include "base/aos3d.h"
#include "base/soa3d.h"
#include "implementation.h"
#include "volumes/placed_box.h"
#ifdef VECGEOM_COMPARISON
#include "TGeoBBox.h"
#include "UBox.hh"
#endif

namespace vecgeom {

void PlacedBox::Inside(SOA3D<Precision> const &points,
                       bool *const output) const {
  InsideBackend<1, 0>(*this, points, output);
}

void PlacedBox::Inside(AOS3D<Precision> const &points,
                       bool *const output) const {
  InsideBackend<1, 0>(*this, points, output);
}

void PlacedBox::DistanceToIn(SOA3D<Precision> const &positions,
                             SOA3D<Precision> const &directions,
                             Precision const *const step_max,
                             Precision *const output) const {
  DistanceToInBackend<1, 0>(
    *this, positions, directions, step_max, output
  );
}

void PlacedBox::DistanceToIn(AOS3D<Precision> const &positions,
                             AOS3D<Precision> const &directions,
                             Precision const *const step_max,
                             Precision *const output) const {
  DistanceToInBackend<1, 0>(
    *this, positions, directions, step_max, output
  );
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

#endif // VECGEOM_CUDA

#ifdef VECGEOM_COMPARISON

VPlacedVolume const* PlacedBox::ConvertToUnspecialized() const {
  return new PlacedBox(logical_volume_, matrix_);
}

TGeoShape const* PlacedBox::ConvertToRoot() const {
  return new TGeoBBox("", x(), y(), z());
}

::VUSolid const* PlacedBox::ConvertToUSolids() const {
  return new UBox("", x(), y(), z());
}

#endif // VECGEOM_COMPARISON

} // End namespace vecgeom