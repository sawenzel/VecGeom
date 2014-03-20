/**
 * @file placed_box.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include "backend/implementation.h"
#include "base/aos3d.h"
#include "base/soa3d.h"
#include "volumes/placed_box.h"
#ifdef VECGEOM_BENCHMARK
#include "TGeoBBox.h"
#include "UBox.hh"
#endif

namespace VECGEOM_NAMESPACE {


void PlacedBox::Inside(SOA3D<Precision> const &points,
                       bool *const output) const {
  Inside_Looper<1, 0>(*this, points, output);
}

void PlacedBox::Inside(AOS3D<Precision> const &points,
                       bool *const output) const {
  Inside_Looper<1, 0>(*this, points, output);
}


void PlacedBox::DistanceToIn(SOA3D<Precision> const &positions,
                             SOA3D<Precision> const &directions,
                             Precision const *const step_max,
                             Precision *const output) const {
  DistanceToIn_Looper<1, 0>(
    *this, positions, directions, step_max, output
  );
}

void PlacedBox::DistanceToIn(AOS3D<Precision> const &positions,
                             AOS3D<Precision> const &directions,
                             Precision const *const step_max,
                             Precision *const output) const {
  DistanceToIn_Looper<1, 0>(
    *this, positions, directions, step_max, output
  );
}


void PlacedBox::DistanceToOut( AOS3D<Precision> const &position,
							   AOS3D<Precision> const &direction,
                               Precision const * const step_max,
                               Precision *const distances
							 ) const
{
	// call the looper pattern which calls the appropriate shape methods
	return DistanceToOut_Looper(*this, position, direction, step_max, distances);
}


void PlacedBox::DistanceToOut( SOA3D<Precision> const &position,
							   SOA3D<Precision> const &direction,
                               Precision const * const step_max,
                               Precision *const distances
							 ) const
{
	// call the looper pattern which calls the appropriate shape methods
	return DistanceToOut_Looper(*this, position, direction, step_max, distances);
}


#ifdef VECGEOM_NVCC

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

#endif // VECGEOM_NVCC

#ifdef VECGEOM_BENCHMARK

VPlacedVolume const* PlacedBox::ConvertToUnspecialized() const {
  return new PlacedBox(logical_volume_, matrix_);
}

TGeoShape const* PlacedBox::ConvertToRoot() const {
  return new TGeoBBox("", x(), y(), z());
}

::VUSolid const* PlacedBox::ConvertToUSolids() const {
  return new UBox("", x(), y(), z());
}

#endif // VECGEOM_BENCHMARK

} // End global namespace
