/**
 * @file placed_box.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include "backend/implementation.h"
#include "base/aos3d.h"
#include "base/soa3d.h"
#include "volumes/placed_box.h"
#ifdef VECGEOM_ROOT
#include "TGeoBBox.h"
#endif
#ifdef VECGEOM_USOLIDS
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

void PlacedBox::SafetyToIn( SOA3D<Precision> const &position,
		  Precision *const safeties ) const {
	return SafetyToIn_Looper<1,0>(*this, position, safeties);
}


void PlacedBox::SafetyToIn( AOS3D<Precision> const &position,
		  Precision *const safeties ) const {
	return SafetyToIn_Looper<1,0>(*this, position, safeties);
}


void PlacedBox::SafetyToOut( SOA3D<Precision> const &position,
		  Precision *const safeties ) const {
	return SafetyToOut_Looper(*this, position, safeties);
}

void PlacedBox::SafetyToOut( AOS3D<Precision> const &position,
		  Precision *const safeties ) const {
	return SafetyToOut_Looper(*this, position, safeties);
}

VPlacedVolume const* PlacedBox::ConvertToUnspecialized() const {
  return new PlacedBox(logical_volume_, matrix_);
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedBox::ConvertToRoot() const {
  return new TGeoBBox("", x(), y(), z());
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedBox::ConvertToUSolids() const {
  return new UBox("", x(), y(), z());
}
#endif

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void GpuInterface(LogicalVolume const *const logical_volume,
                  TransformationMatrix const *const matrix,
                  const int id,
                  VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedBox::CopyToGpu(LogicalVolume const *const logical_volume,
                                    TransformationMatrix const *const matrix,
                                    VPlacedVolume *const gpu_ptr) const {
  GpuInterface(logical_volume, matrix, this->id(), gpu_ptr);
  vecgeom::CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedBox::CopyToGpu(
    LogicalVolume const *const logical_volume,
    TransformationMatrix const *const matrix) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedBox>();
  return this->CopyToGpu(logical_volume, matrix, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class TransformationMatrix;
class VPlacedVolume;

__global__
void ConstructOnGpu(LogicalVolume const *const logical_volume,
                    TransformationMatrix const *const matrix,
                    const int id,
                    VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::PlacedBox(
    (vecgeom_cuda::LogicalVolume const*)logical_volume,
    (vecgeom_cuda::TransformationMatrix const*)matrix,
    id
  );
}

void GpuInterface(LogicalVolume const *const logical_volume,
                  TransformationMatrix const *const matrix,
                  const int id,
                  VPlacedVolume *const gpu_ptr) {
  ConstructOnGpu<<<1, 1>>>(logical_volume, matrix, id, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom