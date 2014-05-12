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

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
void PlacedBox::PrintType() const {
  printf("PlacedBox");
}

void PlacedBox::Inside(SOA3D<Precision> const &points,
                       bool *const output) const {
  Inside_Looper(*this, points, output);
}

void PlacedBox::Inside(AOS3D<Precision> const &points,
                       bool *const output) const {
  Inside_Looper(*this, points, output);
}


void PlacedBox::DistanceToIn(SOA3D<Precision> const &positions,
                             SOA3D<Precision> const &directions,
                             Precision const *const step_max,
                             Precision *const output) const {
  DistanceToIn_Looper(
    *this, positions, directions, step_max, output
  );
}

void PlacedBox::DistanceToIn(AOS3D<Precision> const &positions,
                             AOS3D<Precision> const &directions,
                             Precision const *const step_max,
                             Precision *const output) const {
  DistanceToIn_Looper(
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
   return SafetyToIn_Looper(*this, position, safeties);
}


void PlacedBox::SafetyToIn( AOS3D<Precision> const &position,
        Precision *const safeties ) const {
   return SafetyToIn_Looper(*this, position, safeties);
}


void PlacedBox::SafetyToOut( SOA3D<Precision> const &position,
        Precision *const safeties ) const {
   return SafetyToOut_Looper(*this, position, safeties);
}

void PlacedBox::SafetyToOut( AOS3D<Precision> const &position,
        Precision *const safeties ) const {
   return SafetyToOut_Looper(*this, position, safeties);
}

#ifdef VECGEOM_BENCHMARK

VPlacedVolume const* PlacedBox::ConvertToUnspecialized() const {
  return new PlacedBox("", logical_volume_, transformation_);
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

#endif // VECGEOM_BENCHMARK

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void PlacedBox_CopyToGpu(LogicalVolume const *const logical_volume,
                         Transformation3D const *const transformation,
                         const int id,
                         VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedBox::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  vecgeom::PlacedBox_CopyToGpu(logical_volume, transformation, this->id(),
                               gpu_ptr);
  vecgeom::CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedBox::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedBox>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void PlacedBox_ConstructOnGpu(LogicalVolume const *const logical_volume,
                              Transformation3D const *const transformation,
                              const int id, VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::PlacedBox(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
    id
  );
}

void PlacedBox_CopyToGpu(LogicalVolume const *const logical_volume,
                         Transformation3D const *const transformation,
                         const int id,
                         VPlacedVolume *const gpu_ptr) {
  PlacedBox_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation, id,
                                     gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom
