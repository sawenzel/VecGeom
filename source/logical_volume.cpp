/**
 * @file logical_volume.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include <stdio.h>
#include <climits>

#include "backend/backend.h"
#include "base/array.h"
#include "base/transformation_matrix.h"
#include "management/volume_factory.h"
#include "volumes/logical_volume.h"
#include "volumes/placed_volume.h"

namespace VECGEOM_NAMESPACE {

LogicalVolume::~LogicalVolume() {
  for (Iterator<VPlacedVolume const*> i = daughters().begin();
       i != daughters().end(); ++i) {
    delete *i;
  }
  delete daughters_;
}

#ifndef VECGEOM_NVCC

VPlacedVolume* LogicalVolume::Place(
    char const *const label,
    TransformationMatrix const *const matrix) const {
  return unplaced_volume()->PlaceVolume(label, this, matrix);
}

VPlacedVolume* LogicalVolume::Place(
    TransformationMatrix const *const matrix) const {
  return Place("", matrix);
}

VPlacedVolume* LogicalVolume::Place(char const *const label) const {
  return unplaced_volume()->PlaceVolume(
           label, this, &TransformationMatrix::kIdentity
         );
}

VPlacedVolume* LogicalVolume::Place() const {
  return Place("");
}

void LogicalVolume::PlaceDaughter(char const *const label,
                                  LogicalVolume const *const volume,
                                  TransformationMatrix const *const matrix) {
  VPlacedVolume const *const placed = volume->Place(label, matrix);
  daughters_->push_back(placed);
}

void LogicalVolume::PlaceDaughter(LogicalVolume const *const volume,
                                  TransformationMatrix const *const matrix) {
  PlaceDaughter("", volume, matrix);
}

void LogicalVolume::PlaceDaughter(VPlacedVolume const *const placed) {
  daughters_->push_back(placed);
}

#endif

std::ostream& operator<<(std::ostream& os, LogicalVolume const &vol) {
  os << *vol.unplaced_volume() << " [";
  for (Iterator<VPlacedVolume const*> i = vol.daughters().begin();
       i != vol.daughters().end(); ++i) {
    if (i != vol.daughters().begin()) os << ", ";
    os << (**i);
  }
  os << "]";
  return os;
}

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void GpuInterface(VUnplacedVolume const *const unplaced_volume,
                  Vector<VPlacedVolume const*> *const daughters,
                  LogicalVolume *const output);

LogicalVolume* LogicalVolume::CopyToGpu(
    VUnplacedVolume const *const unplaced_volume,
    Vector<Daughter> *const daughters,
    LogicalVolume *const gpu_ptr) const {

  GpuInterface(unplaced_volume, daughters, gpu_ptr);
  vecgeom::CudaAssertError();
  return gpu_ptr;

}

LogicalVolume* LogicalVolume::CopyToGpu(
    VUnplacedVolume const *const unplaced_volume,
    Vector<Daughter> *const daughters) const {

  LogicalVolume *const gpu_ptr = vecgeom::AllocateOnGpu<LogicalVolume>();
  return this->CopyToGpu(unplaced_volume, daughters, gpu_ptr);

}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class VUnplacedVolume;
class VPlacedVolume;
class LogicalVolume;

__global__
void ConstructOnGpu(VUnplacedVolume const *const unplaced_volume,
                    Vector<VPlacedVolume const*> *daughters,
                    LogicalVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::LogicalVolume(
    (vecgeom_cuda::VUnplacedVolume const*)unplaced_volume,
    (vecgeom_cuda::Vector<vecgeom_cuda::VPlacedVolume const*> *)daughters
  );
}

void GpuInterface(VUnplacedVolume const *const unplaced_volume,
                  Vector<VPlacedVolume const*> *const daughters,
                  LogicalVolume *const gpu_ptr) {
  ConstructOnGpu<<<1, 1>>>(unplaced_volume, daughters, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom