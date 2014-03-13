#include <stdio.h>
#include <climits>
#include "backend.h"
#include "base/array.h"
#include "base/transformation_matrix.h"
#include "management/volume_factory.h"
#include "volumes/logical_volume.h"
#include "volumes/placed_volume.h"

namespace vecgeom {

LogicalVolume::~LogicalVolume() {
  for (Iterator<VPlacedVolume const*> i = daughters().begin();
       i != daughters().end(); ++i) {
    delete *i;
  }
  delete daughters_;
}

VPlacedVolume* LogicalVolume::Place(
    TransformationMatrix const *const matrix) const {
  return unplaced_volume()->PlaceVolume(this, matrix);
}

VPlacedVolume* LogicalVolume::Place() const {
  return Place(&TransformationMatrix::kIdentity);
}

void LogicalVolume::PlaceDaughter(LogicalVolume const *const volume,
                                  TransformationMatrix const *const matrix) {
  VPlacedVolume const *const placed = volume->Place(matrix);
  daughters_->push_back(placed);
}

VECGEOM_CUDA_HEADER_BOTH
void LogicalVolume::PrintContent(const int depth) const {
  for (int i = 0; i < depth; ++i) printf("  ");
  unplaced_volume()->Print();
  printf("\n");
  for (Iterator<VPlacedVolume const*> vol = daughters_->begin();
       vol != daughters_->end(); ++vol) {
    (*vol)->logical_volume()->PrintContent(depth + 1);
  }
}

std::ostream& operator<<(std::ostream& os, LogicalVolume const &vol) {
  os << *vol.unplaced_volume() << std::endl;
  for (Iterator<VPlacedVolume const*> i = vol.daughters().begin();
       i != vol.daughters().end(); ++i) {
    os << "  " << (**i) << std::endl;
  }
  return os;
}

#ifdef VECGEOM_CUDA

namespace {

__global__
void ConstructOnGpu(VUnplacedVolume const *const unplaced_volume,
                    Vector<Daughter> *const daughters,
                    LogicalVolume *const output) {
  new(output) LogicalVolume(unplaced_volume, daughters);
}

} // End anonymous namespace

LogicalVolume* LogicalVolume::CopyToGpu(
    VUnplacedVolume const *const unplaced_volume,
    Vector<Daughter> *const daughters,
    LogicalVolume *const gpu_ptr) const {

  ConstructOnGpu<<<1, 1>>>(unplaced_volume, daughters, gpu_ptr);
  CudaAssertError();
  return gpu_ptr;

}

LogicalVolume* LogicalVolume::CopyToGpu(
    VUnplacedVolume const *const unplaced_volume,
    Vector<Daughter> *const daughters) const {

  LogicalVolume *const gpu_ptr = AllocateOnGpu<LogicalVolume>();
  return CopyToGpu(unplaced_volume, daughters, gpu_ptr);

}

#endif // VECGEOM_CUDA

} // End namespace vecgeom