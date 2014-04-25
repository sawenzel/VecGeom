/**
 * @file logical_volume.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include <stdio.h>
#include <climits>

#include "backend/backend.h"
#include "base/array.h"
#include "base/transformation3d.h"
#include "management/volume_factory.h"
#include "volumes/logical_volume.h"
#include "volumes/placed_volume.h"

namespace VECGEOM_NAMESPACE {

int LogicalVolume::g_id_count = 0;
std::list<LogicalVolume *> LogicalVolume::g_volume_list =
    std::list<LogicalVolume *>();

LogicalVolume::~LogicalVolume() {
  delete label_;
  for (Iterator<VPlacedVolume const*> i = daughters().begin();
       i != daughters().end(); ++i) {
    delete *i;
  }
  delete daughters_;
  g_volume_list.remove(this);
}

#ifndef VECGEOM_NVCC

VPlacedVolume* LogicalVolume::Place(
    char const *const label,
    Transformation3D const *const transformation) const {
  return unplaced_volume()->PlaceVolume(label, this, transformation);
}

VPlacedVolume* LogicalVolume::Place(
    Transformation3D const *const transformation) const {
  return Place(label_->c_str(), transformation);
}

VPlacedVolume* LogicalVolume::Place(char const *const label) const {
  return unplaced_volume()->PlaceVolume(
           label, this, &Transformation3D::kIdentity
         );
}

VPlacedVolume* LogicalVolume::Place() const {
  return Place(label_->c_str());
}

void LogicalVolume::PlaceDaughter(
    char const *const label,
    LogicalVolume const *const volume,
    Transformation3D const *const transformation) {
  VPlacedVolume const *const placed = volume->Place(label, transformation);
  daughters_->push_back(placed);
}

void LogicalVolume::PlaceDaughter(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation) {
  PlaceDaughter(volume->label().c_str(), volume, transformation);
}

void LogicalVolume::PlaceDaughter(VPlacedVolume const *const placed) {
  daughters_->push_back(placed);
}

#endif

VECGEOM_CUDA_HEADER_BOTH
void LogicalVolume::Print(const int indent) const {
  for (int i = 0; i < indent; ++i) printf("  ");
  printf("LogicalVolume [%i]", id_);
#ifndef VECGEOM_NVCC
  if (label_->size()) {
    printf(" \"%s\"", label_->c_str());
  }
#endif
  printf(":\n");
  for (int i = 0; i <= indent; ++i) printf("  ");
  unplaced_volume_->Print();
  printf("\n");
  for (int i = 0; i <= indent; ++i) printf("  ");
  if( daughters_->size() > 0){
     printf("Contains %i daughter", daughters_->size());
     if (daughters_->size() != 1) printf("s");
  }
}

VECGEOM_CUDA_HEADER_BOTH
void LogicalVolume::PrintContent(const int indent) const {
  for (int i = 0; i < indent; ++i) printf("  ");
  Print(indent);
  if( daughters_->size() > 0){
    printf(":");
    for (Iterator<Daughter> i = daughters_->begin(), i_end = daughters_->end();
        i != i_end; ++i) {
      (*i)->PrintContent(indent+2);
  }}
}

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

void LogicalVolume_CopyToGpu(VUnplacedVolume const *const unplaced_volume,
                             Vector<VPlacedVolume const*> *const daughters,
                             LogicalVolume *const output);

LogicalVolume* LogicalVolume::CopyToGpu(
    VUnplacedVolume const *const unplaced_volume,
    Vector<Daughter> *const daughters,
    LogicalVolume *const gpu_ptr) const {

  LogicalVolume_CopyToGpu(unplaced_volume, daughters, gpu_ptr);
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
    reinterpret_cast<vecgeom_cuda::VUnplacedVolume const*>(unplaced_volume),
    reinterpret_cast<vecgeom_cuda::Vector<vecgeom_cuda::VPlacedVolume const*>*>(
      daughters
    )
  );
}

void LogicalVolume_CopyToGpu(VUnplacedVolume const *const unplaced_volume,
                             Vector<VPlacedVolume const*> *const daughters,
                             LogicalVolume *const gpu_ptr) {
  ConstructOnGpu<<<1, 1>>>(unplaced_volume, daughters, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom
