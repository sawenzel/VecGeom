/// \file LogicalVolume.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/LogicalVolume.h"

#include "backend/Backend.h"
#ifdef VECGEOM_CUDA
#include "backend/cuda/Interface.h"
#endif
#include "base/Array.h"
#include "base/Transformation3D.h"
#include "base/Vector.h"
#include "management/GeoManager.h"
#include "management/VolumeFactory.h"
#include "volumes/PlacedVolume.h"

#include <cassert>
#include <climits>
#include <stdio.h>

namespace VECGEOM_NAMESPACE {

int LogicalVolume::g_id_count = 0;

#ifndef VECGEOM_NVCC
LogicalVolume::LogicalVolume(char const *const label,
                             VUnplacedVolume const *const unplaced_volume)
  :  unplaced_volume_(unplaced_volume), id_(0), label_(NULL),
     user_extension_(NULL), daughters_() {
  id_ = g_id_count++;
  GeoManager::Instance().RegisterLogicalVolume(this);
  label_ = new std::string(label);
  daughters_ = new Vector<Daughter>();
  }

LogicalVolume::LogicalVolume(LogicalVolume const & other)
  : unplaced_volume_(), id_(0), label_(),
    user_extension_(NULL), daughters_()
{
  printf("COPY CONSTRUCTOR FOR LogicalVolumes NOT IMPLEMENTED");
}

LogicalVolume * LogicalVolume::operator=( LogicalVolume const & other )
{
  printf("COPY CONSTRUCTOR FOR LogicalVolumes NOT IMPLEMENTED");
  return NULL;
}

#endif

LogicalVolume::~LogicalVolume() {
  delete label_;
  for (Daughter* i = daughters().begin(); i != daughters().end();
       ++i) {
    delete *i;
  }
  delete daughters_;
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

VPlacedVolume const* LogicalVolume::PlaceDaughter(
    char const *const label,
    LogicalVolume const *const volume,
    Transformation3D const *const transformation) {
    std::cerr << label << std::endl;
    VPlacedVolume const *const placed = volume->Place(label, transformation);
    daughters_->push_back(placed);
    return placed;
}

VPlacedVolume const* LogicalVolume::PlaceDaughter(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation) {
    return PlaceDaughter(volume->GetLabel().c_str(), volume, transformation);
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
    for (Daughter* i = daughters_->begin(), *i_end = daughters_->end();
        i != i_end; ++i) {
      (*i)->PrintContent(indent+2);
  }}
}

std::ostream& operator<<(std::ostream& os, LogicalVolume const &vol) {
  os << *vol.unplaced_volume() << " [";
  for (Daughter* i = vol.daughters().begin();
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
template <typename Type> class Vector;

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
