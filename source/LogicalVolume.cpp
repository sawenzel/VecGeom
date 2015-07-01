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

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

int LogicalVolume::gIdCount = 0;

#ifndef VECGEOM_NVCC
LogicalVolume::LogicalVolume(char const *const label,
                             VUnplacedVolume const *const unplaced_volume)
  :  fUnplacedVolume(unplaced_volume), fId(0), fLabel(NULL),
     fUserExtensionPtr(NULL), fDaughters() {
  fId = gIdCount++;
  GeoManager::Instance().RegisterLogicalVolume(this);
  fLabel = new std::string(label);
  fDaughters = new Vector<Daughter>();
  }

LogicalVolume::LogicalVolume(LogicalVolume const & other)
  : fUnplacedVolume(), fId(0), fLabel(NULL),
    fUserExtensionPtr(NULL), fDaughters()
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
  delete fLabel;
  for (Daughter* i = GetDaughters().begin(); i != GetDaughters().end();
       ++i) {
    delete *i;
  }
  delete fDaughters;
}

#ifndef VECGEOM_NVCC

VPlacedVolume* LogicalVolume::Place(
    char const *const label,
    Transformation3D const *const transformation) const {
  return GetUnplacedVolume()->PlaceVolume(label, this, transformation);
}

VPlacedVolume* LogicalVolume::Place(
    Transformation3D const *const transformation) const {
  return Place(fLabel->c_str(), transformation);
}

VPlacedVolume* LogicalVolume::Place(char const *const label) const {
  return GetUnplacedVolume()->PlaceVolume(
           label, this, &Transformation3D::kIdentity
         );
}

VPlacedVolume* LogicalVolume::Place() const {
  return Place(fLabel->c_str());
}

VPlacedVolume const* LogicalVolume::PlaceDaughter(
    char const *const label,
    LogicalVolume const *const volume,
    Transformation3D const *const transformation) {
    VPlacedVolume const *const placed = volume->Place(label, transformation);
    //  std::cerr << label <<" LogVol@"<< this <<" and placed@"<< placed << std::endl;
    fDaughters->push_back(placed);
    return placed;
}

VPlacedVolume const* LogicalVolume::PlaceDaughter(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation) {
    return PlaceDaughter(volume->GetLabel().c_str(), volume, transformation);
}

void LogicalVolume::PlaceDaughter(VPlacedVolume const *const placed) {
  fDaughters->push_back(placed);
}

#endif

VECGEOM_CUDA_HEADER_BOTH
void LogicalVolume::Print(const int indent) const {
  for (int i = 0; i < indent; ++i) printf("  ");
  printf("LogicalVolume [%i]", fId);
#ifndef VECGEOM_NVCC
  if (fLabel->size()) {
    printf(" \"%s\"", fLabel->c_str());
  }
#endif
  printf(":\n");
  for (int i = 0; i <= indent; ++i) printf("  ");
  fUnplacedVolume->Print();
  printf("\n");
  for (int i = 0; i <= indent; ++i) printf("  ");
  if( fDaughters->size() > 0){
     printf("Contains %i daughter", fDaughters->size());
     if (fDaughters->size() != 1) printf("s");
  }
}

VECGEOM_CUDA_HEADER_BOTH
void LogicalVolume::PrintContent(const int indent) const {
  for (int i = 0; i < indent; ++i) printf("  ");
  Print(indent);
  if( fDaughters->size() > 0){
    printf(":");
    for (Daughter* i = fDaughters->begin(), *i_end = fDaughters->end();
        i != i_end; ++i) {
      (*i)->PrintContent(indent+2);
  }}
}

std::ostream& operator<<(std::ostream& os, LogicalVolume const &vol) {
  os << *vol.GetUnplacedVolume() << " [";
  for (Daughter* i = vol.GetDaughters().begin();
       i != vol.GetDaughters().end(); ++i) {
    if (i != vol.GetDaughters().begin()) os << ", ";
    os << (**i);
  }
  os << "]";
  return os;
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::LogicalVolume> LogicalVolume::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const unplaced_vol,
   DevicePtr<cuda::Vector<CudaDaughter_t>> GetDaughter,
   DevicePtr<cuda::LogicalVolume> const gpu_ptr) const
{
   gpu_ptr.Construct( unplaced_vol, GetDaughter );
   CudaAssertError();
   return gpu_ptr;
}

DevicePtr<cuda::LogicalVolume> LogicalVolume::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const unplaced_vol,
   DevicePtr<cuda::Vector<CudaDaughter_t>> daughter) const
{
   DevicePtr<cuda::LogicalVolume> gpu_ptr;
   gpu_ptr.Allocate();
   return this->CopyToGpu(unplaced_vol,daughter,gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::LogicalVolume>::SizeOf();
template void DevicePtr<cuda::LogicalVolume>::Construct(
    DevicePtr<cuda::VUnplacedVolume> const,
    DevicePtr<cuda::Vector<cuda::VPlacedVolume const*>>) const;

}

#endif // VECGEOM_NVCC

} // End global namespace
