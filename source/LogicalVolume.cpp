// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \file LogicalVolume.cpp
/// \author created by Johannes de Fine Licht, Sandro Wenzel (CERN)

#include "VecGeom/volumes/LogicalVolume.h"

#ifdef VECGEOM_ENABLE_CUDA
#include "VecGeom/backend/cuda/Interface.h"
#endif
#include "VecGeom/base/Assert.h"
#include "VecGeom/base/Array.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Vector.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/navigation/SimpleSafetyEstimator.h"
#include "VecGeom/navigation/NewSimpleNavigator.h"
#include "VecGeom/navigation/SimpleLevelLocator.h"
#include "VecGeom/volumes/UnplacedAssembly.h"
#include "VecGeom/volumes/UnplacedBooleanVolume.h"
#include <climits>
#include <stdio.h>
#include <set>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// NOTE: This is in the wrong place (but SimpleSafetyEstimator does not yet have a source file).
#ifdef VECCORE_CUDA
VECCORE_ATT_DEVICE
SimpleSafetyEstimator *gSimpleSafetyEstimator = nullptr;

VECCORE_ATT_DEVICE
VSafetyEstimator *SimpleSafetyEstimator::Instance()
{
  if (gSimpleSafetyEstimator == nullptr) gSimpleSafetyEstimator = new SimpleSafetyEstimator();
  return gSimpleSafetyEstimator;
}
#endif

#ifdef VECCORE_CUDA
VECCORE_ATT_DEVICE
VNavigator *gSimpleNavigator = nullptr;

template <>
VECCORE_ATT_DEVICE VNavigator *NewSimpleNavigator<false>::Instance()
{
  if (gSimpleNavigator == nullptr) gSimpleNavigator = new NewSimpleNavigator();
  return gSimpleNavigator;
}
#endif

int LogicalVolume::gIdCount = 0;

#ifndef VECCORE_CUDA
LogicalVolume::LogicalVolume(char const *const label, VUnplacedVolume const *const unplaced_volume)
    : fUnplacedVolume(unplaced_volume), fId(0), fLabel(nullptr),
      fLevelLocator(SimpleAssemblyLevelLocator::GetInstance()), fSafetyEstimator(SimpleSafetyEstimator::Instance()),
      fNavigator(NewSimpleNavigator<>::Instance()), fDaughters()
{
  fId = gIdCount++;
  GeoManager::Instance().RegisterLogicalVolume(this);
  fLabel     = new std::string(label);
  fDaughters = new Vector<Daughter>();

  // if the definint unplaced volume is an assembly, we need to make the back connection
  // I have chosen this implicit method for user convenience (in disfavour of an explicit function call)
  if (unplaced_volume->IsAssembly()) {
    (const_cast<UnplacedAssembly *>(static_cast<UnplacedAssembly const *const>(unplaced_volume)))
        ->SetLogicalVolume(this);
  }
}

#else
VECCORE_ATT_DEVICE
LogicalVolume::LogicalVolume(VUnplacedVolume const *const unplaced_vol, unsigned int id, Vector<Daughter> *GetDaughter)
    // Id for logical volumes is not needed on the device for CUDA
    : fUnplacedVolume(unplaced_vol), fId(id), fLabel(nullptr), fDaughters(GetDaughter), fLevelLocator(nullptr),
      fSafetyEstimator(nullptr), fNavigator(nullptr)
{
}

#endif

LogicalVolume::~LogicalVolume()
{
  delete fLabel;
  for (Daughter *i = GetDaughters().begin(); i != GetDaughters().end(); ++i) {
    // delete *i;
  }
#ifndef VECCORE_CUDA // this guard might have to be extended
  GeoManager::Instance().DeregisterLogicalVolume(fId);
#endif
  delete fDaughters;
}

#ifndef VECCORE_CUDA

VPlacedVolume *LogicalVolume::Place(char const *const label, Transformation3D const *const transformation) const
{
  return GetUnplacedVolume()->PlaceVolume(label, this, transformation);
}

VPlacedVolume *LogicalVolume::Place(Transformation3D const *const transformation) const
{
  return Place(fLabel->c_str(), transformation);
}

VPlacedVolume *LogicalVolume::Place(char const *const label) const
{
  return Place(label, &Transformation3D::kIdentity);
}

VPlacedVolume *LogicalVolume::Place() const { return Place(fLabel->c_str()); }

VPlacedVolume const *LogicalVolume::PlaceDaughter(char const *const label, LogicalVolume *const volume,
                                                  Transformation3D const *const transformation)
{
  VPlacedVolume *const placed = volume->Place(label, transformation);
  //  std::cerr << label <<" LogVol@"<< this <<" and placed@"<< placed << std::endl;
  PlaceDaughter(placed);
  return placed;
}

VPlacedVolume const *LogicalVolume::PlaceDaughter(LogicalVolume *const volume,
                                                  Transformation3D const *const transformation)
{
  return PlaceDaughter(volume->GetLabel().c_str(), volume, transformation);
}

#endif

void LogicalVolume::PlaceDaughter(VPlacedVolume *const placed)
{
#ifndef VECCORE_CUDA
  VECGEOM_VALIDATE(BooleanHelper::IsFiniteBody(placed), << "Volumes placed in geometry must be finite bodies");
#endif
  int ichild = fDaughters->size();
  VECGEOM_ASSERT(
      placed->GetChildId() < 0 &&
      "===FFF=== LogicalVolume::PlaceDaughter: Not allowed to add the same placed volume twice - make a copy first");
  placed->SetChildId(ichild);
  fDaughters->push_back(placed);

  // a good moment to update the bounding boxes
  // in case this thing is an assembly
  if (fUnplacedVolume->IsAssembly()) {
    static_cast<UnplacedAssembly *>(const_cast<VUnplacedVolume *>((GetUnplacedVolume())))->UpdateExtent();
  }
}

// void LogicalVolume::SetDaughter(unsigned int i, VPlacedVolume const *pvol) { daughters_->operator[](i) = pvol; }

VECCORE_ATT_HOST_DEVICE
void LogicalVolume::Print(const int indent) const
{
  for (int i = 0; i < indent; ++i)
    printf("  ");
  printf("LogicalVolume [%i]", fId);
#ifndef VECCORE_CUDA
  if (fLabel->size()) {
    printf(" \"%s\"", fLabel->c_str());
  }
#endif
  printf(":\n");
  for (int i = 0; i <= indent; ++i)
    printf("  ");
  fUnplacedVolume->Print();
  printf("\n");
  for (int i = 0; i <= indent; ++i)
    printf("  ");
  if (fDaughters->size() > 0) {
    printf("Contains %zu daughter", fDaughters->size());
    if (fDaughters->size() != 1) printf("s");
  }
}

VECCORE_ATT_HOST_DEVICE
void LogicalVolume::PrintContent(const int indent) const
{
  for (int i = 0; i < indent; ++i)
    printf("  ");
  Print(indent);
  if (fDaughters->size() > 0) {
    printf(":");
    for (Daughter *i = fDaughters->begin(), *i_end = fDaughters->end(); i != i_end; ++i) {
      (*i)->PrintContent(indent + 2);
    }
  }
}

bool LogicalVolume::ContainsAssembly() const
{
  for (Daughter *i = GetDaughters().begin(); i != GetDaughters().end(); ++i) {
    if ((*i)->GetUnplacedVolume()->IsAssembly()) {
      return true;
    }
  }
  return false;
}

std::set<LogicalVolume *> LogicalVolume::GetSetOfDaughterLogicalVolumes() const
{
  std::set<LogicalVolume *> s;
  for (auto pv : GetDaughters()) {
    s.insert(const_cast<LogicalVolume *>(pv->GetLogicalVolume()));
  }
  return s;
}

std::ostream &operator<<(std::ostream &os, LogicalVolume const &vol)
{
  os << *vol.GetUnplacedVolume() << " [";
  for (Daughter *i = vol.GetDaughters().begin(); i != vol.GetDaughters().end(); ++i) {
    if (i != vol.GetDaughters().begin()) os << ", ";
    os << (**i);
  }
  os << "]";
  return os;
}

size_t LogicalVolume::GetNTotal() const
{
  size_t accum(fDaughters->size());
  for (size_t d = 0; d < fDaughters->size(); ++d) {
    accum += fDaughters->operator[](d)->GetLogicalVolume()->GetNTotal();
  }
  return accum;
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::LogicalVolume> LogicalVolume::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const unplaced_vol, int id,
                                                        DevicePtr<cuda::Vector<CudaDaughter_t>> GetDaughter,
                                                        DevicePtr<cuda::LogicalVolume> const gpu_ptr) const
{
  gpu_ptr.Construct(unplaced_vol, id, GetDaughter);
  VECGEOM_DEVICE_API_CALL(GetLastError());
  return gpu_ptr;
}

DevicePtr<cuda::LogicalVolume> LogicalVolume::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const unplaced_vol, int id,
                                                        DevicePtr<cuda::Vector<CudaDaughter_t>> daughter) const
{
  DevicePtr<cuda::LogicalVolume> gpu_ptr;
  gpu_ptr.Allocate();
  return this->CopyToGpu(unplaced_vol, id, daughter, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::LogicalVolume>::SizeOf();
template void DevicePtr<cuda::LogicalVolume>::Construct(DevicePtr<cuda::VUnplacedVolume> const, int,
                                                        DevicePtr<cuda::Vector<cuda::VPlacedVolume const *>>) const;
} // namespace cxx

#endif // VECCORE_CUDA

} // namespace vecgeom
