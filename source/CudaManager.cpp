/// \file CudaManager.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "management/CudaManager.h"

#include "backend/cuda/Interface.h"
#include "base/Array.h"
#include "base/Stopwatch.h"
#include "management/GeoManager.h"
#include "management/VolumeFactory.h"
#include "volumes/PlacedVolume.h"
#include "volumes/PlacedBooleanVolume.h"

#include <algorithm>
#include <cassert>
#include <stdio.h>

namespace vecgeom {
inline namespace cxx {

CudaManager::CudaManager() {
  synchronized = true;
  world_ = NULL;
  world_gpu_ = NULL;
  verbose_ = 0;
  total_volumes = 0;
}

VPlacedVolume const* CudaManager::world() const {
  assert(world_ != NULL);
  return world_;
}

vecgeom::cuda::VPlacedVolume const* CudaManager::world_gpu() const {
  assert(world_gpu_ != NULL);
  return world_gpu_;
}

vecgeom::cuda::VPlacedVolume const* CudaManager::Synchronize() {

  if (verbose_ > 0) std::cerr << "Starting synchronization to GPU.\n";

  // Will return null if no geometry is loaded
  if (synchronized) return world_gpu_;

  CleanGpu();

  // Populate the memory map with GPU addresses

  AllocateGeometry();

  // Create new objects with pointers adjusted to point to GPU memory, then
  // copy them to the allocated memory locations on the GPU.

  if (verbose_ > 1) std::cerr << "Copying geometry to GPU...";

  if (verbose_ > 2) std::cerr << "\nCopying logical volumes...";
  for (std::set<LogicalVolume const*>::const_iterator i =
       logical_volumes_.begin(); i != logical_volumes_.end(); ++i) {

    (*i)->CopyToGpu(
      LookupUnplaced((*i)->unplaced_volume()),
      LookupDaughters((*i)->daughters_),
      LookupLogical(*i)
    );

  }
  if (verbose_ > 2) std::cerr << " OK\n";

  if (verbose_ > 2) std::cerr << "Copying unplaced volumes...";
  for (std::set<VUnplacedVolume const*>::const_iterator i =
       unplaced_volumes_.begin(); i != unplaced_volumes_.end(); ++i) {

    (*i)->CopyToGpu(LookupUnplaced(*i));

  }
  if (verbose_ > 2) std::cout << " OK\n";

  if (verbose_ > 2) std::cout << "Copying placed volumes...";
  for (std::set<VPlacedVolume const*>::const_iterator i =
       placed_volumes_.begin(); i != placed_volumes_.end(); ++i) {

    (*i)->CopyToGpu(
      LookupLogical((*i)->logical_volume()),
      LookupTransformation((*i)->transformation()),
      LookupPlaced(*i)
    );

  }
  if (verbose_ > 2) std::cout << " OK\n";

  if (verbose_ > 2) std::cout << "Copying transformations_...";
  for (std::set<Transformation3D const*>::const_iterator i =
       transformations_.begin(); i != transformations_.end(); ++i) {

     (*i)->CopyToGpu(LookupTransformation(*i));

  }
  if (verbose_ > 2) std::cout << " OK\n";

  if (verbose_ > 2) std::cout << "Copying daughter arrays...";
  std::vector<CudaDaughter_t> daughter_array;
  for (std::set<Vector<Daughter_t> *>::const_iterator i =
       daughters_.begin(); i != daughters_.end(); ++i) {

    // First handle C arrays that must now point to GPU locations
    const int daughter_count = (*i)->size();
    daughter_array.resize( daughter_count );
    int j = 0;
    for (Daughter_t* k = (*i)->begin(); k != (*i)->end(); ++k) {
      daughter_array[j] = LookupPlaced(*k);
      j++;
    }
    vecgeom::CopyToGpu(
       &(daughter_array[0]), LookupDaughterArray(*i), daughter_count*sizeof(Daughter)
    );

    // Create array object wrapping newly copied C arrays
    (*i)->CopyToGpu(LookupDaughterArray(*i), LookupDaughters(*i));

  }
  if (verbose_ > 1) std::cout << " OK\n";

  synchronized = true;

  world_gpu_ = reinterpret_cast<vecgeom::cuda::VPlacedVolume *>(
    LookupPlaced(world_)
  );

  if (verbose_ > 0) std::cout << "Geometry synchronized to GPU.\n";

  return world_gpu_;

}

void CudaManager::LoadGeometry(VPlacedVolume const *const volume) {

  if (world_ == volume) return;

  CleanGpu();

  logical_volumes_.clear();
  unplaced_volumes_.clear();
  placed_volumes_.clear();
  transformations_.clear();
  daughters_.clear();

  world_ = volume;

  ScanGeometry(volume);

  // Already set by CleanGpu(), but keep it here for good measure
  synchronized = false;

}

void CudaManager::LoadGeometry() {
  LoadGeometry(GeoManager::Instance().GetWorld());
}

void CudaManager::CleanGpu() {

  if (memory_map.size() == 0 && world_gpu_ == NULL) return;

  if (verbose_ > 1) std::cout << "Cleaning GPU...";

  for (auto i = allocated_memory_.begin(), i_end = allocated_memory_.end();
       i != i_end; ++i) {
    FreeFromGpu(*i);
  }
  allocated_memory_.clear();
  memory_map.clear();

  world_gpu_ = NULL;
  synchronized = false;

  if (verbose_ > 1) std::cout << " OK\n";

}

void CudaManager::AllocateGeometry() {

  if (verbose_ > 1) std::cout << "Allocating geometry on GPU...";

  {
    if (verbose_ > 2) std::cout << "Allocating logical volumes...";

    LogicalVolume *gpu_array =
        AllocateOnGpu<LogicalVolume>(
          logical_volumes_.size()*sizeof(LogicalVolume)
        );
    allocated_memory_.push_back(gpu_array);

    for (std::set<LogicalVolume const*>::const_iterator i =
         logical_volumes_.begin(); i != logical_volumes_.end(); ++i) {
      memory_map[ToCpuAddress(*i)] = ToGpuAddress(gpu_array);
      gpu_array++;

    }

    if (verbose_ > 2) std::cout << " OK\n";
  }

  {
    if (verbose_ > 2) std::cout << "Allocating unplaced volumes...";

    for (std::set<VUnplacedVolume const*>::const_iterator i =
         unplaced_volumes_.begin(); i != unplaced_volumes_.end(); ++i) {

      const GpuAddress gpu_address =
          AllocateOnGpu<GpuAddress*>((*i)->memory_size());
      allocated_memory_.push_back(gpu_address);
      memory_map[ToCpuAddress(*i)] = gpu_address;

    }

    if (verbose_ > 2) std::cout << " OK\n";
  }

  {
    if (verbose_ > 2) std::cout << "Allocating placed volumes...";

    for (std::set<VPlacedVolume const*>::const_iterator i =
         placed_volumes_.begin(); i != placed_volumes_.end(); ++i) {

      const GpuAddress gpu_address =
          AllocateOnGpu<GpuAddress*>((*i)->memory_size());
      allocated_memory_.push_back(gpu_address);
      memory_map[ToCpuAddress(*i)] = gpu_address;

    }

    if (verbose_ > 2) std::cout << " OK\n";
  }

  {
    if (verbose_ > 2) std::cout << "Allocating transformations...";

    for (std::set<Transformation3D const*>::const_iterator i =
         transformations_.begin(); i != transformations_.end(); ++i) {

      const GpuAddress gpu_address =
          AllocateOnGpu<Transformation3D>((*i)->memory_size());
      allocated_memory_.push_back(gpu_address);
      memory_map[ToCpuAddress(*i)] = ToGpuAddress(gpu_address);
    }

    if (verbose_ > 2) std::cout << " OK\n";
  }

  {
    if (verbose_ > 2) std::cout << "Allocating daughter lists...";

    Vector<Daughter> *gpu_array =
        AllocateOnGpu<Vector<Daughter> >(
          daughters_.size()*sizeof(Vector<Daughter>)
        );
    allocated_memory_.push_back(gpu_array);

    Daughter *gpu_c_array =
        AllocateOnGpu<Daughter>(total_volumes*sizeof(Daughter));
    allocated_memory_.push_back(gpu_c_array);

    for (std::set<Vector<Daughter> *>::const_iterator i =
         daughters_.begin(); i != daughters_.end(); ++i) {

      memory_map[ToCpuAddress(*i)] = ToGpuAddress(gpu_array);
      memory_map[ToCpuAddress(gpu_array)] = ToGpuAddress(gpu_c_array);
      gpu_array++;
      gpu_c_array += (*i)->size();

    }

    if (verbose_ > 2) std::cout << " OK\n";
  }

  if (verbose_ == 2) std::cout << " OK\n";


  fprintf(stderr,"NUMBER OF PLACED VOLUMES %ld\n", placed_volumes_.size());
  fprintf(stderr,"NUMBER OF UNPLACED VOLUMES %ld\n", unplaced_volumes_.size());

}

void CudaManager::ScanGeometry(VPlacedVolume const *const volume) {

  if (placed_volumes_.find(volume) == placed_volumes_.end()) {
    placed_volumes_.insert(volume);
  }
  if (logical_volumes_.find(volume->logical_volume())
      == logical_volumes_.end()) {
    logical_volumes_.insert(volume->logical_volume());
  }
  if (transformations_.find(volume->transformation())
      == transformations_.end()) {
    transformations_.insert(volume->transformation());
  }
  if (unplaced_volumes_.find(volume->unplaced_volume())
      == unplaced_volumes_.end()) {
    unplaced_volumes_.insert(volume->unplaced_volume());
  }
  if (daughters_.find(volume->logical_volume()->daughters_)
      == daughters_.end()) {
    daughters_.insert(volume->logical_volume()->daughters_);
  }

  if( dynamic_cast<PlacedBooleanVolume const*>(volume) ){
    fprintf(stderr,"found a PlacedBooleanVolume");
    PlacedBooleanVolume const* v =  dynamic_cast<PlacedBooleanVolume const*>(volume);
    ScanGeometry(v->GetUnplacedVolume()->fLeftVolume);
    ScanGeometry(v->GetUnplacedVolume()->fRightVolume);
  }


  for (Daughter_t* i = volume->daughters().begin();
       i != volume->daughters().end(); ++i) {
    ScanGeometry(*i);
  }

  total_volumes++;
}

template <typename Type>
typename CudaManager::GpuAddress CudaManager::Lookup(
    Type const *const key) {
  const CpuAddress cpu_address = ToCpuAddress(key);
  GpuAddress output = memory_map[cpu_address];
  assert(output != NULL);
  return output;
}

DevicePtr<VUnplacedVolume> CudaManager::LookupUnplaced(
    VUnplacedVolume const *const host_ptr) {
  return DevicePtr<VUnplacedVolume>(Lookup(host_ptr));
}

DevicePtr<LogicalVolume> CudaManager::LookupLogical(
    LogicalVolume const *const host_ptr) {
  return DevicePtr<LogicalVolume>(Lookup(host_ptr));
}

DevicePtr<VPlacedVolume> CudaManager::LookupPlaced(
    VPlacedVolume const *const host_ptr) {
  return DevicePtr<VPlacedVolume>(Lookup(host_ptr));
}

DevicePtr<Transformation3D> CudaManager::LookupTransformation(
    Transformation3D const *const host_ptr) {
  return DevicePtr<Transformation3D>(Lookup(host_ptr));
}

DevicePtr<cuda::Vector<CudaDaughter_t> > CudaManager::LookupDaughters(
    Vector<Daughter> *const host_ptr) {
  return static_cast<Vector<Daughter>*>(Lookup(host_ptr));
}

DevicePtr<CudaDaughter_t> CudaManager::LookupDaughterArray(
    Vector<Daughter> *const host_ptr) {
  Vector<Daughter> const *const daughters_ = LookupDaughters(host_ptr);
  return static_cast<Daughter*>(Lookup(daughters_));
}

void CudaManager::PrintGeometry() const {
  CudaManagerPrintGeometry(world_gpu());
}

// template <typename TrackContainer>
// void CudaManager::LocatePointsTemplate(TrackContainer const &container,
//                                        const int n, const int depth,
//                                        int *const output) const {
//   CudaManagerLocatePoints(world_gpu(), container, n, depth, output);
// }

// void CudaManager::LocatePoints(SOA3D<Precision> const &container,
//                                const int depth, int *const output) const {
//   Precision *const x_gpu =
//       AllocateOnGpu<Precision>(sizeof(Precision)*container.size());
//   Precision *const y_gpu =
//       AllocateOnGpu<Precision>(sizeof(Precision)*container.size());
//   Precision *const z_gpu =
//       AllocateOnGpu<Precision>(sizeof(Precision)*container.size());
//   SOA3D<Precision> *const soa3d_gpu = container.CopyToGpu(x_gpu, y_gpu, z_gpu);
//   LocatePointsTemplate(soa3d_gpu, container.size(), depth, output);
//   CudaFree(x_gpu);
//   CudaFree(y_gpu);
//   CudaFree(z_gpu);
//   CudaFree(soa3d_gpu);
// }

// void CudaManager::LocatePoints(AOS3D<Precision> const &container,
//                                const int depth, int *const output) const {
//   Vector3D<Precision> *const data =
//       AllocateOnGpu<Vector3D<Precision> >(
//         container.size()*sizeof(Vector3D<Precision>)
//       );
//   AOS3D<Precision> *const aos3d_gpu = container.CopyToGpu(data);
//   LocatePointsTemplate(aos3d_gpu, container.size(), depth, output);
//   CudaFree(data);
//   CudaFree(aos3d_gpu);
// }

} } // End namespace vecgeom
