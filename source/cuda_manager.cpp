/**
 * @file cuda_manager.cu
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include <algorithm>
#include <cassert>
#include "backend/cuda/interface.h"
#include "base/array.h"
#include "management/cuda_manager.h"
#include "management/volume_factory.h"
#include "volumes/placed_volume.h"

namespace vecgeom {

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

VPlacedVolume const* CudaManager::world_gpu() const {
  assert(world_gpu_ != NULL);
  return world_gpu_;
}

VPlacedVolume const* CudaManager::Synchronize() {

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
       logical_volumes.begin(); i != logical_volumes.end(); ++i) {

    (*i)->CopyToGpu(
      LookupUnplaced((*i)->unplaced_volume()),
      LookupDaughters((*i)->daughters_),
      LookupLogical(*i)
    );

  }
  if (verbose_ > 2) std::cerr << " OK\n";

  if (verbose_ > 2) std::cerr << "Copying unplaced volumes...";
  for (std::set<VUnplacedVolume const*>::const_iterator i =
       unplaced_volumes.begin(); i != unplaced_volumes.end(); ++i) {

    (*i)->CopyToGpu(LookupUnplaced(*i));

  }
  if (verbose_ > 2) std::cout << " OK\n";

  if (verbose_ > 2) std::cout << "Copying placed volumes...";
  for (std::set<VPlacedVolume const*>::const_iterator i =
       placed_volumes.begin(); i != placed_volumes.end(); ++i) {

    (*i)->CopyToGpu(
      LookupLogical((*i)->logical_volume()),
      LookupMatrix((*i)->matrix()),
      LookupPlaced(*i)
    );

  }
  if (verbose_ > 2) std::cout << " OK\n";

  if (verbose_ > 2) std::cout << "Copying transformation matrices...";
  for (std::set<TransformationMatrix const*>::const_iterator i =
       matrices.begin(); i != matrices.end(); ++i) {

    (*i)->CopyToGpu(LookupMatrix(*i));

  }
  if (verbose_ > 2) std::cout << " OK\n";

  if (verbose_ > 2) std::cout << "Copying daughter arrays...";
  for (std::set<Vector<Daughter> *>::const_iterator i =
       daughters.begin(); i != daughters.end(); ++i) {

    // First handle C arrays that must now point to GPU locations
    const int daughter_count = (*i)->size();
    Daughter *const daughter_array = new Daughter[daughter_count];
    int j = 0;
    for (Iterator<Daughter> k = (*i)->begin(); k != (*i)->end(); ++k) {
      daughter_array[j] = LookupPlaced(*k);
      j++;
    }
    vecgeom::CopyToGpu(
      daughter_array, LookupDaughterArray(*i), daughter_count*sizeof(Daughter)
    );

    // Create array object wrapping newly copied C arrays
    (*i)->CopyToGpu(LookupDaughterArray(*i), LookupDaughters(*i));

  }
  if (verbose_ > 1) std::cout << " OK\n";

  synchronized = true;

  world_gpu_ = LookupPlaced(world_);

  if (verbose_ > 0) std::cout << "Geometry synchronized to GPU.\n";

  return world_gpu_;

}

void CudaManager::LoadGeometry(VPlacedVolume const *const volume) {

  CleanGpu();

  logical_volumes.clear();
  unplaced_volumes.clear();
  placed_volumes.clear();
  matrices.clear();
  daughters.clear();

  world_ = volume;

  ScanGeometry(volume);

  // Already set by CleanGpu(), but keep it here for good measure
  synchronized = false;

}

void CudaManager::CleanGpu() {

  if (memory_map.size() == 0 && world_gpu_ == NULL) return;

  if (verbose_ > 1) std::cout << "Cleaning GPU...";

  for (MemoryMap::iterator i = memory_map.begin(); i != memory_map.end(); ++i) {
    FreeFromGpu(i->second);
  }
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
          logical_volumes.size()*sizeof(LogicalVolume)
        );

    for (std::set<LogicalVolume const*>::const_iterator i =
         logical_volumes.begin(); i != logical_volumes.end(); ++i) {

      memory_map[ToCpuAddress(*i)] = ToGpuAddress(gpu_array);
      gpu_array++;

    }

    if (verbose_ > 2) std::cout << " OK\n";
  }

  {
    if (verbose_ > 2) std::cout << "Allocating unplaced volumes...";

    for (std::set<VUnplacedVolume const*>::const_iterator i =
         unplaced_volumes.begin(); i != unplaced_volumes.end(); ++i) {

      const GpuAddress gpu_address =
          AllocateOnGpu<GpuAddress*>((*i)->memory_size());
      memory_map[ToCpuAddress(*i)] = gpu_address;

    }

    if (verbose_ > 2) std::cout << " OK\n";
  }

  {
    if (verbose_ > 2) std::cout << "Allocating placed volumes...";

    for (std::set<VPlacedVolume const*>::const_iterator i =
         placed_volumes.begin(); i != placed_volumes.end(); ++i) {

      const GpuAddress gpu_address =
          AllocateOnGpu<GpuAddress*>((*i)->memory_size());
      memory_map[ToCpuAddress(*i)] = gpu_address;

    }

    if (verbose_ > 2) std::cout << " OK\n";
  }

  {
    if (verbose_ > 2) std::cout << "Allocating transformation matrices...";

    for (std::set<TransformationMatrix const*>::const_iterator i =
         matrices.begin(); i != matrices.end(); ++i) {

      const GpuAddress gpu_address =
          AllocateOnGpu<TransformationMatrix>((*i)->memory_size());
      memory_map[ToCpuAddress(*i)] = ToGpuAddress(gpu_address);

    }

    if (verbose_ > 2) std::cout << " OK\n";
  }

  {
    if (verbose_ > 2) std::cout << "Allocating daughter lists...";

    Vector<Daughter> *gpu_array =
        AllocateOnGpu<Vector<Daughter> >(
          daughters.size()*sizeof(Vector<Daughter>)
        );

    Daughter *gpu_c_array =
        AllocateOnGpu<Daughter>(total_volumes*sizeof(Daughter));

    for (std::set<Vector<Daughter> *>::const_iterator i =
         daughters.begin(); i != daughters.end(); ++i) {

      memory_map[ToCpuAddress(*i)] = ToGpuAddress(gpu_array);
      memory_map[ToCpuAddress(gpu_array)] = ToGpuAddress(gpu_c_array);
      gpu_array++;
      gpu_c_array += (*i)->size();

    }

    if (verbose_ > 2) std::cout << " OK\n";
  }

  if (verbose_ == 2) std::cout << " OK\n";

}

void CudaManager::ScanGeometry(VPlacedVolume const *const volume) {

  if (placed_volumes.find(volume) == placed_volumes.end()) {
    placed_volumes.insert(volume);
  }
  if (logical_volumes.find(volume->logical_volume()) == logical_volumes.end()) {
    logical_volumes.insert(volume->logical_volume());
  }
  if (matrices.find(volume->matrix()) == matrices.end()) {
    matrices.insert(volume->matrix());
  }
  if (unplaced_volumes.find(volume->unplaced_volume())
      == unplaced_volumes.end()) {
    unplaced_volumes.insert(volume->unplaced_volume());
  }
  if (daughters.find(volume->logical_volume()->daughters_) == daughters.end()) {
    daughters.insert(volume->logical_volume()->daughters_);
  }
  for (Iterator<Daughter> i = volume->daughters().begin();
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

VUnplacedVolume* CudaManager::LookupUnplaced(
    VUnplacedVolume const *const host_ptr) {
  return static_cast<VUnplacedVolume*>(Lookup(host_ptr));
}

LogicalVolume* CudaManager::LookupLogical(
    LogicalVolume const *const host_ptr) {
  return static_cast<LogicalVolume*>(Lookup(host_ptr));
}

VPlacedVolume* CudaManager::LookupPlaced(
    VPlacedVolume const *const host_ptr) {
  return static_cast<VPlacedVolume*>(Lookup(host_ptr));
}

TransformationMatrix* CudaManager::LookupMatrix(
    TransformationMatrix const *const host_ptr) {
  return static_cast<TransformationMatrix*>(Lookup(host_ptr));
}

Vector<Daughter>* CudaManager::LookupDaughters(
    Vector<Daughter> *const host_ptr) {
  return static_cast<Vector<Daughter>*>(Lookup(host_ptr));
}

Daughter* CudaManager::LookupDaughterArray(
    Vector<Daughter> *const host_ptr) {
  Vector<Daughter> const *const daughters = LookupDaughters(host_ptr);
  return static_cast<Daughter*>(Lookup(daughters));
}

void CudaManager::PrintGeometry() const {
  CudaManagerPrintGeometry(world_gpu());
}

template <typename TrackContainer>
void CudaManager::LocatePointsTemplate(TrackContainer const &container,
                                       const int n, const int depth,
                                       int *const output) const {
  CudaManagerLocatePoints(world_gpu(), container, n, depth, output);
}

void CudaManager::LocatePoints(SOA3D<Precision> const &container,
                               const int depth, int *const output) const {
  Precision *const x_gpu =
      AllocateOnGpu<Precision>(sizeof(Precision)*container.size());
  Precision *const y_gpu =
      AllocateOnGpu<Precision>(sizeof(Precision)*container.size());
  Precision *const z_gpu =
      AllocateOnGpu<Precision>(sizeof(Precision)*container.size());
  SOA3D<Precision> *const soa3d_gpu = container.CopyToGpu(x_gpu, y_gpu, z_gpu);
  LocatePointsTemplate(soa3d_gpu, container.size(), depth, output);
  CudaFree(x_gpu);
  CudaFree(y_gpu);
  CudaFree(z_gpu);
  CudaFree(soa3d_gpu);
}

void CudaManager::LocatePoints(AOS3D<Precision> const &container,
                               const int depth, int *const output) const {
  Vector3D<Precision> *const data =
      AllocateOnGpu<Vector3D<Precision> >(
        container.size()*sizeof(Vector3D<Precision>)
      );
  AOS3D<Precision> *const aos3d_gpu = container.CopyToGpu(data);
  LocatePointsTemplate(aos3d_gpu, container.size(), depth, output);
  CudaFree(data);
  CudaFree(aos3d_gpu);
}

} // End global namespace