#include <algorithm>
#include "management/cuda_manager.h"
#include "backend/cuda_backend.cuh"
#include "base/array.h"

namespace vecgeom {

/**
 * Synchronizes the loaded geometry to the GPU by allocating space, creating new
 * objects with correct pointers, then copying them to the GPU.
 */
void CudaManager::Synchronize() {

  if (synchronized) return;

  CleanGpu();

  // Populate the memory map with GPU addresses

  AllocateGeometry();

  // Create new objects with pointers adjusted to point to GPU memory, then
  // copy them to the allocated memory locations on the GPU.

  for (std::set<LogicalVolume const*>::const_iterator i =
       logical_volumes.begin(); i != logical_volumes.end(); ++i) {

    LogicalVolume *gpu_object = new LogicalVolume(
      static_cast<VUnplacedVolume const*>(
        memory_map[ToCpuAddress(&(*i)->unplaced_volume())]
      ),
      static_cast<Container<VPlacedVolume const*> *>(
        memory_map[ToCpuAddress(&(*i)->daughters())]
      )
    );

    CopyToGpu(
      gpu_object, static_cast<LogicalVolume*>(memory_map[ToCpuAddress(*i)]), 1
    );

    delete gpu_object;

  }

  for (std::set<VUnplacedVolume const*>::const_iterator i =
       unplaced_volumes.begin(); i != unplaced_volumes.end(); ++i) {

    (*i)->CopyToGpu(
      static_cast<VUnplacedVolume*>(memory_map[ToCpuAddress(*i)])
    );

  }

  for (std::set<VPlacedVolume const*>::const_iterator i =
       placed_volumes.begin(); i != placed_volumes.end(); ++i) {

    VPlacedVolume *gpu_object = new VPlacedVolume(
      static_cast<LogicalVolume const*>(
        memory_map[ToCpuAddress(&(*i)->logical_volume())]
      ),
      static_cast<TransformationMatrix const*>(
        memory_map[ToCpuAddress(&(*i)->matrix())]
      )
    );

    CopyToGpu(
      gpu_object, static_cast<VPlacedVolume*>(memory_map[ToCpuAddress(*i)]), 1
    );

    delete gpu_object;

  }

  for (std::set<TransformationMatrix const*>::const_iterator i =
       matrices.begin(); i != matrices.end(); ++i) {

    CopyToGpu(
      *i, static_cast<TransformationMatrix*>(memory_map[ToCpuAddress(*i)]), 1
    );

  }

  for (std::set<Container<Daughter> *>::const_iterator i =
       daughters.begin(); i != daughters.end(); ++i) {

    // First copy the C-arrays
    const int n_daughters =
        (static_cast<Vector<Daughter> *>(*i))->size();
    Daughter *arr = new Daughter[n_daughters];
    int j = 0;
    for (Iterator<Daughter> k = (*i)->begin();
         k != (*i)->end(); ++k) {
      arr[j] = *k;
      j++;
    }
    VPlacedVolume const **const arr_gpu = static_cast<VPlacedVolume const **>(
      memory_map[ToCpuAddress(memory_map[ToCpuAddress(*i)])]
    );
    CopyToGpu(arr, arr_gpu, n_daughters);
    delete arr;

    // Then the array containers
    Array<Daughter> *gpu_object =
        new Array<Daughter>(arr_gpu, n_daughters);
    CopyToGpu(
      gpu_object,
      static_cast<Array<Daughter> *>(memory_map[ToCpuAddress(*i)]),
      1
    );
    delete gpu_object;

  }

  synchronized = true;

}

/**
 * Stages a new geometry to be copied to the GPU.
 */
void CudaManager::LoadGeometry(LogicalVolume const &volume) {

  CleanGpu();

  logical_volumes.clear();
  unplaced_volumes.clear();
  placed_volumes.clear();
  matrices.clear();
  daughters.clear();

  ScanGeometry(volume);

  // Already set by CleanGpu(), but it'll stay here for good measure
  synchronized = false;

}

void CudaManager::CleanGpu() {

  for (MemoryMap::iterator i = memory_map.begin(); i != memory_map.end(); ++i) {
    FreeFromGpu(i->second);
  }
  memory_map.clear();

  synchronized = false;

}

/**
 * Allocates all objects retrieved by ScanGeometry() on the GPU, storing
 * pointers in the memory table for future reference.
 */
void CudaManager::AllocateGeometry() {

  for (std::set<LogicalVolume const*>::const_iterator i =
       logical_volumes.begin(); i != logical_volumes.end(); ++i) {

    LogicalVolume *const gpu_address = AllocateOnGpu<LogicalVolume>();
    memory_map[ToCpuAddress(*i)] = ToGpuAddress(gpu_address);

  }

  for (std::set<VUnplacedVolume const*>::const_iterator i =
       unplaced_volumes.begin(); i != unplaced_volumes.end(); ++i) {

    const GpuAddress gpu_address = AllocateOnGpu((*i)->byte_size());
    memory_map[ToCpuAddress(*i)] = gpu_address;

  }

  for (std::set<VPlacedVolume const*>::const_iterator i =
       placed_volumes.begin(); i != placed_volumes.end(); ++i) {

    const GpuAddress gpu_address = AllocateOnGpu((*i)->byte_size());
    memory_map[ToCpuAddress(*i)] = gpu_address;

  }

  for (std::set<TransformationMatrix const*>::const_iterator i =
       matrices.begin(); i != matrices.end(); ++i) {

    TransformationMatrix *const gpu_address =
        AllocateOnGpu<TransformationMatrix>();
    memory_map[ToCpuAddress(*i)] = ToGpuAddress(gpu_address);

  }

  for (std::set<Container<Daughter> *>::const_iterator i =
       daughters.begin(); i != daughters.end(); ++i) {

    Array<Daughter> *const gpu_address =
        AllocateOnGpu<Array<Daughter> >();
    memory_map[ToCpuAddress(*i)] = ToGpuAddress(gpu_address);

    // Also allocate the C-array necessary when creating the array object.
    // Will index with the GPU address of the Array object.
    const GpuAddress gpu_array = AllocateOnGpu(
      (static_cast<Vector<Daughter> const*>(*i))->size()
      * sizeof(Daughter)
    );
    memory_map[ToCpuAddress(gpu_address)] = ToGpuAddress(gpu_array);

  }

}

/**
 * Recursively scans logical volumes to retrieve all unique objects for copying
 * to the GPU.
 */
void CudaManager::ScanGeometry(LogicalVolume const &volume) {

  if (logical_volumes.find(&volume) != logical_volumes.end()) {
    logical_volumes.insert(&volume);
  }
  if (unplaced_volumes.find(&volume.unplaced_volume_)
      != unplaced_volumes.end()) {
    unplaced_volumes.insert(&volume.unplaced_volume_);
  }
  if (daughters.find(volume.daughters_) != daughters.end()) {
    daughters.insert(volume.daughters_);
  }
  for (Iterator<Daughter> i = volume.daughters().begin();
       i != volume.daughters().end(); ++i) {
    if (placed_volumes.find(*i) != placed_volumes.end()) {
      placed_volumes.insert(*i);
    }
    if (matrices.find(&((*i)->matrix_)) != matrices.end()) {
      matrices.insert(&(*i)->matrix_);
    }
    ScanGeometry((*i)->logical_volume());
  }

}

void CudaManager::PrintContent() const {
  std::cout << "-- Logical volumes with daughters:\n";
  for (std::set<LogicalVolume const*>::const_iterator i =
       logical_volumes.begin(); i != logical_volumes.end(); ++i) {
    std::cout << (**i);
  }
  std::cout << "-- Unplaced volumes:\n";
  for (std::set<VUnplacedVolume const*>::const_iterator i =
       unplaced_volumes.begin(); i != unplaced_volumes.end(); ++i) {
    std::cout << (**i) << std::endl;
  }
  std::cout << "-- Placed volumes:\n";
  for (std::set<VPlacedVolume const*>::const_iterator i =
       placed_volumes.begin(); i != placed_volumes.end(); ++i) {
    std::cout << (**i) << std::endl;
  }
  std::cout << "-- Transformation matrices:\n";
  for (std::set<TransformationMatrix const*>::const_iterator i =
       matrices.begin(); i != matrices.end(); ++i) {
    std::cout << (**i) << std::endl;
  }
}

} // End namespace vecgeom