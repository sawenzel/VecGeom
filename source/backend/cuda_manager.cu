#include <algorithm>
#include <cassert>
#include "backend/cuda_backend.cuh"
#include "base/array.h"
#include "management/cuda_manager.h"
#include "management/volume_factory.h"

namespace vecgeom {

LogicalVolume const* CudaManager::world() const {
  assert(world_ != NULL);
  return world_;
}

LogicalVolume const* CudaManager::world_gpu() const {
  assert(world_gpu_ != NULL);
  return world_gpu_;
}

LogicalVolume const* CudaManager::Synchronize() {

  if (verbose > 0) std::cerr << "Starting synchronization to GPU.\n";

  // Will return null if no geometry is loaded
  if (synchronized) return world_gpu_;

  CleanGpu();

  // Populate the memory map with GPU addresses

  AllocateGeometry();

  // Create new objects with pointers adjusted to point to GPU memory, then
  // copy them to the allocated memory locations on the GPU.

  if (verbose > 1) std::cerr << "Copying geometry to GPU...";

  if (verbose > 2) std::cerr << "Copying logical volumes...";
  for (std::set<LogicalVolume const*>::const_iterator i =
       logical_volumes.begin(); i != logical_volumes.end(); ++i) {

    LogicalVolume *gpu_object = new LogicalVolume(
      static_cast<VUnplacedVolume const*>(
        memory_map[ToCpuAddress((*i)->unplaced_volume())]
      ),
      static_cast<Container<VPlacedVolume const*> *>(
        memory_map[ToCpuAddress(&(*i)->daughters())]
      )
    );

    CopyToGpu(
      gpu_object, static_cast<LogicalVolume*>(memory_map[ToCpuAddress(*i)])
    );

    delete gpu_object;

  }
  if (verbose > 2) std::cerr << " OK\n";

  if (verbose > 2) std::cerr << "Copying unplaced volumes...";
  for (std::set<VUnplacedVolume const*>::const_iterator i =
       unplaced_volumes.begin(); i != unplaced_volumes.end(); ++i) {

    (*i)->CopyToGpu(
      static_cast<VUnplacedVolume*>(memory_map[ToCpuAddress(*i)])
    );

  }
  if (verbose > 2) std::cerr << " OK\n";

  if (verbose > 2) std::cerr << "Copying placed volumes...";
  for (std::set<VPlacedVolume const*>::const_iterator i =
       placed_volumes.begin(); i != placed_volumes.end(); ++i) {

    VPlacedVolume *gpu_object =
        VolumeFactory::Instance().CreateSpecializedVolume(
          (*i)->logical_volume(),
          (*i)->matrix()
        );
    gpu_object->set_logical_volume(
      static_cast<LogicalVolume const*>(
        memory_map[ToCpuAddress((*i)->logical_volume())]
      )
    );
    gpu_object->set_matrix(
      static_cast<TransformationMatrix const*>(
        memory_map[ToCpuAddress((*i)->matrix())]
      )
    );

    CopyToGpu(
      gpu_object, static_cast<VPlacedVolume*>(memory_map[ToCpuAddress(*i)])
    );

    delete gpu_object;

  }
  if (verbose > 2) std::cerr << " OK\n";

  if (verbose > 2) std::cerr << "Copying transformation matrices...";
  for (std::set<TransformationMatrix const*>::const_iterator i =
       matrices.begin(); i != matrices.end(); ++i) {

    CopyToGpu(
      *i, static_cast<TransformationMatrix*>(memory_map[ToCpuAddress(*i)])
    );

  }
  if (verbose > 2) std::cerr << " OK\n";

  if (verbose > 2) std::cerr << "Copying daughter lists...";
  for (std::set<Container<Daughter> *>::const_iterator i =
       daughters.begin(); i != daughters.end(); ++i) {

    // First copy the C-arrays
    const int n_daughters = (static_cast<Vector<Daughter> *>(*i))->size();
    Daughter *arr = new Daughter[n_daughters];
    int j = 0;
    for (Iterator<Daughter> k = (*i)->begin(); k != (*i)->end(); ++k) {
      arr[j] = *k;
      j++;
    }
    VPlacedVolume const **const arr_gpu = static_cast<VPlacedVolume const **>(
      memory_map[ToCpuAddress(memory_map[ToCpuAddress(*i)])]
    );
    CopyToGpu(arr, arr_gpu, n_daughters*sizeof(VPlacedVolume const*));
    delete arr;

    // Then the array containers
    Array<Daughter> *gpu_object =
        new Array<Daughter>(arr_gpu, n_daughters);
    CopyToGpu(
      gpu_object, static_cast<Array<Daughter> *>(memory_map[ToCpuAddress(*i)])
    );
    delete gpu_object;

  }
  if (verbose > 1) std::cerr << " OK\n";

  synchronized = true;

  world_gpu_ =
      static_cast<LogicalVolume const*>(memory_map[ToCpuAddress(world_)]);

  if (verbose > 0) std::cerr << "Geometry synchronized to GPU.\n";

  return world_gpu_;

}

void CudaManager::LoadGeometry(LogicalVolume const *const volume) {

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

  if (verbose > 1) std::cerr << "Cleaning GPU...";

  for (MemoryMap::iterator i = memory_map.begin(); i != memory_map.end(); ++i) {
    FreeFromGpu(i->second);
  }
  memory_map.clear();

  world_gpu_ = NULL;
  synchronized = false;

  if (verbose > 1) std::cerr << " OK\n";

}

void CudaManager::AllocateGeometry() {

  if (verbose > 1) std::cerr << "Allocating geometry on GPU...";

  if (verbose > 2) {
    size_t free_memory = 0, total_memory = 0;
    CudaAssertError(cudaMemGetInfo(&free_memory, &total_memory));
    std::cerr << "\nAvailable memory: " << free_memory << " / "
                                        << total_memory << std::endl;
  }

  if (verbose > 2) std::cerr << "Allocating logical volumes...";
  for (std::set<LogicalVolume const*>::const_iterator i =
       logical_volumes.begin(); i != logical_volumes.end(); ++i) {

    LogicalVolume *const gpu_address = AllocateOnGpu<LogicalVolume>();
    memory_map[ToCpuAddress(*i)] = ToGpuAddress(gpu_address);

  }
  if (verbose > 2) std::cerr << " OK\n";

  if (verbose > 2) std::cerr << "Allocating unplaced volumes...";
  for (std::set<VUnplacedVolume const*>::const_iterator i =
       unplaced_volumes.begin(); i != unplaced_volumes.end(); ++i) {

    const GpuAddress gpu_address =
        AllocateOnGpu<GpuAddress*>((*i)->byte_size());
    memory_map[ToCpuAddress(*i)] = gpu_address;

  }
  if (verbose > 2) std::cerr << " OK\n";

  if (verbose > 2) std::cerr << "Allocating placed volumes...";
  for (std::set<VPlacedVolume const*>::const_iterator i =
       placed_volumes.begin(); i != placed_volumes.end(); ++i) {

    const GpuAddress gpu_address =
        AllocateOnGpu<GpuAddress*>((*i)->byte_size());
    memory_map[ToCpuAddress(*i)] = gpu_address;

  }
  if (verbose > 2) std::cerr << " OK\n";

  if (verbose > 2) std::cerr << "Allocating transformation matrices...";
  for (std::set<TransformationMatrix const*>::const_iterator i =
       matrices.begin(); i != matrices.end(); ++i) {

    TransformationMatrix *const gpu_address =
        AllocateOnGpu<TransformationMatrix>();
    memory_map[ToCpuAddress(*i)] = ToGpuAddress(gpu_address);

  }
  if (verbose > 2) std::cerr << " OK\n";

  if (verbose > 2) std::cerr << "Allocating daughter lists...";
  for (std::set<Container<Daughter> *>::const_iterator i =
       daughters.begin(); i != daughters.end(); ++i) {

    Array<Daughter> *const gpu_address =
        AllocateOnGpu<Array<Daughter> >();
    memory_map[ToCpuAddress(*i)] = ToGpuAddress(gpu_address);

    // Also allocate the C-array necessary when creating the array object.
    // Will index with the GPU address of the Array object.
    const GpuAddress gpu_array = AllocateOnGpu<GpuAddress*>(
      (static_cast<Vector<Daughter> const*>(*i))->size()
      * sizeof(Daughter)
    );
    memory_map[ToCpuAddress(gpu_address)] = ToGpuAddress(gpu_array);

  }
  if (verbose > 1) std::cerr << " OK\n";

}

void CudaManager::ScanGeometry(LogicalVolume const *const volume) {

  if (logical_volumes.find(volume) == logical_volumes.end()) {
    logical_volumes.insert(volume);
  }
  if (unplaced_volumes.find(volume->unplaced_volume_)
      == unplaced_volumes.end()) {
    unplaced_volumes.insert(volume->unplaced_volume_);
  }
  if (daughters.find(volume->daughters_) == daughters.end()) {
    daughters.insert(volume->daughters_);
  }
  for (Iterator<Daughter> i = volume->daughters().begin();
       i != volume->daughters().end(); ++i) {
    if (placed_volumes.find(*i) == placed_volumes.end()) {
      placed_volumes.insert(*i);
    }
    if (matrices.find((*i)->matrix_) == matrices.end()) {
      matrices.insert((*i)->matrix_);
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