/**
 * @file cuda_manager.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_MANAGEMENT_CUDAMANAGER_H_
#define VECGEOM_MANAGEMENT_CUDAMANAGER_H_

#include <map>
#include <set>
 
#include "base/global.h"
#include "base/vector.h"
#include "volumes/box.h"

namespace vecgeom_cuda { class VPlacedVolume; }

// Compile for vecgeom namespace to work as interface
namespace vecgeom {

#ifdef VECGEOM_NVCC
// Forward declarations for NVCC compilation
class VUnplacedVolume;
class VPlacedVolume;
typedef VPlacedVolume const* Daughter;
#endif

class CudaManager {

private:

  bool synchronized;
  int verbose_;
  int total_volumes;

  std::set<VUnplacedVolume const*> unplaced_volumes_;
  std::set<LogicalVolume const*> logical_volumes_;
  std::set<VPlacedVolume const*> placed_volumes_;
  std::set<Transformation3D const*> transformations_;
  std::set<Vector<Daughter> *> daughters_;

  typedef void const* CpuAddress;
  typedef void* GpuAddress;
  typedef std::map<const CpuAddress, GpuAddress> MemoryMap;

  VPlacedVolume const *world_;
  vecgeom_cuda::VPlacedVolume *world_gpu_;

  /**
   * Contains a mapping between objects stored in host memory and pointers to
   * equivalent objects stored on the GPU. Stored GPU pointers are pointing to
   * allocated memory, but do not necessary have meaningful data stored at the
   * addresses yet.
   * \sa AllocateGeometry()
   * \sa CleanGpu()
   */
  MemoryMap memory_map;

public:

  /**
   * Retrieve singleton instance.
   */
  static CudaManager& Instance() {
    static CudaManager instance;
    return instance;
  }

  VPlacedVolume const* world() const;

  vecgeom_cuda::VPlacedVolume const* world_gpu() const;

  /**
   * Stages a new geometry to be copied to the GPU.
   */
  void LoadGeometry(VPlacedVolume const *const volume);

  void LoadGeometry();

  /**
   * Synchronizes the loaded geometry to the GPU by allocating space,
   * creating new objects with correct pointers, then copying them to the GPU.
   * \return Pointer to top volume on the GPU.
   */
  vecgeom_cuda::VPlacedVolume const* Synchronize();

  /**
   * Deallocates all GPU pointers stored in the memory table.
   */
  void CleanGpu();

  /**
   * Launch a CUDA kernel that recursively outputs the geometry loaded onto the
   * device.
   */
  void PrintGeometry() const;

  // /**
  //  * Launch a CUDA kernel that will locate points in the geometry
  //  */
  // void LocatePoints(SOA3D<Precision> const &container, const int depth,
  //                   int *const output) const;

  // /**
  //  * Launch a CUDA kernel that will locate points in the geometry
  //  */
  // void LocatePoints(AOS3D<Precision> const &container, const int depth,
  //                   int *const output) const;

  void set_verbose(const int verbose) { verbose_ = verbose; }

  template <typename Type>
  GpuAddress Lookup(Type const *const key);

  VUnplacedVolume* LookupUnplaced(
      VUnplacedVolume const *const host_ptr);

  LogicalVolume* LookupLogical(LogicalVolume const *const host_ptr);

  VPlacedVolume* LookupPlaced(VPlacedVolume const *const host_ptr);

  Transformation3D* LookupTransformation(
      Transformation3D const *const host_ptr);

  Vector<Daughter>* LookupDaughters(Vector<Daughter> *const host_ptr);

  Daughter* LookupDaughterArray(
      Vector<Daughter> *const host_ptr);

private:

  CudaManager();
  CudaManager(CudaManager const&);
  CudaManager& operator=(CudaManager const&);

  /**
   * Recursively scans placed volumes to retrieve all unique objects
   * for copying to the GPU.
   */
  void ScanGeometry(VPlacedVolume const *const volume);

  /**
   * Allocates all objects retrieved by ScanGeometry() on the GPU, storing
   * pointers in the memory table for future reference.
   */
  void AllocateGeometry();

  /**
   * Converts object pointers to void pointers so they can be used as lookup in
   * the memory table.
   */
  template <typename Type>
  CpuAddress ToCpuAddress(Type const *const ptr) const {
    return static_cast<CpuAddress>(ptr);
  }

  /**
   * Converts object pointers to void pointers so they can be stored in
   * the memory table.
   */
  template <typename Type>
  GpuAddress ToGpuAddress(Type *const ptr) const {
    return static_cast<GpuAddress>(ptr);
  }

  // template <typename TrackContainer>
  // void LocatePointsTemplate(TrackContainer const &container, const int n,
  //                           const int depth, int *const output) const;

};

void CudaManagerPrintGeometry(vecgeom_cuda::VPlacedVolume const *const world);

// void CudaManagerLocatePoints(VPlacedVolume const *const world,
//                              SOA3D<Precision> const *const points,
//                              const int n, const int depth, int *const output);

// void CudaManagerLocatePoints(VPlacedVolume const *const world,
//                              AOS3D<Precision> const *const points,
//                              const int n, const int depth, int *const output);

} // End namespace vecgeom

#endif // VECGEOM_MANAGEMENT_CUDAMANAGER_H_