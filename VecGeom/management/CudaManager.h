/// \file CudaManager.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch); Sandro Wenzel(sandro.wenzel@cern.ch)

#ifndef VECGEOM_MANAGEMENT_CUDAMANAGER_H_
#define VECGEOM_MANAGEMENT_CUDAMANAGER_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/management/DeviceGlobals.h"

#include "VecGeom/base/Vector.h"
#include "VecGeom/volumes/Box.h"

#ifdef VECGEOM_CUDA_INTERFACE
#include "VecGeom/backend/cuda/Interface.h"
#endif

#include <list>
#include <map>
#include <set>

// Compile for vecgeom namespace to work as interface
namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class VPlacedVolume;);
VECGEOM_DEVICE_FORWARD_DECLARE(void CudaManagerPrintGeometry(vecgeom::cuda::VPlacedVolume const *const world););
VECGEOM_DEVICE_FORWARD_DECLARE(void InitDeviceCompactPlacedVolBufferPtr(void *););
VECGEOM_DEVICE_FORWARD_DECLARE(void InitDeviceLogicalVolumesPtr(void *););

// we put some global data into a separate namespace
// this is done since CUDA does not support static const members in class definitions
namespace globaldevicegeomdata {
inline VECCORE_ATT_DEVICE VPlacedVolume *gCompactPlacedVolBuffer = nullptr;
inline VECCORE_ATT_DEVICE LogicalVolume *gDeviceLogicalVolumes   = nullptr;
inline VECCORE_ATT_DEVICE NavIndex_t *gNavIndex                  = nullptr; // address of navigation index table
inline VECCORE_ATT_DEVICE int gMaxDepth                          = 0;
} // namespace globaldevicegeomdata

#ifndef VECCORE_CUDA
inline
#endif
    namespace cxx {

#ifdef VECCORE_CUDA
// Forward declarations for NVCC compilation
class VUnplacedVolume;
class VPlacedVolume;
class LogicalVolume;
class Transformation3D;
template <typename Type>
class Vector;
#endif

class CudaManager {

private:
  bool synchronized_;
  int verbose_;
  int total_volumes_;

  using Daughter_t        = VPlacedVolume const *;
  using CudaDaughter_t    = cuda::VPlacedVolume const *;
  using CudaDaughterPtr_t = DevicePtr<cuda::VPlacedVolume>;

  std::set<VUnplacedVolume const *> unplaced_volumes_;
  std::set<LogicalVolume const *> logical_volumes_;
  std::set<VPlacedVolume const *> placed_volumes_;
  std::set<Transformation3D const *> transformations_;
  std::set<Vector<Daughter_t> *> daughters_;

  typedef void const *CpuAddress;
  typedef DevicePtr<char> GpuAddress;
  typedef std::map<const CpuAddress, GpuAddress> MemoryMap;
  typedef std::map<GpuAddress, CpuAddress> PlacedVolumeMemoryMap;
  typedef std::map<GpuAddress, GpuAddress> GpuMemoryMap;

  VPlacedVolume const *world_;
  DevicePtr<vecgeom::cuda::VPlacedVolume> world_gpu_;
  DevicePtr<vecgeom::cuda::VPlacedVolume> fPlacedVolumeBufferOnDevice;
  DevicePtr<NavIndex_t> fNavTableOnDevice;

private:
  /**
   * Contains a mapping between objects stored in host memory and pointers to
   * equivalent objects stored on the GPU. Stored GPU pointers are pointing to
   * allocated memory, but do not necessary have meaningful data stored at the
   * addresses yet.
   * \sa AllocateGeometry()
   * \sa CleanGpu()
   */
  MemoryMap memory_map_;
  GpuMemoryMap gpu_memory_map_;
  /**
   * inverse memory_map for fast GPU pointer to CPU conversion
   *
   */
  PlacedVolumeMemoryMap fGPUtoCPUmapForPlacedVolumes_;

  std::list<GpuAddress> allocated_memory_;

public:
  /**
   * Retrieve singleton instance.
   */
  static CudaManager &Instance()
  {
    static CudaManager instance;
    return instance;
  }

  VPlacedVolume const *world() const;

  vecgeom::cuda::VPlacedVolume const *world_gpu() const;

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
  DevicePtr<const vecgeom::cuda::VPlacedVolume> Synchronize();

  /**
   * @brief Synchronizes the navigation table on the coprocessor
   */
  bool SynchronizeNavigationTable() { return AllocateNavIndexOnCoproc(); };

  /**
   * Deallocates all GPU pointers stored in the memory table.
   */
  void CleanGpu();

  /**
   * Forget the geometry (to prepare for a new call to LoadGeomtry)
   */
  void Clear();

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

  void set_verbose(const int verbose) { verbose_ = verbose; }

  template <typename Type>
  GpuAddress Lookup(Type const *const key) const;

  template <typename Type>
  GpuAddress Lookup(DevicePtr<Type> key) const;

  DevicePtr<cuda::VUnplacedVolume> LookupUnplaced(VUnplacedVolume const *const host_ptr) const;

  DevicePtr<cuda::LogicalVolume> LookupLogical(LogicalVolume const *const host_ptr) const;

  DevicePtr<cuda::VPlacedVolume> LookupPlaced(VPlacedVolume const *const host_ptr) const;
  VPlacedVolume const *LookupPlacedCPUPtr(const void *address);

  DevicePtr<cuda::Transformation3D> LookupTransformation(Transformation3D const *const host_ptr) const;

  DevicePtr<cuda::Vector<CudaDaughter_t>> LookupDaughters(Vector<Daughter_t> *const host_ptr) const;

  DevicePtr<CudaDaughter_t> LookupDaughterArray(Vector<Daughter_t> *const host_ptr) const;

private:
  CudaManager();
  CudaManager(CudaManager const &);
  CudaManager &operator=(CudaManager const &);

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
  static CpuAddress ToCpuAddress(Type const *const ptr)
  {
    return static_cast<CpuAddress>(ptr);
  }

  /**
   * Helper routine allocate GPU memory for a collection of object
   */
  template <typename Coll>
  bool AllocateCollectionOnCoproc(const char *verbose_title, const Coll &data, bool isplaced = false);

  /**
   * Helper routine allocate GPU memory for placed volume objects
   */
  bool AllocatePlacedVolumesOnCoproc();

  /** Allocator method for the navigation index table */
  bool AllocateNavIndexOnCoproc();

  // template <typename TrackContainer>
  // void LocatePointsTemplate(TrackContainer const &container, const int n,
  //                           const int depth, int *const output) const;

  /// Copy all placed volumes to the device.
  void CopyPlacedVolumes() const;
};

// void CudaManagerLocatePoints(VPlacedVolume const *const world,
//                              SOA3D<Precision> const *const points,
//                              const int n, const int depth, int *const output);

inline VPlacedVolume const *CudaManager::LookupPlacedCPUPtr(const void *address)
{
  const VPlacedVolume *cpu_ptr =
      (const VPlacedVolume *)fGPUtoCPUmapForPlacedVolumes_[GpuAddress(const_cast<void *>(address))];
  VECGEOM_ASSERT(cpu_ptr != NULL);
  return cpu_ptr;
}
} // namespace cxx
} // namespace vecgeom

#endif // VECGEOM_MANAGEMENT_CUDAMANAGER_H_
