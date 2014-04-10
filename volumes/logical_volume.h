/**
 * @file logical_volume.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_VOLUMES_LOGICALVOLUME_H_
#define VECGEOM_VOLUMES_LOGICALVOLUME_H_

#include <string>
#include "base/global.h"
#include "base/vector.h"
#include "volumes/unplaced_volume.h"

namespace VECGEOM_NAMESPACE {

typedef VPlacedVolume const* Daughter;

/**
 * @brief Class responsible for storing the unplaced volume, material and
 *        daughter volumes of a mother volume.
 */
class LogicalVolume {

private:

  VUnplacedVolume const *unplaced_volume_;

  /** a pointer member to register arbitrary objects with logical volume;
        included for the moment to model UserExtension like in TGeoVolume
  */
  void * user_extension_;

  Vector<Daughter> *daughters_;


  friend class CudaManager;

public:

  /**
   * Standard constructor when constructing geometries. Will initiate an empty
   * daughter list which can be populated by placing daughters.
   * \sa PlaceDaughter()
   */
  LogicalVolume(VUnplacedVolume const *const unplaced_volume)
      : unplaced_volume_(unplaced_volume), user_extension_(0) {
    daughters_ = new Vector<Daughter>();
  }

  VECGEOM_CUDA_HEADER_DEVICE
  LogicalVolume(VUnplacedVolume const *const unplaced_volume,
                Vector<Daughter> *daughters)
      : unplaced_volume_(unplaced_volume), user_extension_(0), daughters_(daughters) {}

  ~LogicalVolume();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VUnplacedVolume const* unplaced_volume() const { return unplaced_volume_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector<Daughter> const& daughters() const { return *daughters_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector<Daughter> const * daughtersp() const { return daughters_; }

  VECGEOM_INLINE
  void setUserExtensionPtr( void * userpointer ) { user_extension_ = userpointer; }

  VECGEOM_INLINE
  void * getUserExtensionPtr( ) const {  return user_extension_;  }

  #ifndef VECGEOM_NVCC

  VPlacedVolume* Place(TransformationMatrix const *const matrix) const;

  VPlacedVolume* Place() const;

  void PlaceDaughter(LogicalVolume const *const volume,
                     TransformationMatrix const *const matrix);

  void PlaceDaughter(VPlacedVolume const *const placed);

  #endif

  VECGEOM_CUDA_HEADER_BOTH
  int CountVolumes() const;

  friend std::ostream& operator<<(std::ostream& os, LogicalVolume const &vol);

  #ifdef VECGEOM_CUDA_INTERFACE
  LogicalVolume* CopyToGpu(VUnplacedVolume const *const unplaced_volume,
                           Vector<Daughter> *daughters) const;
  LogicalVolume* CopyToGpu(VUnplacedVolume const *const unplaced_volume,
                           Vector<Daughter> *daughters,
                           LogicalVolume *const gpu_ptr) const;
  #endif

};

} // End global namespace

#endif // VECGEOM_VOLUMES_LOGICALVOLUME_H_
