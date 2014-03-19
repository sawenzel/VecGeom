/**
 * @file volume_pointers.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BENCHMARKING_VOLUMEPOINTERS_H_
#define VECGEOM_BENCHMARKING_VOLUMEPOINTERS_H_

#include "base/global.h"

class TGeoNode;
class VUSolid;

namespace VECGEOM_NAMESPACE {

enum BenchmarkType {kSpecialized, kUnspecialized, kUSolids, kRoot};

/**
 * @brief Converts a VecGeom volume to unspecialized, USolids and ROOT
 *        representations for performance comparison purposes.
 */
class VolumePointers {

private:

  VPlacedVolume const *specialized_ = NULL;
  VPlacedVolume const *unspecialized_ = NULL;
  TGeoShape const *root_ = NULL;
  ::VUSolid const *usolids_ = NULL;
  /** Remember which objects can be safely deleted. */
  BenchmarkType initial_;

public:

  VolumePointers(VPlacedVolume const *const volume);

  /**
   * Deep copies from other object to avoid ownership issues.
   */
  VolumePointers(VolumePointers const &other);

  ~VolumePointers();

  VolumePointers& operator=(VolumePointers const &other);

  VPlacedVolume const* specialized() const { return specialized_; }

  VPlacedVolume const* unspecialized() const { return unspecialized_; }

  TGeoShape const* root() const { return root_; }

  ::VUSolid const* usolids() const { return usolids_; }

private:

  /**
   * Converts the currently stored specialized volume to each other
   * representation not yet instantiated.
   */
  void ConvertVolume();
  
  void Deallocate();

};

} // End global namespace

#endif // VECGEOM_BENCHMARKING_VOLUMEPOINTERS_H_