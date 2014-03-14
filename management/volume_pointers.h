/**
 * \author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BENCHMARKING_VOLUMEPOINTERS_H_
#define VECGEOM_BENCHMARKING_VOLUMEPOINTERS_H_

#include "base/global.h"

class TGeoNode;
class VUSolid;

namespace vecgeom {

enum BenchmarkType {kSpecialized, kUnspecialized, kUSolids, kRoot};

class VolumePointers {

private:

  VPlacedVolume const *specialized_;
  VPlacedVolume const *unspecialized_;
  TGeoShape const *root_;
  ::VUSolid const *usolids_;
  BenchmarkType initial_;

public:

  VolumePointers(VPlacedVolume const *const volume);

  VolumePointers(VolumePointers const &other);

  ~VolumePointers();

  VolumePointers& operator=(VolumePointers const &other);

  VPlacedVolume const* specialized() const { return specialized_; }

  VPlacedVolume const* unspecialized() const { return unspecialized_; }

  TGeoShape const* root() const { return root_; }

  ::VUSolid const* usolids() const { return usolids_; }

private:

  void ConvertVolume();
  
  void Deallocate();

};

} // End namespace vecgeom

#endif // VECGEOM_BENCHMARKING_VOLUMEPOINTERS_H_