/// \file VolumePointers.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BENCHMARKING_VOLUMEPOINTERS_H_
#define VECGEOM_BENCHMARKING_VOLUMEPOINTERS_H_

#include "base/Global.h"

#include "benchmarking/BenchmarkResult.h"

class TGeoShape;
class VUSolid;
class G4VSolid;

namespace vecgeom {

class VPlacedVolume;

/// \brief Converts a VecGeom volume to unspecialized, USolids and ROOT
///        representations for performance comparison purposes.
class VolumePointers {

private:

  VPlacedVolume const *fSpecialized;
  VPlacedVolume const *fUnspecialized;
#ifdef VECGEOM_ROOT
  TGeoShape const *fRoot;
#endif
#ifdef VECGEOM_USOLIDS
  ::VUSolid const *fUSolids;
#endif
#ifdef VECGEOM_GEANT4
  G4VSolid const *fGeant4;
#endif
  /** Remember which objects can be safely deleted. */
  EBenchmarkedLibrary fInitial;

public:

  VolumePointers(VPlacedVolume const *const volume);

  /**
   * Deep copies from other object to avoid ownership issues.
   */
  VolumePointers(VolumePointers const &other);

  ~VolumePointers();

  VolumePointers& operator=(VolumePointers const &other);

  VPlacedVolume const* Specialized() const { return fSpecialized; }

  VPlacedVolume const* Unspecialized() const { return fUnspecialized; }

#ifdef VECGEOM_ROOT
  TGeoShape const* ROOT() const { return fRoot; }
#endif

#ifdef VECGEOM_USOLIDS
  ::VUSolid const* USolids() const { return fUSolids; }
#endif

#ifdef VECGEOM_GEANT4
  G4VSolid const* Geant4() const { return fGeant4; }
#endif

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