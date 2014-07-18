/// \file PlacedPolyhedron.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDPOLYHEDRON_H_
#define VECGEOM_VOLUMES_PLACEDPOLYHEDRON_H_

#include "base/Global.h"
#include "backend/Backend.h"
 
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedPolyhedron.h"

namespace VECGEOM_NAMESPACE {

class PlacedPolyhedron : public VPlacedVolume {

public:

#ifndef VECGEOM_NVCC

  PlacedPolyhedron(char const *const label,
            LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logicalVolume, transformation, boundingBox) {}

  PlacedPolyhedron(LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox)
      : PlacedPolyhedron("", logicalVolume, transformation, boundingBox) {}

#else

  __device__
  PlacedPolyhedron(LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox,
            const int id)
      : VPlacedVolume(logicalVolume, transformation, boundingBox, id) {}

#endif

  virtual ~PlacedPolyhedron() {}

  // Accessors

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedPolyhedron const* GetUnplacedVolume() const {
    return static_cast<UnplacedPolyhedron const *>(
        logical_volume()->unplaced_volume());
  }

  // CUDA specific

  virtual int memory_size() const { return sizeof(*this); }

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transformation,
      VPlacedVolume *const gpu_ptr) const;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transformation) const;
#endif

  // Comparison specific

#ifdef VECGEOM_BENCHMARK
  virtual VPlacedVolume const* ConvertToUnspecialized() const {
    Assert(0, "NYI");
  }
#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const {
    Assert(0, "NYI");
  }
#endif
#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const {
    Assert(0, "NYI");
  }
#endif
#endif // VECGEOM_BENCHMARK

};

} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDPOLYHEDRON_H_