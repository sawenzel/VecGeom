/// @file ShapeImplementationHelper.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_PLACEDPARALLELEPIPED_H_

#include "base/global.h"
#include "volumes/placed_volume.h"
#include "volumes/UnplacedParallelepiped.h"

namespace VECGEOM_NAMESPACE {

class PlacedParallelepiped : public VPlacedVolume {

public:

  typedef UnplacedParallelepiped UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedParallelepiped(char const *const label,
                       LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedParallelepiped(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : PlacedParallelepiped("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedParallelepiped(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual ~PlacedParallelepiped() {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedParallelepiped const* GetUnplacedVolume() const {
    return static_cast<UnplacedParallelepiped const *>(
        logical_volume()->unplaced_volume());
  }

  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> const& GetDimensions() const {
    return GetUnplacedVolume()->GetDimensions();
  }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetX() const { return GetUnplacedVolume()->GetX(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetY() const { return GetUnplacedVolume()->GetY(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetZ() const { return GetUnplacedVolume()->GetZ(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlpha() const { return GetUnplacedVolume()->GetAlpha(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTheta() const { return GetUnplacedVolume()->GetTheta(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhi() const { return GetUnplacedVolume()->GetPhi(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanAlpha() const { return GetUnplacedVolume()->GetTanAlpha(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanThetaSinPhi() const {
    return GetUnplacedVolume()->GetTanThetaSinPhi();
  }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanThetaCosPhi() const {
    return GetUnplacedVolume()->GetTanThetaCosPhi();
  }

#ifdef VECGEOM_BENCHMARK
  virtual VPlacedVolume const* ConvertToUnspecialized() const;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const;
#endif
#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const;
#endif
#endif // VECGEOM_BENCHMARK

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation,
                                   VPlacedVolume *const gpu_ptr) const;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transformation) const;
#endif

};

} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDPARALLELEPIPED_H_