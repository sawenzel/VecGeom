/// \file PlacedOrb.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDORB_H_
#define VECGEOM_VOLUMES_PLACEDORB_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedOrb.h"

namespace VECGEOM_NAMESPACE {

class PlacedOrb : public VPlacedVolume {

public:

  typedef UnplacedOrb UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedOrb(char const *const label,
                       LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedOrb(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : PlacedOrb("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedOrb(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual ~PlacedOrb() {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb const* GetUnplacedVolume() const {
    return static_cast<UnplacedOrb const *>(
        logical_volume()->unplaced_volume());
  }
  

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetRadius() const { return GetUnplacedVolume()->GetRadius(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision>  dimensions() const { return GetUnplacedVolume()->dimensions(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision x() const { return GetUnplacedVolume()->x(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision y() const { return GetUnplacedVolume()->y(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision z() const { return GetUnplacedVolume()->z(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetRadialTolerance() const { return GetUnplacedVolume()->GetRadialTolerance(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolO() const { return GetUnplacedVolume()->GetfRTolO(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolI() const { return GetUnplacedVolume()->GetfRTolI(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetVolume() const { return GetUnplacedVolume()->GetVolume(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetSurfaceArea() const { return GetUnplacedVolume()->GetSurfaceArea(); }




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

#endif // VECGEOM_VOLUMES_PLACEDORB_H_
