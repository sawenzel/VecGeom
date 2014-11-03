/// \file PlacedTube.h
/// \author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDTUBE_H_
#define VECGEOM_VOLUMES_PLACEDTUBE_H_

#include "base/Global.h"
#include "backend/Backend.h"
 
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/TubeImplementation.h"

namespace VECGEOM_NAMESPACE {

class PlacedTube : public VPlacedVolume {

public:

  typedef UnplacedTube UnplacedShape_t;


#ifndef VECGEOM_NVCC

  PlacedTube(char const *const label,
                       LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedTube(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : PlacedTube("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedTube(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedTube() {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTube const* GetUnplacedVolume() const {
    return static_cast<UnplacedTube const *>(
        logical_volume()->unplaced_volume());
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmin() const { return GetUnplacedVolume()->rmin(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmax() const { return GetUnplacedVolume()->rmax(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision z() const { return GetUnplacedVolume()->z(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision sphi() const { return GetUnplacedVolume()->sphi(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dphi() const { return GetUnplacedVolume()->dphi(); }

#ifndef VECGEOM_NVCC
  virtual VPlacedVolume const* ConvertToUnspecialized() const;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const;
#endif
#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const* ConvertToGeant4() const;
#endif
#endif // VECGEOM_NVCC

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

#endif // VECGEOM_VOLUMES_PLACEDTUBE_H_










