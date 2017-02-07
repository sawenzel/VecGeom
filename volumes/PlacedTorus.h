/// \file PlacedTorus.h

#ifndef VECGEOM_VOLUMES_PLACEDTORUS_H_
#define VECGEOM_VOLUMES_PLACEDTORUS_H_

#include "base/Global.h"
#include "backend/Backend.h"
 
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/TorusImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedTorus; )
VECGEOM_DEVICE_DECLARE_CONV( PlacedTorus );

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTorus : public VPlacedVolume {

public:

  typedef UnplacedTorus UnplacedShape_t;


#ifndef VECGEOM_NVCC

  PlacedTorus(char const *const label,
          LogicalVolume const *const logical_volume,
          Transformation3D const *const transformation,
          PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedTorus(LogicalVolume const *const logical_volume,
          Transformation3D const *const transformation,
          PlacedBox const *const boundingBox)
      : PlacedTorus("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedTorus(LogicalVolume const *const logical_volume,
          Transformation3D const *const transformation,
          PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedTorus() {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTorus const* GetUnplacedVolume() const {
    return static_cast<UnplacedTorus const *>(
        GetLogicalVolume()->GetUnplacedVolume());
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmin() const { return GetUnplacedVolume()->rmin(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmax() const { return GetUnplacedVolume()->rmax(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rtor() const { return GetUnplacedVolume()->rtor(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision sphi() const { return GetUnplacedVolume()->sphi(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dphi() const { return GetUnplacedVolume()->dphi(); }

  virtual Precision Capacity() override { return GetUnplacedVolume()->volume(); }

  void Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const override {
      GetUnplacedVolume()->Extent(aMin, aMax);
    }

#ifndef VECGEOM_NVCC
  virtual
   Vector3D<Precision> GetPointOnSurface() const override {
     return GetUnplacedVolume()->GetPointOnSurface();
   }
 bool Normal(Vector3D<Precision>const& point, Vector3D<Precision>& normal) const override {
	  return GetUnplacedVolume()->Normal(point, normal);
  }
  virtual VPlacedVolume const* ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const override;
#endif
#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const override;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const* ConvertToGeant4() const override;
#endif
#endif // VECGEOM_NVCC


}; 

} } // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDTORUS_H_










