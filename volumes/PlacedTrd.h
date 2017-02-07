/// @file PlacedTrd.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDTRD_H_
#define VECGEOM_VOLUMES_PLACEDTRD_H_

#include "base/Global.h"
#include "backend/Backend.h"
#ifndef VECGEOM_NVCC
	#include "base/RNG.h"
	#include <cassert>
	#include <cmath>
#endif
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/TrdImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedTrd; )
VECGEOM_DEVICE_DECLARE_CONV( PlacedTrd );

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTrd : public VPlacedVolume {

public:
  typedef UnplacedTrd UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedTrd(char const *const label,
                       LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedTrd(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : PlacedTrd("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedTrd(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedTrd() {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrd const* GetUnplacedVolume() const {
    return static_cast<UnplacedTrd const *>(
        GetLogicalVolume()->GetUnplacedVolume());
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dx1() const { return GetUnplacedVolume()->dx1(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dx2() const { return GetUnplacedVolume()->dx2(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dy1() const { return GetUnplacedVolume()->dy1(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dy2() const { return GetUnplacedVolume()->dy2(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dz() const { return GetUnplacedVolume()->dz(); }

  void Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const override {
      GetUnplacedVolume()->Extent(aMin, aMax);
    }

#ifndef VECGEOM_NVCC
  virtual
  Precision Capacity() override { return GetUnplacedVolume()->Capacity(); }

  virtual
  Precision SurfaceArea() override { return GetUnplacedVolume()->SurfaceArea();}

  virtual Vector3D<Precision> GetPointOnSurface() const override {
     return GetUnplacedVolume()->GetPointOnSurface();
  }

  bool Normal(Vector3D<Precision>const& point, Vector3D<Precision>& normal) const override {
     return GetUnplacedVolume()->Normal(point, normal);
  }

  /*
  void Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const override {
      GetUnplacedVolume()->Extent(aMin, aMax);
  }
  */

  virtual VPlacedVolume const* ConvertToUnspecialized() const override;

#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const override;
#endif

#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const override;
#endif

#ifdef VECGEOM_GEANT4
  G4VSolid const* ConvertToGeant4() const override;
#endif
#endif // VECGEOM_NVCC

};

} } // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDTUBE_H_
