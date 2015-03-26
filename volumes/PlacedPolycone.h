/*
 * PlacedPolycone.h
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_PLACEDPOLYCONE_H_
#define VECGEOM_VOLUMES_PLACEDPOLYCONE_H_


#include "base/Global.h"
#include "backend/Backend.h"

#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"

#include "volumes/UnplacedPolycone.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedPolycone; )
VECGEOM_DEVICE_DECLARE_CONV( PlacedPolycone );

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedPolycone : public VPlacedVolume {

public:
  typedef UnplacedPolycone UnplacedShape_t;

#ifndef VECGEOM_NVCC
  PlacedPolycone(char const *const label,
          LogicalVolume const *const logical_volume,
          Transformation3D const *const transformation,
          PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedPolycone(LogicalVolume const *const logical_volume,
          Transformation3D const *const transformation,
          PlacedBox const *const boundingBox)
      : PlacedPolycone("", logical_volume, transformation, boundingBox) {}

#else
  __device__
  PlacedPolycone(LogicalVolume const *const logical_volume,
          Transformation3D const *const transformation,
          PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}
#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedPolycone() {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedPolycone const* GetUnplacedVolume() const {
    return static_cast<UnplacedPolycone const *>(
        GetLogicalVolume()->unplaced_volume());
  }


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

  virtual Precision Capacity() override {
     return GetUnplacedVolume()->Capacity();
   }

  virtual
  void Extent(Vector3D<Precision> & aMin, Vector3D<Precision> & aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

  std::string GetEntityType() const { return GetUnplacedVolume()->GetEntityType() ;}

  //virtual
  //bool Normal(Vector3D<Precision> const & point, Vector3D<Precision> & normal ) const
  //{
      //bool valid;
      //BoxImplementation<translation::kIdentity, rotation::kIdentity>::NormalKernel<kScalar>(
              //*GetUnplacedVolume(),
              //point,
              //normal, valid);
      //return valid;
  //}

  virtual
  Vector3D<Precision> GetPointOnSurface() const {
    return GetUnplacedVolume()->GetPointOnSurface();
  }

  virtual double SurfaceArea() override {
     return GetUnplacedVolume()->SurfaceArea();
  }
#endif

}; // end of class

} // end inline namespace

} // end global namespace

#endif /* VECGEOM_VOLUMES_PLACEDPOLYCONE_H_ */
