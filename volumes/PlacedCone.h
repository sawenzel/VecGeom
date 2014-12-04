/*
 * PlacedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */
//#define VECGEOM_ROOT
//#define VECGEOM_GEANT4
//#define VECGEOM_BENCHMARK
//#define VECGEOM_USOLIDS


#ifndef VECGEOM_VOLUMES_PLACEDCONE_H_
#define VECGEOM_VOLUMES_PLACEDCONE_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedCone.h"
#include "volumes/kernel/ConeImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedCone; )
VECGEOM_DEVICE_DECLARE_CONV( PlacedCone );

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedCone : public VPlacedVolume {

public:

  typedef UnplacedCone UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedCone(char const *const label,
                       LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedCone(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : PlacedCone("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedCone(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedCone() {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedCone const* GetUnplacedVolume() const {
    return static_cast<UnplacedCone const *>(
        logical_volume()->unplaced_volume());
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
#endif // VECGEOM_BENCHMARK

  Precision GetRmin1() const {return GetUnplacedVolume()->GetRmin1();}
  Precision GetRmax1() const {return GetUnplacedVolume()->GetRmax1();}
  Precision GetRmin2() const {return GetUnplacedVolume()->GetRmin2();}
  Precision GetRmax2() const {return GetUnplacedVolume()->GetRmax2();}
  Precision GetDz() const {return GetUnplacedVolume()->GetDz();}
  Precision GetSPhi() const {return GetUnplacedVolume()->GetSPhi();}
  Precision GetDPhi() const {return GetUnplacedVolume()->GetDPhi();}
  Precision GetInnerSlope() const {return GetUnplacedVolume()->GetInnerSlope();}
  Precision GetOuterSlope() const {return GetUnplacedVolume()->GetOuterSlope();}
  Precision GetInnerOffset() const {return GetUnplacedVolume()->GetInnerOffset();}
  Precision GetOuterOffset() const {return GetUnplacedVolume()->GetOuterOffset();}

  Precision Capacity() const { return GetUnplacedVolume()->Capacity();}

#ifdef VECGEOM_USOLIDS
  virtual
  void Extent(Vector3D<Precision> & aMin, Vector3D<Precision> & aMax) const
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

  virtual
  bool Normal(Vector3D<Precision> const & point, Vector3D<Precision> & normal ) const
  {
      bool valid;
      ConeImplementation<translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::NormalKernel<kScalar>(
              *GetUnplacedVolume(),
              point,
              normal, valid);
      return valid;
  }

  virtual
  Vector3D<Precision> GetPointOnSurface() const
  {
    return GetUnplacedVolume()->GetPointOnSurface();
  }

  virtual Precision Capacity() {
    return GetUnplacedVolume()->Capacity();
  }

  virtual double SurfaceArea() {
     return GetUnplacedVolume()->SurfaceArea();
  }

  virtual std::string GetEntityType() const {
      return "Cone";
  }
#endif


}; // end class

} } // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDCONE_H_
