/// \file PlacedOrb.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDORB_H_
#define VECGEOM_VOLUMES_PLACEDORB_H_

#include "base/Global.h"
#include "backend/Backend.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/UnplacedOrb.h"
#include "volumes/kernel/OrbImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedOrb; )
VECGEOM_DEVICE_DECLARE_CONV( PlacedOrb );

inline namespace VECGEOM_IMPL_NAMESPACE {

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
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedOrb() {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  UnplacedOrb const* GetUnplacedVolume() const {
    return static_cast<UnplacedOrb const *>(
        logical_volume()->unplaced_volume());
  }
  

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetRadius() const { return GetUnplacedVolume()->GetRadius(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetfRTolO() const { return GetUnplacedVolume()->GetfRTolO(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetfRTolI() const { return GetUnplacedVolume()->GetfRTolI(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetfRTolerance() const { return GetUnplacedVolume()->GetfRTolerance(); }
  
   #ifdef VECGEOM_USOLIDS  
   Precision Capacity() const  { return GetUnplacedVolume()->Capacity(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision SurfaceArea() const  { return GetUnplacedVolume()->SurfaceArea(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  std::string GetEntityType() const { return GetUnplacedVolume()->GetEntityType() ;}
  
  VECGEOM_CUDA_HEADER_BOTH
  void Extent( Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const { return GetUnplacedVolume()->Extent(aMin,aMax);}
  
  VECGEOM_CUDA_HEADER_BOTH
  void GetParametersList(int aNumber, double *aArray) const { return GetUnplacedVolume()->GetParametersList(aNumber, aArray);} 
  
  //#ifdef VECGEOM_USOLIDS
  //VECGEOM_CUDA_HEADER_BOTH
  //#endif
  Vector3D<Precision>  GetPointOnSurface() const { return GetUnplacedVolume()->GetPointOnSurface();}
 
  // VECGEOM_CUDA_HEADER_BOTH
  // void ComputeBBox() const { return GetUnplacedVolume()->ComputeBBox();}
  
  #endif

  VECGEOM_CUDA_HEADER_BOTH
  bool Normal(Vector3D<Precision> const & point, Vector3D<Precision> & normal ) const
  {
      bool valid;
      OrbImplementation<translation::kIdentity, rotation::kIdentity>::NormalKernel<kScalar>(
              *GetUnplacedVolume(),
              point,
              normal, valid);
      return valid;
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
#endif // VECGEOM_NVCC

};

} } // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDORB_H_
