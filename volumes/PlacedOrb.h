/// \file PlacedOrb.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDORB_H_
#define VECGEOM_VOLUMES_PLACEDORB_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedOrb.h"
#include "volumes/kernel/OrbImplementation.h"

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
  Precision GetfRTolO() const { return GetUnplacedVolume()->GetfRTolO(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolI() const { return GetUnplacedVolume()->GetfRTolI(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolerance() const { return GetUnplacedVolume()->GetfRTolerance(); }
  
   VECGEOM_CUDA_HEADER_BOTH
  Precision Capacity() const  { return GetUnplacedVolume()->Capacity(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision SurfaceArea() const  { return GetUnplacedVolume()->SurfaceArea(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  std::string GetEntityType() const { return GetUnplacedVolume()->GetEntityType() ;}
  
  VECGEOM_CUDA_HEADER_BOTH
  void Extent( Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const { return GetUnplacedVolume()->Extent(aMin,aMax);}
  
  VECGEOM_CUDA_HEADER_BOTH
  void GetParametersList(int aNumber, double *aArray) const { return GetUnplacedVolume()->GetParametersList(aNumber, aArray);} 
  
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision>  GetPointOnSurface() const { return GetUnplacedVolume()->GetPointOnSurface();}
 
  
  VECGEOM_CUDA_HEADER_BOTH
  void ComputeBBox() const { return GetUnplacedVolume()->ComputeBBox();}

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



#ifdef VECGEOM_BENCHMARK
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