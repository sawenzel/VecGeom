/// \file PlacedSphere.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDSPHERE_H_
#define VECGEOM_VOLUMES_PLACEDSPHERE_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedSphere.h"
#include "volumes/kernel/SphereImplementation.h"

namespace VECGEOM_NAMESPACE {

class PlacedSphere : public VPlacedVolume {

public:

  typedef UnplacedSphere UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedSphere(char const *const label,
                       LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedSphere(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : PlacedSphere("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedSphere(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual ~PlacedSphere() {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedSphere const* GetUnplacedVolume() const {
    return static_cast<UnplacedSphere const *>(
        logical_volume()->unplaced_volume());
  }
  

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetInsideRadius() const { return GetUnplacedVolume()->GetInsideRadius(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetInnerRadius() const { return GetUnplacedVolume()->GetInnerRadius(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetOuterRadius() const { return GetUnplacedVolume()->GetOuterRadius(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetStartPhiAngle() const { return GetUnplacedVolume()->GetStartPhiAngle(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDeltaPhiAngle() const { return GetUnplacedVolume()->GetDeltaPhiAngle(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetStartThetaAngle() const { return GetUnplacedVolume()->GetStartThetaAngle(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDeltaThetaAngle() const { return GetUnplacedVolume()->GetDeltaThetaAngle(); }
  
  //Functions to get Tolerance
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetFRminTolerance() const { return GetUnplacedVolume()->GetFRminTolerance(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAngTolerance() const { return GetUnplacedVolume()->GetAngTolerance(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  bool IsFullSphere() const { return GetUnplacedVolume()->IsFullSphere(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  bool IsFullPhiSphere() const { return GetUnplacedVolume()->IsFullPhiSphere(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  bool IsFullThetaSphere() const { return GetUnplacedVolume()->IsFullThetaSphere(); }
  
  // Old access functions
VECGEOM_CUDA_HEADER_BOTH  
Precision GetRmin() const { return GetUnplacedVolume()->GetRmin(); }

VECGEOM_CUDA_HEADER_BOTH  
Precision GetRmax() const { return GetUnplacedVolume()->GetRmax(); }

VECGEOM_CUDA_HEADER_BOTH  
Precision GetSPhi() const { return GetUnplacedVolume()->GetSPhi(); }

VECGEOM_CUDA_HEADER_BOTH  
Precision GetDPhi() const { return GetUnplacedVolume()->GetDPhi(); }

VECGEOM_CUDA_HEADER_BOTH  
Precision GetSTheta() const { return GetUnplacedVolume()->GetSTheta(); }

VECGEOM_CUDA_HEADER_BOTH  
Precision GetDTheta() const { return GetUnplacedVolume()->GetDTheta(); }

VECGEOM_CUDA_HEADER_BOTH  
Precision Capacity() const { return GetUnplacedVolume()->Capacity(); }



  /*
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolO() const { return GetUnplacedVolume()->GetfRTolO(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolI() const { return GetUnplacedVolume()->GetfRTolI(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolerance() const { return GetUnplacedVolume()->GetfRTolerance(); }
  
   VECGEOM_CUDA_HEADER_BOTH
  Precision Capacity() const  { return GetUnplacedVolume()->Capacity(); }
  */
  
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
      SphereImplementation<translation::kIdentity, rotation::kIdentity>::NormalKernel<kScalar>(
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

#endif // VECGEOM_VOLUMES_PLACEDSPHERE_H_