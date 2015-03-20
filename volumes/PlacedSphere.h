/// \file PlacedSphere.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDSPHERE_H_
#define VECGEOM_VOLUMES_PLACEDSPHERE_H_

#include "base/Global.h"
#include "backend/Backend.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/UnplacedSphere.h"
#include "volumes/kernel/SphereImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedSphere; )
VECGEOM_DEVICE_DECLARE_CONV( PlacedSphere );

inline namespace VECGEOM_IMPL_NAMESPACE {

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
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedSphere() {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  UnplacedSphere const* GetUnplacedVolume() const {
    return static_cast<UnplacedSphere const *>(
        GetLogicalVolume()->unplaced_volume());
  }
  

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetInsideRadius() const { return GetUnplacedVolume()->GetInsideRadius(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetInnerRadius() const { return GetUnplacedVolume()->GetInnerRadius(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetOuterRadius() const { return GetUnplacedVolume()->GetOuterRadius(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetStartPhiAngle() const { return GetUnplacedVolume()->GetStartPhiAngle(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDeltaPhiAngle() const { return GetUnplacedVolume()->GetDeltaPhiAngle(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetStartThetaAngle() const { return GetUnplacedVolume()->GetStartThetaAngle(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDeltaThetaAngle() const { return GetUnplacedVolume()->GetDeltaThetaAngle(); }
  
  //Functions to get Tolerance
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetFRminTolerance() const { return GetUnplacedVolume()->GetFRminTolerance(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetAngTolerance() const { return GetUnplacedVolume()->GetAngTolerance(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool IsFullSphere() const { return GetUnplacedVolume()->IsFullSphere(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool IsFullPhiSphere() const { return GetUnplacedVolume()->IsFullPhiSphere(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool IsFullThetaSphere() const { return GetUnplacedVolume()->IsFullThetaSphere(); }
  
  //Function to return all Trignometric values 
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetHDPhi() const { return GetUnplacedVolume()->GetHDPhi();}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetCPhi() const { return GetUnplacedVolume()->GetCPhi() ;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetEPhi() const { return GetUnplacedVolume()->GetEPhi();}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetSinCPhi() const { return GetUnplacedVolume()->GetSinCPhi();}
  
  VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
  Precision GetCosCPhi() const { return GetUnplacedVolume()->GetCosCPhi();}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetSinSPhi() const { return GetUnplacedVolume()->GetSinSPhi();}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetCosSPhi() const { return GetUnplacedVolume()->GetCosSPhi();}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetSinEPhi() const { return GetUnplacedVolume()->GetSinEPhi();}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetCosEPhi() const { return GetUnplacedVolume()->GetCosEPhi();}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetETheta() const { return GetUnplacedVolume()->GetETheta();}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetSinSTheta() const { return GetUnplacedVolume()->GetSinSTheta();}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetCosSTheta() const { return GetUnplacedVolume()->GetCosSTheta();}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetSinETheta() const { return GetUnplacedVolume()->GetSinETheta();}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetCosETheta() const { return GetUnplacedVolume()->GetCosETheta();}
  
  // Old access functions
VECGEOM_CUDA_HEADER_BOTH 
  VECGEOM_INLINE
Precision GetRmin() const { return GetUnplacedVolume()->GetRmin(); }

VECGEOM_CUDA_HEADER_BOTH  
VECGEOM_INLINE
Precision GetRmax() const { return GetUnplacedVolume()->GetRmax(); }

VECGEOM_CUDA_HEADER_BOTH 
VECGEOM_INLINE
Precision GetSPhi() const { return GetUnplacedVolume()->GetSPhi(); }

VECGEOM_CUDA_HEADER_BOTH 
VECGEOM_INLINE
Precision GetDPhi() const { return GetUnplacedVolume()->GetDPhi(); }

VECGEOM_CUDA_HEADER_BOTH 
VECGEOM_INLINE
Precision GetSTheta() const { return GetUnplacedVolume()->GetSTheta(); }

VECGEOM_CUDA_HEADER_BOTH 
VECGEOM_INLINE
Precision GetDTheta() const { return GetUnplacedVolume()->GetDTheta(); }

VECGEOM_CUDA_HEADER_BOTH  
virtual Precision Capacity() { return GetUnplacedVolume()->Capacity(); }


  /*
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolO() const { return GetUnplacedVolume()->GetfRTolO(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolI() const { return GetUnplacedVolume()->GetfRTolI(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolerance() const { return GetUnplacedVolume()->GetfRTolerance(); }
  
  */
  
  VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
  Precision SurfaceArea() const  { return GetUnplacedVolume()->SurfaceArea(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  std::string GetEntityType() const { return GetUnplacedVolume()->GetEntityType() ;}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Extent( Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const { return GetUnplacedVolume()->Extent(aMin,aMax);}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void GetParametersList(int aNumber, double *aArray) const { return GetUnplacedVolume()->GetParametersList(aNumber, aArray);}
  
#if !defined(VECGEOM_NVCC)
  Vector3D<Precision> GetPointOnSurface() const {
    return GetUnplacedVolume()->GetPointOnSurface();
  }
#endif
 
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
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

#endif // VECGEOM_VOLUMES_PLACEDSPHERE_H_
