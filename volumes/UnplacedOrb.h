/// \file UnplacedOrb.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDORB_H_
#define VECGEOM_VOLUMES_UNPLACEDORB_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#ifndef VECGEOM_NVCC
  #include "base/RNG.h"
#include <cassert>
#include <cmath>
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedOrb; )
VECGEOM_DEVICE_DECLARE_CONV( UnplacedOrb );

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedOrb : public VUnplacedVolume, public AlignedBase {

private:

 // Member variables go here
  Precision fR,fRTolerance, fRTolI, fRTolO;
  
  // Precomputed values computed from parameters
  Precision fCubicVolume, fSurfaceArea;
  
  //Tolerance compatiable with USolids
  Precision epsilon;// = 2e-11; 
  Precision frTolerance;//=1e-9;

public:
    
  //constructor
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb();

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb(const Precision r);
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision MyMax(Precision a, Precision b);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetRadius() const { return fR; }
  
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetfRTolO() const { return fRTolO; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetfRTolI() const { return fRTolI; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetfRTolerance() const { return fRTolerance; }
  
  VECGEOM_CUDA_HEADER_BOTH
  //VECGEOM_INLINE
  void SetRadius (const Precision r);
  
  //_____________________________________________________________________________
  
  VECGEOM_CUDA_HEADER_BOTH
  void Extent( Vector3D<Precision> &, Vector3D<Precision> &) const;
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision Capacity() const { return fCubicVolume; }
  
  VECGEOM_CUDA_HEADER_BOTH
  //VECGEOM_INLINE
  Precision SurfaceArea() const { return fSurfaceArea; }
  
#if !defined(VECGEOM_NVCC)
  virtual Vector3D<Precision> GetPointOnSurface() const;
#endif 
  
  //VECGEOM_CUDA_HEADER_BOTH
  std::string GetEntityType() const;
  
    
  VECGEOM_CUDA_HEADER_BOTH
  void GetParametersList(int aNumber, double *aArray) const; 
  
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb* Clone() const;

  //VECGEOM_CUDA_HEADER_BOTH
  std::ostream& StreamInfo(std::ostream &os) const;
    
  // VECGEOM_CUDA_HEADER_BOTH
  // void ComputeBBox() const; 
  
public:
  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  virtual void Print(std::ostream &os) const;

  #ifndef VECGEOM_NVCC

  template <TranslationCode trans_code, RotationCode rot_code>
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  static VPlacedVolume* CreateSpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL);

  #else

  template <TranslationCode trans_code, RotationCode rot_code>
  __device__
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
                               const int id,
                               VPlacedVolume *const placement = NULL);

  __device__
  static VPlacedVolume* CreateSpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      const int id, VPlacedVolume *const placement = NULL);

  #endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedOrb>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif
  
  

private:

  //virtual void Print(std::ostream &os) const;

  #ifndef VECGEOM_NVCC

  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL) const {
    return CreateSpecializedVolume(volume, transformation, trans_code, rot_code,
                                   placement);
  }

#else

  __device__
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      const int id, VPlacedVolume *const placement = NULL) const {
    return CreateSpecializedVolume(volume, transformation, trans_code, rot_code,
                                   id, placement);
  }

#endif

};

} } // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDORB_H_
