/// \file UnplacedOrb.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDORB_H_
#define VECGEOM_VOLUMES_UNPLACEDORB_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"

namespace VECGEOM_NAMESPACE {

class UnplacedOrb : public VUnplacedVolume, public AlignedBase {

private:

 // Member variables go here
  Precision fR,fRTolerance, fRTolI, fRTolO;
  Vector3D<Precision> dimensions_;

  // Precomputed values computed from parameters
  Precision fCubicVolume, fSurfaceArea;

public:
    
  //constructor
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb();

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb(const Precision r);

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetRadius() const { return fR; }
  
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision>  dimensions() const { return dimensions_; }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision x() const { return dimensions_[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision y() const { return dimensions_[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision z() const { return dimensions_[2]; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetRadialTolerance() const { return fRTolerance; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolO() const { return fRTolO; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolI() const { return fRTolI; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetVolume() const { return fCubicVolume; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetSurfaceArea() const { return fSurfaceArea; }

  VECGEOM_CUDA_HEADER_BOTH
  void SetRadius (const Precision r);
  
public:
  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VUnplacedVolume* CopyToGpu() const;
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const;
#endif

private:

  virtual void Print(std::ostream &os) const;

  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
      const int id,
#endif
      VPlacedVolume *const placement = NULL) const;

};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDORB_H_
