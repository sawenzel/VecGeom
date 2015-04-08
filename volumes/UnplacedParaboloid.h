//===-- volumes/UnplacedParaboloid.h - Instruction class definition -------*- C++ -*-===//
///
/// \file volumes/UnplacedParaboloid.h
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)
/// \brief This file contains the declaration of the UnplacedParaboloid class
///
/// _____________________________________________________________________________
/// A paraboloid is the solid bounded by the following surfaces:
/// - 2 planes parallel with XY cutting the Z axis at Z=-dz and Z=+dz
/// - the surface of revolution of a parabola described by:
/// z = a*(x*x + y*y) + b
/// The parameters a and b are automatically computed from:
/// - rlo is the radius of the circle of intersection between the
/// parabolic surface and the plane z = -dz
/// - rhi is the radius of the circle of intersection between the
/// parabolic surface and the plane z = +dz
/// -dz = a*rlo^2 + b
/// dz = a*rhi^2 + b      where: rhi>rlo, both >= 0
///
/// note:
/// dd = 1./(rhi^2 - rlo^2);
/// a = 2.*dz*dd;
/// b = - dz * (rlo^2 + rhi^2)*dd;
///
/// in respect with the G4 implementation we have:
/// k1=1/a
/// k2=-b/a
///
/// a=1/k1
/// b=-k2/k1
//===----------------------------------------------------------------------===//

#ifndef VECGEOM_VOLUMES_UNPLACEDPARABOLOID_H_
#define VECGEOM_VOLUMES_UNPLACEDPARABOLOID_H_

#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedParaboloid; )
VECGEOM_DEVICE_DECLARE_CONV( UnplacedParaboloid );

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedParaboloid : public VUnplacedVolume, public AlignedBase {

private:

    Precision fRlo, fRhi, fDz;
    // Precomputed values computed from parameters
    Precision fA, fB,
    
    //useful parameters in order to improve performance
    fAinv,
    fBinv,
    fA2,
    fB2,
    fRlo2,
    fRhi2,
    //Inside tolerance for plane at dZ
    fTolIz,
    //Outside tolerance for plane at -dZ
    fTolOz,
    //Inside tolerance for Rlo, squared
    fTolIrlo2,
    //Outside tolerance for Rlo, squared
    fTolOrlo2,
    //Inside tolerance for Rhi, squared
    fTolIrhi2,
    //Outside tolerance for Rhi, squared
    fTolOrhi2;
    
    Precision fDx, fDy;
    
public:
    
    //constructor
    VECGEOM_CUDA_HEADER_BOTH
    UnplacedParaboloid();
    
    VECGEOM_CUDA_HEADER_BOTH
    UnplacedParaboloid(const Precision rlo, const Precision rhi, const Precision dz);
    
    //get and set
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRlo() const { return fRlo; }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRhi() const { return fRhi; }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetDz() const { return fDz; }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetA() const { return fA; }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetB() const { return fB; }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetAinv() const { return fAinv; }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetBinv() const { return fBinv; }
   
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetA2() const { return fA2; }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetB2() const { return fB2; }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRlo2() const { return fRlo2; }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRhi2() const { return fRhi2; }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTolIz() const { return fTolIz; }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTolOz() const { return fTolOz; }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTolIrlo2() const { return fTolIrlo2; }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTolOrlo2() const { return fTolOrlo2; }

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTolIrhi2() const { return fTolIrhi2; }
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTolOrhi2() const { return fTolOrhi2; }

    VECGEOM_CUDA_HEADER_BOTH
    void SetRloAndRhiAndDz(const Precision rlo, const Precision rhi, const Precision dz);
    
    VECGEOM_CUDA_HEADER_BOTH
    void SetRlo(const Precision rlo);
    
    VECGEOM_CUDA_HEADER_BOTH
    void SetRhi(const Precision rhi);
    
    VECGEOM_CUDA_HEADER_BOTH
    void SetDx(const Precision dx);
    
    VECGEOM_CUDA_HEADER_BOTH
    void SetDy(const Precision dy);
    
    VECGEOM_CUDA_HEADER_BOTH
    void SetDz(const Precision dz);
    
#if !defined(VECGEOM_NVCC)
//__________________________________________________________________
    
    void Normal(const Precision *point, const Precision *dir, Precision *norm) const;

//__________________________________________________________________
    
    void Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const;

//__________________________________________________________________

    // Computes capacity of the shape in [length^3]
    Precision Capacity() const { return kPi*fDz*(fRlo*fRlo+fRhi*fRhi);}
//__________________________________________________________________
    
    Precision SurfaceArea() const;
//__________________________________________________________________

    virtual Vector3D<Precision> GetPointOnSurface() const;
//__________________________________________________________________
    
    std::string GetEntityType() const{ return "Paraboloid";}
#endif // !VECGEOM_NVCC

    VECGEOM_CUDA_HEADER_BOTH
    void ComputeBoundingBox();
//__________________________________________________________________

    
    VECGEOM_CUDA_HEADER_BOTH
    void GetParameterList() const{;}
    
//__________________________________________________________________
    
    VECGEOM_CUDA_HEADER_BOTH
    UnplacedParaboloid* Clone() const{ return new UnplacedParaboloid(fRlo, fRhi, fDz);}

//__________________________________________________________________
    
   
    VECGEOM_CUDA_HEADER_BOTH
    void StreamInfo(std::ostream &os) const;
    
//__________________________________________________________________
    

    virtual int memory_size() const { return sizeof(*this); }

//__________________________________________________________________
    

    VECGEOM_CUDA_HEADER_BOTH
    virtual void Print() const;
//__________________________________________________________________
    

    virtual void Print(std::ostream &os) const;
    
//__________________________________________________________________
    


    template <TranslationCode transCodeT, RotationCode rotCodeT>
    VECGEOM_CUDA_HEADER_DEVICE
    static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedParaboloid>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

private:

  
    //Specialized Volume
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

} } // End global namespace

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif

#endif // VECGEOM_VOLUMES_UNPLACEDPARABOLOID_H_
