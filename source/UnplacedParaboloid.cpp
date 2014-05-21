/// \file UnplacedParaboloid.cpp
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/UnplacedParaboloid.h"

#include "management/volume_factory.h"
#include "volumes/SpecializedParaboloid.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {
    
    
    VECGEOM_CUDA_HEADER_BOTH
    UnplacedParaboloid::UnplacedParaboloid()
    {
        //dummy constructor
        fRlo = 0;
        fRhi = 0;
        fDz = 0;
        fA = 0;
        fB = 0;
    }
    
    VECGEOM_CUDA_HEADER_BOTH
    UnplacedParaboloid::UnplacedParaboloid(const Precision rlo,  const Precision rhi, const Precision dz)
    {
        SetRloAndRhiAndDz(rlo, rhi, dz);
    }

    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedParaboloid::SetRlo(const Precision rlo) {
        SetRloAndRhiAndDz(rlo, fRhi, fDz);
    }
    
    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedParaboloid::SetRhi(const Precision rhi) {
        SetRloAndRhiAndDz(fRlo, rhi, fDz);
    }
    
    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedParaboloid::SetDz(const Precision dz) {
        SetRloAndRhiAndDz(fRlo, fRhi, dz);
    }
    
    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedParaboloid::SetRloAndRhiAndDz(const Precision rlo,
                                               const Precision rhi, const Precision dz) {
        
        if ((rlo<0) || (rhi<0) || (dz<=0)) {
            
            std::cout<<"Error SetRloAndRhiAndDz: invadil dimensions. Check (rlo>=0) (rhi>=0) (dz>0)\n";
            return;
        }
        fRlo = rlo;
        fRhi = rhi;
        fDz = dz;
        Precision dd = 1./(fRhi*fRhi - fRlo*fRlo);
        fA = 2.*fDz*dd;
        fB = - fDz * (fRlo*fRlo + fRhi*fRhi)*dd;
    }
    
    
    void UnplacedParaboloid::Print() const {
        printf("UnplacedParaboloid {%.2f, %.2f, %.2f, %.2f, %.2f}",
               GetRlo(), GetRhi(), GetDz(), GetA(), GetB());
    }
    
    void UnplacedParaboloid::Print(std::ostream &os) const {
        os << "UnplacedParaboloid {" << GetRlo() << ", " << GetRhi() << ", " << GetDz()
        << ", " << GetA() << ", " << GetB();
    }


template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedParaboloid::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {
  if (placement) {
    return new(placement) SpecializedParaboloid<transCodeT, rotCodeT>(
#ifdef VECGEOM_NVCC
        logical_volume, transformation, NULL, id); // TODO: add bounding box?
#else
        logical_volume, transformation);
#endif
  }
  return new SpecializedParaboloid<transCodeT, rotCodeT>(
#ifdef VECGEOM_NVCC
      logical_volume, transformation, NULL, id); // TODO: add bounding box?
#else
      logical_volume, transformation);
#endif
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedParaboloid::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {
  return VolumeFactory::CreateByTransformation<
      UnplacedParaboloid>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void UnplacedParaboloid_CopyToGpu(VUnplacedVolume *const gpu_ptr);

VUnplacedVolume* UnplacedParaboloid::CopyToGpu(
    VUnplacedVolume *const gpu_ptr) const {
  UnplacedParaboloid_CopyToGpu(gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VUnplacedVolume* UnplacedParaboloid::CopyToGpu() const {
  VUnplacedVolume *const gpu_ptr = AllocateOnGpu<UnplacedParaboloid>();
  return this->CopyToGpu(gpu_ptr);
}

#endif

#ifdef VECGEOM_NVCC

class VUnplacedVolume;

__global__
void UnplacedParaboloid_ConstructOnGpu(VUnplacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::UnplacedParaboloid();
}

void UnplacedParaboloid_CopyToGpu(VUnplacedVolume *const gpu_ptr) {
  UnplacedParaboloid_ConstructOnGpu<<<1, 1>>>(gpu_ptr);
}

#endif

} // End namespace vecgeom