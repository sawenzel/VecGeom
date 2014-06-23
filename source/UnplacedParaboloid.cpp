/// \file UnplacedParaboloid.cpp
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/UnplacedParaboloid.h"

#include "management/VolumeFactory.h"
#include "volumes/SpecializedParaboloid.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {
    
//__________________________________________________________________
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
        
        fRlo2= fRlo*fRlo;
        fRhi2= fRhi*fRhi;
        
        fDz = dz;
        Precision dd = 1./(fRhi2 - fRlo2);
        fA = 2.*fDz*dd;
        fB = - fDz * (fRlo2 + fRhi2)*dd;
        
        
        fAinv=1/fA;
        fBinv=1/fB;
        fA2=fA*fA;
        fB2=fB*fB;
        
        //Inside tolerance for plane at dZ
        fTolIz= fDz-kHalfTolerance;
        //Outside tolerance for plane at -dZ
        fTolOz=fDz+kHalfTolerance;
        //Inside tolerance for Rlo, squared
        fTolIrlo2= (fRlo - kHalfTolerance)*(fRlo - kHalfTolerance);
        //Outside tolerance for Rlo, squared
        fTolOrlo2= (fRlo + kHalfTolerance)*(fRlo + kHalfTolerance);
        //Inside tolerance for Rhi, squared
        fTolIrhi2=(fRhi - kHalfTolerance)*(fRhi - kHalfTolerance);
        //Outside tolerance for Rhi, squared
        fTolOrhi2=(fRhi + kHalfTolerance)*(fRhi + kHalfTolerance);
        
        ComputeBoundingBox();
    }

//__________________________________________________________________

    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedParaboloid::Normal(const Precision *point, const Precision *dir, Precision *norm){
       
        // Compute normal to closest surface from POINT.
        norm[0] = norm[1] = 0.0;
        if (Abs(point[2]) > fDz) {
            //norm[2] = TMath::Sign(1., dir[2]); ------------------>
            dir[2]>0 ? norm[2]=1 : norm[2]=-1;
            return;
        }
        Precision safz = fDz-Abs(point[2]);
        Precision r = Sqrt(point[0]*point[0]+point[1]*point[1]);
        Precision safr = Abs(r-Sqrt((point[2]-fB)*fAinv));
        if (safz<safr) {
            //norm[2] = TMath::Sign(1., dir[2]); --------->
            dir[2]>0 ? norm[2]=1 : norm[2]=-1;
            return;
        }
        Precision talf = -2.*fA*r;
        Precision calf = 1./Sqrt(1.+talf*talf);
        Precision salf = talf * calf;
        Precision phi = ATan2(point[1], point[0]);
        
        norm[0] = salf*cos(phi);
        norm[1] = salf*sin(phi);
        norm[2] = calf;
        Precision ndotd = norm[0]*dir[0]+norm[1]*dir[1]+norm[2]*dir[2];
        if (ndotd < 0) {
            norm[0] = -norm[0];
            norm[1] = -norm[1];
            norm[2] = -norm[2];
        }
    }
    
//__________________________________________________________________

    // Returns the full 3D cartesian extent of the solid.
    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedParaboloid::Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax){
        
        aMin.x() = -fDx;
        aMax.x() = fDx;
        aMin.y() = -fDy;
        aMax.y() = fDy;
        aMin.z() = -fDz;
        aMax.z() = fDz;
    }
    
//__________________________________________________________________

    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedParaboloid::GetPointOnSurface(){
        
        //NYI
        ;}
    
//__________________________________________________________________
    
    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedParaboloid::ComputeBoundingBox(){
        fDx=Max(fRhi, fRlo);
        fDy=fDx;
        //fDz=fDz;
    }

//__________________________________________________________________
    
    void UnplacedParaboloid::Print() const {
        printf("UnplacedParaboloid {%.2f, %.2f, %.2f, %.2f, %.2f}",
               GetRlo(), GetRhi(), GetDz(), GetA(), GetB());
    }
//__________________________________________________________________
    
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
