/// \file UnplacedParaboloid.cpp
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/UnplacedParaboloid.h"

#include "management/VolumeFactory.h"
#include "volumes/SpecializedParaboloid.h"

#include <stdio.h>
#if !defined(VECGEOM_NVCC) && defined(VECGEOM_USOLIDS)
#include "base/RNG.h"
#endif

namespace VECGEOM_NAMESPACE {
    
//__________________________________________________________________
    VECGEOM_CUDA_HEADER_BOTH
    UnplacedParaboloid::UnplacedParaboloid() :
fRlo(0),
fRhi(0),
fDz(0),
fA(0),
fB(0),
fAinv(0),
fBinv(0),
fA2(0),
fB2(0),
fRlo2(0),
fRhi2(0),
fTolIz(0),
fTolOz(0),
fTolIrlo2(0),
fTolOrlo2(0),
fTolIrhi2(0),
fTolOrhi2(0),
fDx(0),
fDy(0)
    {
        //dummy constructor
    }
//__________________________________________________________________
    
    VECGEOM_CUDA_HEADER_BOTH
    UnplacedParaboloid::UnplacedParaboloid(const Precision rlo,  const Precision rhi, const Precision dz):
fRlo(0),
fRhi(0),
fDz(0),
fA(0),
fB(0),
fAinv(0),
fBinv(0),
fA2(0),
fB2(0),
fRlo2(0),
fRhi2(0),
fTolIz(0),
fTolOz(0),
fTolIrlo2(0),
fTolOrlo2(0),
fTolIrhi2(0),
fTolOrhi2(0),
fDx(0),
fDy(0)
    {
        SetRloAndRhiAndDz(rlo, rhi, dz);
    }
//__________________________________________________________________

    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedParaboloid::SetRlo(const Precision rlo) {
        SetRloAndRhiAndDz(rlo, fRhi, fDz);
    }
//__________________________________________________________________
    
    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedParaboloid::SetRhi(const Precision rhi) {
        SetRloAndRhiAndDz(fRlo, rhi, fDz);
    }

//__________________________________________________________________
    
    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedParaboloid::SetDz(const Precision dz) {
        SetRloAndRhiAndDz(fRlo, fRhi, dz);
    }
//__________________________________________________________________
    
    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedParaboloid::SetRloAndRhiAndDz(const Precision rlo,
                                               const Precision rhi, const Precision dz) {
        
        if ((rlo<0) || (rhi<0) || (dz<=0)) {
            
            printf("Error SetRloAndRhiAndDz: invadil dimensions. Check (rlo>=0) (rhi>=0) (dz>0)\n");
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
    void UnplacedParaboloid::Normal(const Precision *point, const Precision *dir, Precision *norm) const{
       
        // Compute normal to closest surface from POINT.
        norm[0] = norm[1] = 0.0;
        if (Abs(point[2]) > fDz) {
            //norm[2] = TMath::Sign(1., dir[2]);
            dir[2]>0 ? norm[2]=1 : norm[2]=-1;
            return;
        }
        Precision safz = fDz-Abs(point[2]);
        Precision r = Sqrt(point[0]*point[0]+point[1]*point[1]);
        Precision safr = Abs(r-Sqrt((point[2]-fB)*fAinv));
        if (safz<safr) {
            //norm[2] = TMath::Sign(1., dir[2]);
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
    void UnplacedParaboloid::Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const{
        
        aMin.x() = -fDx;
        aMax.x() = fDx;
        aMin.y() = -fDy;
        aMax.y() = fDy;
        aMin.z() = -fDz;
        aMax.z() = fDz;
    }
    
//__________________________________________________________________

    VECGEOM_CUDA_HEADER_BOTH
    Precision UnplacedParaboloid::SurfaceArea() const
    {
    
        //G4 implementation
        Precision h1, h2, A1, A2;
        h1=-fB+fDz;
        h2=-fB-fDz;
        
        // Calculate surface area for the paraboloid full paraboloid
        // cutoff at z = dz (not the cutoff area though).
        A1 = fRhi2 + 4 * h1*h1;
        A1 *= (A1*A1); // Sets A1 = A1^3
        A1 = kPi * fRhi /6 / (h1*h1) * ( sqrt(A1) - fRhi2 * fRhi);
        
        // Calculate surface area for the paraboloid full paraboloid
        // cutoff at z = -dz (not the cutoff area though).
        A2 = fRlo2 + 4 * (h2*h2);
        A2 *= (A2*A2); // Sets A2 = A2^3
        
        if(h2 != 0)
            A2 = kPi * fRlo /6 / (h2*h2) * (Sqrt(A2) - fRlo2 * fRlo);
        else
            A2 = 0.;
        return (A1 - A2 + (fRlo2 + fRhi2)*kPi);
    
    }
    //__________________________________________________________________

#if !defined(VECGEOM_NVCC) && defined(VECGEOM_USOLIDS)
    Vector3D<Precision> UnplacedParaboloid::GetPointOnSurface() const{
        
        //G4 implementation
        Precision A = SurfaceArea();
        Precision z = RNG::Instance().uniform(0.,1.);
        Precision phi = RNG::Instance().uniform(0.,2*kPi);
            if(kPi*(fRlo2 + fRhi2)/A >= z)
            {
                Precision rho;
                //points on the cutting circle surface at -dZ
                if(kPi * fRlo2/ A > z)
                {
                    rho = fRlo * Sqrt(RNG::Instance().uniform(0.,1.));
                    return Vector3D<Precision>(rho * cos(phi), rho * sin(phi), -fDz);
                }
                //points on the cutting circle surface at dZ
                else
                {
                    rho = fRhi * Sqrt(RNG::Instance().uniform(0.,1.));
                    return Vector3D<Precision>(rho * cos(phi), rho * sin(phi), fDz);
                }
            }
            //points on the paraboloid surface
            else
            {
                z = RNG::Instance().uniform(0.,1.)*2*fDz - fDz;
                return Vector3D<Precision>(Sqrt(z*fAinv -fB*fAinv)*cos(phi), Sqrt(z*fAinv -fB*fAinv)*sin(phi), z);
            }
        }
#endif
    
//__________________________________________________________________
    
    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedParaboloid::ComputeBoundingBox() {
        fDx= Max(fRhi, fRlo);
        fDy=fDx;
    }

//__________________________________________________________________
    
    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedParaboloid::StreamInfo(std::ostream &os) const{
        //NYI
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
