/// \file UnplacedTorus.cpp

#include "volumes/UnplacedTorus.h"
#include "volumes/SpecializedTorus.h"

#include "management/VolumeFactory.h"

namespace VECGEOM_NAMESPACE {

void UnplacedTorus::Print() const {
  printf("UnplacedTorus {%.2f, %.2f, %.2f, %.2f, %.2f}",
         rmin(), rmax(), rtor(), sphi(), dphi() );
}

void UnplacedTorus::Print(std::ostream &os) const {
  os << "UnplacedTube {" << rmin() << ", " << rmax() << ", " << rtor()
     << ", " << sphi() << ", " << dphi();
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTorus::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedTorus<transCodeT, rotCodeT>(logical_volume,
							  transformation
#ifdef VECGEOM_NVCC
							  , NULL, id
#endif
);
    return placement;
  }
  return new SpecializedTorus<transCodeT, rotCodeT>(logical_volume,
                                                  transformation
#ifdef VECGEOM_NVCC
						    , NULL, id
#endif
						    );
}


VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTorus::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {

  return VolumeFactory::CreateByTransformation<
      UnplacedTorus>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void UnplacedTorus_CopyToGpu(
    const Precision rmin, const Precision rmax, const Precision rtor, const Precision sphi, const Precision dphi,
    VUnplacedVolume *const gpu_ptr);

VUnplacedVolume* UnplacedTorus::CopyToGpu(
    VUnplacedVolume *const gpu_ptr) const {
  UnplacedTorus_CopyToGpu(fRmin, fRmax, fRtor, fSphi, fDphi, gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VUnplacedVolume* UnplacedTorus::CopyToGpu() const {
  VUnplacedVolume *const gpu_ptr = AllocateOnGpu<UnplacedTorus>();
  return this->CopyToGpu(gpu_ptr);
}

#endif

#ifdef VECGEOM_NVCC

class VUnplacedVolume;

__global__
void UnplacedTorus_ConstructOnGpu(
    const Precision rmin, const Precision rmax, const Precision rtor, const Precision sphi, const Precision dphi,
    VUnplacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::UnplacedTorus(rmin, rmax, rtor, sphi, dphi);
}

void UnplacedTorus_CopyToGpu(
    const Precision rmin, const Precision rmax, const Precision rtor, const Precision sphi, const Precision dphi,
    VUnplacedVolume *const gpu_ptr) {
  UnplacedTorus_ConstructOnGpu<<<1, 1>>>(rmin, rmax, rtor, sphi, dphi, gpu_ptr);
}

#endif

} // End namespace vecgeom
