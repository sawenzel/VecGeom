/// \file UnplacedTorus.cpp

#include "volumes/UnplacedTorus.h"
#include "volumes/SpecializedTorus.h"

#include "management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

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

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTorus::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
   return CopyToGpuImpl<UnplacedTorus>(in_gpu_ptr, fRmin, fRmax, fRtor, fSphi, fDphi);
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTorus::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedTorus>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedTorus>::SizeOf();
template void DevicePtr<cuda::UnplacedTorus>::Construct(
    const Precision rmin, const Precision rmax, const Precision rtor,
    const Precision sphi, const Precision dphi) const;

} // End cxx namespace

#endif

} // End global namespace

