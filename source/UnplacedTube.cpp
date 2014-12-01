/// \file UnplacedTube.cpp
/// \author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/UnplacedTube.h"
#include "volumes/SpecializedTube.h"
#include "volumes/utilities/GenerationUtilities.h"

#include "management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void UnplacedTube::Print() const {
  printf("UnplacedTube {%.2f, %.2f, %.2f, %.2f, %.2f}",
         rmin(), rmax(), z(), sphi(), dphi() );
}

void UnplacedTube::Print(std::ostream &os) const {
  os << "UnplacedTube {" << rmin() << ", " << rmax() << ", " << z()
     << ", " << sphi() << ", " << dphi();
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTube::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {

      using namespace TubeTypes;
      __attribute__((unused)) const UnplacedTube &tube = static_cast<const UnplacedTube&>( *(logical_volume->unplaced_volume()) );

      #ifdef VECGEOM_NVCC
        #define RETURN_SPECIALIZATION(tubeTypeT) return CreateSpecializedWithPlacement< \
            SpecializedTube<transCodeT, rotCodeT, tubeTypeT> >(logical_volume, transformation, id, placement)
      #else
        #define RETURN_SPECIALIZATION(tubeTypeT) return CreateSpecializedWithPlacement< \
            SpecializedTube<transCodeT, rotCodeT, tubeTypeT> >(logical_volume, transformation, placement)
      #endif

#ifdef GENERATE_TUBE_SPECIALIZATIONS
      if(tube.rmin() <= 0) {
        if(tube.dphi() >= 2*M_PI)  RETURN_SPECIALIZATION(NonHollowTube);
        if(tube.dphi() == M_PI)    RETURN_SPECIALIZATION(NonHollowTubeWithPiSector); // == M_PI ???

        if(tube.dphi() < M_PI)     RETURN_SPECIALIZATION(NonHollowTubeWithSmallerThanPiSector);
        if(tube.dphi() > M_PI)     RETURN_SPECIALIZATION(NonHollowTubeWithBiggerThanPiSector);
      }
      else if(tube.rmin() > 0) {
        if(tube.dphi() >= 2*M_PI)  RETURN_SPECIALIZATION(HollowTube);
        if(tube.dphi() == M_PI)    RETURN_SPECIALIZATION(HollowTubeWithPiSector); // == M_PI ???

        if(tube.dphi() < M_PI)     RETURN_SPECIALIZATION(HollowTubeWithSmallerThanPiSector);
        if(tube.dphi() > M_PI)     RETURN_SPECIALIZATION(HollowTubeWithBiggerThanPiSector);
      }
#endif

      RETURN_SPECIALIZATION(UniversalTube);

      #undef RETURN_SPECIALIZATION
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTube::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {

  return VolumeFactory::CreateByTransformation<
      UnplacedTube>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTube::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
   return CopyToGpuImpl<UnplacedTube>(in_gpu_ptr, rmin(), rmax(), z(), sphi(), dphi());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTube::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedTube>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedTube>::SizeOf();
template void DevicePtr<cuda::UnplacedTube>::Construct(
    const Precision rmin, const Precision rmax, const Precision z, 
    const Precision sphi, const Precision dphi) const;

} // End cxx namespace

#endif

} // End global namespace
