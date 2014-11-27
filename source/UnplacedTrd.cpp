/// @file UnplacedTrd.cpp
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/UnplacedTrd.h"
#include "volumes/SpecializedTrd.h"
#include "volumes/utilities/GenerationUtilities.h"

#include "management/VolumeFactory.h"


namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void UnplacedTrd::Print() const {
  printf("UnplacedTrd {%.2f, %.2f, %.2f, %.2f, %.2f}",
         dx1(), dx2(), dy1(), dy2(), dz() );
}

void UnplacedTrd::Print(std::ostream &os) const {
  os << "UnplacedTrd {" << dx1() << ", " << dx2() << ", " << dy1()
     << ", " << dy2() << ", " << dz();
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTrd::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {

    using namespace TrdTypes;
    __attribute__((unused)) const UnplacedTrd &trd = static_cast<const UnplacedTrd&>( *(logical_volume->unplaced_volume()) );
    
    #define GENERATE_TRD_SPECIALIZATIONS
    #ifdef GENERATE_TRD_SPECIALIZATIONS
      if(trd.dy1() == trd.dy2()) {
	//          std::cout << "trd1" << std::endl;
          return CreateSpecializedWithPlacement<SpecializedTrd<transCodeT, rotCodeT, TrdTypes::Trd1> >(logical_volume, transformation
#ifdef VECGEOM_NVCC
                 ,id
#endif
                 , placement);
      } else {
	//          std::cout << "trd2" << std::endl;
          return CreateSpecializedWithPlacement<SpecializedTrd<transCodeT, rotCodeT, TrdTypes::Trd2> >(logical_volume, transformation
#ifdef VECGEOM_NVCC
                 ,id
#endif
                 , placement);
    }
    #endif
      //    std::cout << "universal trd" << std::endl; 
	return CreateSpecializedWithPlacement<SpecializedTrd<transCodeT, rotCodeT, TrdTypes::UniversalTrd> >(logical_volume, transformation 
#ifdef VECGEOM_NVCC
                ,id
#endif
                , placement);
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTrd::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {

  return VolumeFactory::CreateByTransformation<
      UnplacedTrd>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}

#ifdef VECGEOM_CUDA_INTERFACE

void UnplacedTrd_CopyToGpu(
    const Precision dx1, const Precision dx2, const Precision dy1, const Precision dy2, const Precision dz,
    VUnplacedVolume *const gpu_ptr);

VUnplacedVolume* UnplacedTrd::CopyToGpu(
    VUnplacedVolume *const gpu_ptr) const {
  UnplacedTrd_CopyToGpu(dx1(), dx2(), dy1(), dy2(), dz(), gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VUnplacedVolume* UnplacedTrd::CopyToGpu() const {
  VUnplacedVolume *const gpu_ptr = AllocateOnGpu<UnplacedTrd>();
  return this->CopyToGpu(gpu_ptr);
}

#endif

#ifdef VECGEOM_NVCC

class VUnplacedVolume;

__global__
void UnplacedTrd_ConstructOnGpu(
    const Precision dx1, const Precision dx2, const Precision dy1, const Precision dy2, const Precision dz,
    VUnplacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::UnplacedTrd(dx1, dx2, dy1, dy2, dz);
}

void UnplacedTrd_CopyToGpu(
    const Precision dx1, const Precision dx2, const Precision dy1, const Precision dy2, const Precision dz,
    VUnplacedVolume *const gpu_ptr) {
  UnplacedTrd_ConstructOnGpu<<<1, 1>>>(dx1, dx2, dy1, dy2, dz, gpu_ptr);
}

#endif



} } // End global namespace
