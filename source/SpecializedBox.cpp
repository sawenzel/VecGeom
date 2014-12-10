#include "volumes/SpecializedBox.h"

namespace vecgeom {

#ifdef VECGEOM_NVCC

inline namespace cuda {

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void ConstructOnGpu(
    const int trans_code, const int rot_code,
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id,
    VPlacedVolume *const gpu_ptr) {
#ifdef VECGEOM_CUDA_NO_VOLUME_SPECIALIZATION
  new(gpu_ptr) vecgeom::cuda::PlacedBox(
    reinterpret_cast<vecgeom::cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom::cuda::Transformation3D const*>(transformation),
    id
  );
#else
  vecgeom::cuda::UnplacedBox::CreateSpecializedVolume(
    reinterpret_cast<vecgeom::cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom::cuda::Transformation3D const*>(transformation),
    trans_code,
    rot_code,
    id,
    reinterpret_cast<vecgeom::cuda::VPlacedVolume*>(gpu_ptr)
  );
#endif
}

void SpecializedBox_CopyToGpu(const int trans_code, const int rot_code,
                              LogicalVolume const *const logical_volume,
                              Transformation3D const *const transformation,
                              const int id, VPlacedVolume *const gpu_ptr) {
  ConstructOnGpu<<<1, 1>>>(trans_code, rot_code, logical_volume,
                           transformation, id, gpu_ptr);
}

} // End namespace cuda

#endif // VECGEOM_NVCC

} // End namespace vecgeom
