#include "volumes/specialized_box.h"

namespace vecgeom {

#ifdef VECGEOM_NVCC

class LogicalVolume;
class TransformationMatrix;
class VPlacedVolume;

__global__
void ConstructOnGpu(
    const int trans_code, const int rot_code,
    LogicalVolume const *const logical_volume,
    TransformationMatrix const *const matrix,
    const int id,
    VPlacedVolume *const gpu_ptr) {
#ifdef VECGEOM_CUDA_NO_SPECIALIZATION
  new(gpu_ptr) vecgeom_cuda::PlacedBox(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::TransformationMatrix const*>(matrix),
    id
  );
#else
  vecgeom_cuda::UnplacedBox::CreateSpecializedVolume(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::TransformationMatrix const*>(matrix),
    trans_code,
    rot_code,
    id,
    reinterpret_cast<vecgeom_cuda::VPlacedVolume*>(gpu_ptr)
  );
#endif
}

void SpecializedBox_CopyToGpu(const int trans_code, const int rot_code,
                              LogicalVolume const *const logical_volume,
                              TransformationMatrix const *const matrix,
                              const int id, VPlacedVolume *const gpu_ptr) {
  ConstructOnGpu<<<1, 1>>>(trans_code, rot_code, logical_volume, matrix, id,
                           gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom