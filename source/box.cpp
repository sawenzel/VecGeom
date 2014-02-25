#include "volumes/box.h"
#ifdef VECGEOM_NVCC
#include "backend/cuda_backend.cuh"
#endif

namespace vecgeom {

#ifdef VECGEOM_NVCC
void UnplacedBox::CopyToGpu(VUnplacedVolume *const target) const {
  vecgeom::CopyToGpu(this, static_cast<UnplacedBox*>(target), 1);
}
#endif

} // End namespace vecgeom