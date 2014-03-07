#include "base/aos3d.h"
#include "base/soa3d.h"

namespace vecgeom {

template <typename Type>
VECGEOM_CUDA_HEADER_BOTH
AOS3D<Type>::AOS3D(TrackContainer<Type> const &other)
    : TrackContainer<Type>(other.size_, true) {
  data_ = static_cast<VecType*>(
            _mm_malloc(sizeof(VecType)*other.size(), kAlignmentBoundary)
          );
  const unsigned count = other.size();
  for (int i = 0; i < count; ++i) {
    data_[i] = other[i];
  }
}

} // End namespace vecgeom