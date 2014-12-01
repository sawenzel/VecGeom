/// \file AOS3D.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "base/AOS3D.h"

#ifdef VECGEOM_CUDA
#include "backend/cuda/Interface.h"
#endif

namespace vecgeom {

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr< cuda::AOS3D<Precision> >::SizeOf();
template void DevicePtr< cuda::AOS3D<Precision> >::Construct(
   DevicePtr<cuda::Vector3D<Precision> > content,
   size_t size) const;

} // End cxx namespace

#endif

} // End namespace vecgeom
