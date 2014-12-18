/// \file SOA3D.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "base/SOA3D.h"
#ifdef VECGEOM_CUDA
#include "backend/cuda/Interface.h"
#endif

namespace vecgeom {

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr< cuda::SOA3D<Precision> >::SizeOf();
template void DevicePtr< cuda::SOA3D<Precision> >::Construct(
   DevicePtr<Precision> x, DevicePtr<Precision> y,DevicePtr< Precision> z,
   size_t size) const;

} // End cxx namespace

#endif

} // End namespace vecgeom
