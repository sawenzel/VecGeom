/// \file Vector.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "base/Vector.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifdef VECGEOM_NVCC

template void DevicePtr<cuda::Vector<Precision> >::SizeOf();
template void DevicePtr<cuda::Vector<Precision> >::Construct(
   DevicePtr<Precision> const arr,
   const int size);

template void DevicePtr<cuda::Vector<cuda::VPlacedVolume* > >::SizeOf();
template void DevicePtr<cuda::Vector<Precision> >::Construct(
   DevicePtr<cuda::VPlacedVolume*> const arr,
   const int size);


#endif

} } // End global namespace
