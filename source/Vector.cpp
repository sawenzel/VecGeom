/// \file Vector.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "base/Vector.h"

namespace vecgeom {

#ifdef VECGEOM_NVCC

inline namespace cuda { class VPlacedVolume; }

namespace cxx {

template size_t DevicePtr<cuda::Vector<Precision> >::SizeOf();
template void DevicePtr<cuda::Vector<Precision> >::Construct(
   DevicePtr<Precision> const arr,
   const int size) const;
template void DevicePtr<cuda::Vector<Precision> >::Construct(
   DevicePtr<Precision> const arr,
   const int size,
   const int maxsize) const;

template size_t DevicePtr<cuda::Vector<cuda::VPlacedVolume* > >::SizeOf();
template void DevicePtr<cuda::Vector<cuda::VPlacedVolume* > >::Construct(
   DevicePtr<cuda::VPlacedVolume*> const arr,
   const int size) const;
template void DevicePtr<cuda::Vector<cuda::VPlacedVolume* > >::Construct(
   DevicePtr<cuda::VPlacedVolume*> const arr,
   const int size,
   const int maxsize) const;

template size_t DevicePtr<cuda::Vector<cuda::VPlacedVolume const * > >::SizeOf();
template void DevicePtr<cuda::Vector<cuda::VPlacedVolume const * > >::Construct(
   DevicePtr<cuda::VPlacedVolume const*> const arr,
   const int size) const;
template void DevicePtr<cuda::Vector<cuda::VPlacedVolume const * > >::Construct(
   DevicePtr<cuda::VPlacedVolume const*> const arr,
   const int size,
   const int maxsize) const;

} // End cxx namespace

#endif

} // End global namespace
