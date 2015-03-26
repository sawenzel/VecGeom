/// \file UnplacedBox.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/UnplacedBox.h"

#include "backend/Backend.h"
#include "management/VolumeFactory.h"
#include "volumes/SpecializedBox.h"
#ifndef VECGEOM_NVCC
  #include "base/RNG.h"
#endif
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
void UnplacedBox::Print() const {
  printf("UnplacedBox {%.2f, %.2f, %.2f}", x(), y(), z());
}

void UnplacedBox::Print(std::ostream &os) const {
  os << "UnplacedBox {" << x() << ", " << y() << ", " << z() << "}";
}


//______________________________________________________________________________
void UnplacedBox::Extent(Vector3D<Precision> & aMin, Vector3D<Precision> & aMax) const
{
  // Returns the full 3D cartesian extent of the solid.
  aMin = -dimensions_;
  aMax =  dimensions_;
}

Vector3D<Precision> UnplacedBox::GetPointOnSurface() const
{
    Vector3D<Precision> p(dimensions_);

    double S[3] = { p[1]*p[2], p[0]*p[2], p[0]*p[1] };

    double rand = (S[0] + S[1] + S[2]) * RNG::Instance().uniform(-1.0, 1.0);

    int axis = 0, direction = rand < 0.0 ? -1 : 1;

    rand = std::abs(rand);

    while (rand > S[axis]) rand -= S[axis], axis++;

    p[0] = (axis == 0) ? direction * dimensions_[0]
                       : p[0] * RNG::Instance().uniform(-1.0, 1.0);
    p[1] = (axis == 1) ? direction * dimensions_[1]
                       : p[1] * RNG::Instance().uniform(-1.0, 1.0);
    p[2] = (axis == 2) ? direction * dimensions_[2]
                       : p[2] * RNG::Instance().uniform(-1.0, 1.0);
    return p;
}


#ifndef VECGEOM_NVCC

template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume* UnplacedBox::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedBox<trans_code, rot_code>(logical_volume,
                                                        transformation);
    return placement;
  }
  return new SpecializedBox<trans_code, rot_code>(logical_volume,
                                                  transformation);
}

VPlacedVolume* UnplacedBox::CreateSpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    VPlacedVolume *const placement) {
  return VolumeFactory::CreateByTransformation<UnplacedBox>(
           volume, transformation, trans_code, rot_code, placement
         );
}

#else

template <TranslationCode trans_code, RotationCode rot_code>
__device__
VPlacedVolume* UnplacedBox::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedBox<trans_code, rot_code>(logical_volume,
                                                        transformation, id);
    return placement;
  }
  return new SpecializedBox<trans_code, rot_code>(logical_volume,
                                                  transformation, id);
}

__device__
VPlacedVolume* UnplacedBox::CreateSpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
    const int id, VPlacedVolume *const placement) {
  return VolumeFactory::CreateByTransformation<UnplacedBox>(
           volume, transformation, trans_code, rot_code, id, placement
         );
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedBox::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
   return CopyToGpuImpl<UnplacedBox>(in_gpu_ptr, x(), y(), z());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedBox::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedBox>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedBox>::SizeOf();
template void DevicePtr<cuda::UnplacedBox>::Construct(
    const Precision x, const Precision y, const Precision z) const;

} // End cxx namespace

#endif

} // End global namespace
