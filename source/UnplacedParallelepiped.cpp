/// \file UnplacedParallelepiped.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/UnplacedParallelepiped.h"

#include "management/VolumeFactory.h"
#include "volumes/SpecializedParallelepiped.h"
#include "volumes/utilities/GenerationUtilities.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
UnplacedParallelepiped::UnplacedParallelepiped(
    Vector3D<Precision> const &dimensions,
    const Precision alpha,  const Precision theta, const Precision phi)
   : fDimensions(dimensions),
fAlpha(0),
fTheta(0),
fPhi(0),
fTanAlpha(0),
fTanThetaSinPhi(0),
fTanThetaCosPhi(0)
{
  SetAlpha(alpha);
  SetThetaAndPhi(theta, phi);
}

VECGEOM_CUDA_HEADER_BOTH
UnplacedParallelepiped::UnplacedParallelepiped(
    const Precision x, const Precision y, const Precision z,
    const Precision alpha, const Precision theta, const Precision phi)
   : fDimensions(x, y, z),
fAlpha(0),
fTheta(0),
fPhi(0),
fTanAlpha(0),
fTanThetaSinPhi(0),
fTanThetaCosPhi(0)
{
  SetAlpha(alpha);
  SetThetaAndPhi(theta, phi);
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedParallelepiped::SetAlpha(const Precision alpha) {
  fAlpha = alpha;
  fTanAlpha = tan(kDegToRad*alpha);
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedParallelepiped::SetTheta(const Precision theta) {
  SetThetaAndPhi(theta, fPhi);
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedParallelepiped::SetPhi(const Precision phi) {
  SetThetaAndPhi(fTheta, phi);
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedParallelepiped::SetThetaAndPhi(const Precision theta,
                                            const Precision phi) {
  fTheta = theta;
  fPhi = phi;
  fTanThetaCosPhi = tan(kDegToRad*theta)*cos(kDegToRad*phi);
  fTanThetaSinPhi = tan(kDegToRad*theta)*sin(kDegToRad*phi);
}

void UnplacedParallelepiped::Print() const {
  printf("UnplacedParallelepiped {%.2f, %.2f, %.2f, %.2f, %.2f, %.2f}",
         GetX(), GetY(), GetZ(), GetTanAlpha(), GetTanThetaCosPhi(),
         GetTanThetaSinPhi());
}

void UnplacedParallelepiped::Print(std::ostream &os) const {
  os << "UnplacedParallelepiped {" << GetX() << ", " << GetY() << ", " << GetZ()
     << ", " << GetTanAlpha() << ", " << GetTanThetaCosPhi() << ", "
     << GetTanThetaSinPhi();
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedParallelepiped::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {

  return CreateSpecializedWithPlacement<SpecializedParallelepiped<transCodeT, rotCodeT> >(
#ifdef VECGEOM_NVCC
      logical_volume, transformation, id, placement); // TODO: add bounding box?
#else
      logical_volume, transformation, placement);
#endif


}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedParallelepiped::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {
  return VolumeFactory::CreateByTransformation<
      UnplacedParallelepiped>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedParallelepiped::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
   return CopyToGpuImpl<UnplacedParallelepiped>(in_gpu_ptr, GetX(), GetY(), GetZ(), fAlpha, fTheta, fPhi);
}

DevicePtr<cuda::VUnplacedVolume> UnplacedParallelepiped::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedParallelepiped>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedParallelepiped>::SizeOf();
template void DevicePtr<cuda::UnplacedParallelepiped>::Construct(
    const Precision x, const Precision y, const Precision z,
    const Precision alpha, const Precision theta, const Precision phi) const;

} // End cxx namespace

#endif

} // End global namespace
