/// @file PlacedSphere.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/PlacedSphere.h"
#include "volumes/Sphere.h"
#include "base/Global.h"
#include "base/AOS3D.h"
#include "base/SOA3D.h"
#include "backend/Backend.h"

#ifdef VECGEOM_USOLIDS
#include "USphere.hh"
#endif

#ifdef VECGEOM_ROOT
#include "TGeoSphere.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Sphere.hh"
#endif

namespace VECGEOM_NAMESPACE {

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedSphere::ConvertToUnspecialized() const {
  return new SimpleSphere(GetLabel().c_str(), logical_volume(),
                                  transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedSphere::ConvertToRoot() const {
  return new TGeoSphere(GetLabel().c_str(),GetInnerRadius(),GetOuterRadius(),
                                      GetStartThetaAngle()*kRadToDeg,(GetStartThetaAngle()+GetDeltaThetaAngle())*kRadToDeg,
                                        GetStartPhiAngle()*kRadToDeg, (GetStartPhiAngle()+GetDeltaPhiAngle())*kRadToDeg);
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedSphere::ConvertToUSolids() const {

return new USphere(GetLabel().c_str(),GetInnerRadius(),GetOuterRadius(),
                                      GetStartPhiAngle(), GetDeltaPhiAngle(),
                                      GetStartThetaAngle(),GetDeltaThetaAngle());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedSphere::ConvertToGeant4() const {
return new G4Sphere(GetLabel().c_str(),GetInnerRadius(),GetOuterRadius(),
                                      GetStartPhiAngle(), GetDeltaPhiAngle(),
                                      GetStartThetaAngle(),GetDeltaThetaAngle());
}
#endif

#endif // VECGEOM_NVCC

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void PlacedSphere_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedSphere::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  vecgeom::PlacedSphere_CopyToGpu(logical_volume, transformation, this->id(),
                                 gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedSphere::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedSphere>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void PlacedSphere_ConstructOnGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::SimpleSphere(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
    NULL,
    id
  );
}

void PlacedSphere_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  PlacedSphere_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation,
                                                id, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom
