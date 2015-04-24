/// \file PlacedHype.cpp
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/PlacedHype.h"
#include "volumes/Hype.h"
#include "base/Global.h"
#include "base/AOS3D.h"
#include "base/SOA3D.h"
#include "backend/Backend.h"

//#if defined(VECGEOM_BENCHMARK) && defined(VECGEOM_ROOT)
#ifdef VECGEOM_ROOT
#include "TGeoHype.h"
#endif
#include <cassert>

#ifdef VECGEOM_USOLIDS
#include "UBox.hh"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Hype.hh"
#endif

namespace VECGEOM_NAMESPACE {

#ifdef VECGEOM_BENCHMARK

VPlacedVolume const* PlacedHype::ConvertToUnspecialized() const {
    std::cout<<"Convert VEC*********\n";
    return new SimpleHype(GetLabel().c_str(), logical_volume(),transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedHype::ConvertToRoot() const {
    std::cout<<"Convert ROOT*********\n";
   return new TGeoHype(GetLabel().c_str(), GetRmin(), GetStIn(), GetRmax(), GetStOut(), GetDz());
    
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedHype::ConvertToUSolids() const {
  //assert(0 && "Hype unsupported for USolids.");
  //return NULL;
 std::cerr << "**************************************************************\n";
    std::cerr << "WARNING: Hyperboloid unsupported for USolids.; returning a box\n";
    std::cerr << "**************************************************************\n";
    return new UBox("",10,10,10);
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedHype::ConvertToGeant4() const {
return new G4Hype(GetLabel().c_str(),GetRmin(),GetRmax(),
                                      GetStIn()*kDegToRad, GetStOut()*kDegToRad,
                                      GetDz());
}
#endif

#endif // VECGEOM_BENCHMARK

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void PlacedHype_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedHype::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  PlacedHype_CopyToGpu(logical_volume, transformation, this->id(),
                             gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedHype::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedHype>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void PlacedHype_ConstructOnGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::SimpleHype(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
    NULL,
    id
  );
}

void PlacedHype_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  PlacedHype_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation,
                                            id, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom
